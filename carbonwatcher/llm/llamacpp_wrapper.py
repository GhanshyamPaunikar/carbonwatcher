"""
TrackedLlamaCpp — wrapper for llama-cpp-python (local GGUF models).

llama-cpp-python reports exact token counts in every response.
Energy is estimated from model size + local hardware.

Usage:
    from carbonwatcher.llm.llamacpp_wrapper import TrackedLlamaCpp

    tracker = TrackedLlamaCpp(
        model_path="./models/llama-3.2-3b.Q4_K_M.gguf",
        experiment="local-inference",
        region="eu-west-1",
    )

    # Chat completion (OpenAI-compatible)
    response = tracker.chat([{"role": "user", "content": "What is 2+2?"}])
    print(response["choices"][0]["message"]["content"])

    # Raw completion
    response = tracker.complete("The capital of France is")
    print(response["choices"][0]["text"])

    tracker.print_summary()
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional, List

from carbonwatcher.llm.tracker import LLMTracker
from carbonwatcher.llm.tokenizer import count_tokens, count_messages


# Energy per 1k tokens by GGUF quantization / model size (Wh)
# Based on llama.cpp community benchmarks (CPU + GPU mixed inference)
# Input prefill ~4× cheaper than autoregressive generation
_GGUF_ENERGY_BY_SIZE = {
    # (param_billions_threshold, wh_per_1k_input, wh_per_1k_output)
    1:  (0.003, 0.012),
    3:  (0.006, 0.024),
    7:  (0.010, 0.040),
    13: (0.018, 0.072),
    30: (0.040, 0.160),
    70: (0.090, 0.360),
}

_DEFAULT_ENERGY = (0.010, 0.040)


def _estimate_energy_from_path(model_path: str) -> tuple[float, float]:
    """Guess model size from filename and return energy estimate."""
    name = os.path.basename(model_path).lower()
    # Look for size hints: "7b", "13b", "70b", "3b", etc.
    import re
    m = re.search(r"(\d+(?:\.\d+)?)b", name)
    if m:
        params = float(m.group(1))
        for threshold, vals in sorted(_GGUF_ENERGY_BY_SIZE.items()):
            if params <= threshold:
                return vals
    return _DEFAULT_ENERGY


class TrackedLlamaCpp:
    """
    Wrapper for llama-cpp-python local inference.
    Token counts come directly from llama.cpp's response metadata.

    Parameters
    ----------
    model_path  : Path to the .gguf model file.
    experiment  : Label for grouping calls in the ledger.
    ledger_path : JSON file to persist LLM call records.
    region      : Your local grid region for carbon intensity.
    tags        : Default tags for every call.
    **llama_kwargs : Extra args passed to Llama() (n_ctx, n_gpu_layers, etc.).
    """

    def __init__(
        self,
        model_path: str,
        experiment: str = "default",
        ledger_path: str = "carbonwatcher_llm_ledger.json",
        region: str = "us-east-1",
        tags: Optional[List[str]] = None,
        **llama_kwargs,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        self._llm = Llama(model_path=model_path, **llama_kwargs)
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self._energy = _estimate_energy_from_path(model_path)

        self.llm_tracker = LLMTracker(
            experiment=experiment, ledger_path=ledger_path, tags=tags or []
        )
        self._default_tags = tags or []

        from carbonwatcher.core.carbon import CarbonEstimator
        self._carbon_est = CarbonEstimator(region=region)

    # ── chat (OpenAI-compatible) ──────────────────────────────────────────────

    def chat(self, messages: list[dict], **kwargs) -> Any:
        """
        OpenAI-compatible chat_completion. Token counts from llama.cpp response.
        """
        t0 = time.perf_counter()
        response = self._llm.create_chat_completion(messages=messages, **kwargs)
        latency = time.perf_counter() - t0

        input_tokens, output_tokens, token_source = self._extract_tokens(
            response, fallback_text=self._messages_text(messages),
            fallback_output=response.get("choices", [{}])[0]
                                  .get("message", {}).get("content", ""),
        )
        self._record(input_tokens, output_tokens, latency, token_source)
        return response

    # ── raw completion ────────────────────────────────────────────────────────

    def complete(self, prompt: str, **kwargs) -> Any:
        """Raw text completion. Token counts from llama.cpp response."""
        t0 = time.perf_counter()
        response = self._llm(prompt, **kwargs)
        latency = time.perf_counter() - t0

        output_text = response.get("choices", [{}])[0].get("text", "")
        input_tokens, output_tokens, token_source = self._extract_tokens(
            response, fallback_text=prompt, fallback_output=output_text,
        )
        self._record(input_tokens, output_tokens, latency, token_source)
        return response

    # ── token extraction ──────────────────────────────────────────────────────

    def _extract_tokens(self, response: dict, fallback_text: str,
                        fallback_output: str) -> tuple[int, int, str]:
        usage = response.get("usage", {})
        input_tokens  = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        if input_tokens > 0 or output_tokens > 0:
            return input_tokens, output_tokens, "llamacpp_reported"

        # Fallback: count ourselves
        input_tokens, src = count_tokens(fallback_text, model=self.model_name)
        output_tokens, _  = count_tokens(fallback_output, model=self.model_name)
        return input_tokens, output_tokens, f"counted:{src}"

    def _messages_text(self, messages: list[dict]) -> str:
        return " ".join(m.get("content", "") for m in messages
                        if isinstance(m.get("content"), str))

    # ── internal record ───────────────────────────────────────────────────────

    def _record(self, input_tokens, output_tokens, latency, token_source):
        wh_in, wh_out = self._energy
        total_wh = (wh_in * input_tokens / 1000) + (wh_out * output_tokens / 1000)
        intensity, int_source = self._carbon_est._resolver.resolve()
        carbon_g = total_wh / 1000.0 * intensity

        from carbonwatcher.llm.tracker import LLMCallRecord
        import uuid
        from datetime import datetime, timezone

        record = LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            experiment=self.llm_tracker.experiment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=self.model_name,
            provider="llamacpp_local",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            total_energy_wh=total_wh,
            carbon_g=carbon_g,
            carbon_kg=carbon_g / 1000,
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=0.0,
            grid_intensity_gco2_kwh=intensity,
            intensity_source=int_source,
            latency_s=latency,
            tags=self._default_tags + [f"token_source:{token_source}"],
            metadata={"token_source": token_source, "model_path": self.model_path},
        )
        self.llm_tracker._records.append(record)
        self.llm_tracker._save()

    def print_summary(self):
        from carbonwatcher.llm.report import print_llm_report
        print_llm_report(self.llm_tracker)

    def __getattr__(self, name):
        """Proxy everything else directly to the Llama instance."""
        return getattr(self._llm, name)
