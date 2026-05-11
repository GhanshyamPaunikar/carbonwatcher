"""
TrackedOllama — drop-in wrapper for the ollama Python client.

Ollama runs models locally (Llama, Mistral, Gemma, Phi, etc.).
Token counts come directly from Ollama's response (it reports prompt_eval_count
and eval_count). Energy is estimated from local hardware + model size.

Usage:
    from carbonwatcher.llm.ollama_wrapper import TrackedOllama

    client = TrackedOllama(model="llama3.2", experiment="my-rag")

    response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
    print(response["message"]["content"])

    # or generate API:
    response = client.generate(prompt="Explain quantum computing")
    print(response["response"])

    client.print_summary()
"""

from __future__ import annotations

import time
from typing import Any, Optional, List

from carbonwatcher.llm.tracker import LLMTracker
from carbonwatcher.llm.tokenizer import count_tokens, count_messages


# Energy estimates for local models (Wh per 1k tokens output).
# Based on: GPU TDP × tokens/sec benchmarks from llama.cpp / ollama community.
# Input decoding is ~4× cheaper than generation (no autoregressive loop).
_LOCAL_MODEL_ENERGY = {
    # (wh_per_1k_input, wh_per_1k_output)
    "llama3.2":        (0.010,  0.040),
    "llama3.1":        (0.012,  0.048),
    "llama3":          (0.012,  0.048),
    "llama2":          (0.012,  0.048),
    "mistral":         (0.010,  0.040),
    "mixtral":         (0.025,  0.100),
    "gemma2":          (0.008,  0.032),
    "gemma":           (0.008,  0.032),
    "phi3":            (0.004,  0.016),
    "phi4":            (0.005,  0.020),
    "qwen2.5":         (0.010,  0.040),
    "qwen2":           (0.010,  0.040),
    "deepseek-r1":     (0.015,  0.060),
    "deepseek-coder":  (0.012,  0.048),
    "codellama":       (0.012,  0.048),
    "vicuna":          (0.012,  0.048),
    "orca-mini":       (0.005,  0.020),
    "neural-chat":     (0.010,  0.040),
    "default":         (0.012,  0.048),   # conservative fallback
}

# Local GPU grid intensity — uses local machine's grid, default US average
_LOCAL_GRID_INTENSITY = 386.0  # gCO2/kWh — overridable


def _lookup_local_energy(model: str) -> tuple[float, float]:
    """Match model name to energy estimate (prefix match)."""
    model_lower = model.lower().split(":")[0]   # strip tag e.g. "llama3.2:latest"
    for key, vals in _LOCAL_MODEL_ENERGY.items():
        if model_lower.startswith(key) or key.startswith(model_lower):
            return vals
    return _LOCAL_MODEL_ENERGY["default"]


class TrackedOllama:
    """
    Drop-in wrapper for the ollama Python client.
    Automatically reads token counts from Ollama's response metadata.

    Parameters
    ----------
    model       : Default Ollama model name (e.g. "llama3.2", "mistral").
    experiment  : Label for grouping calls in the ledger.
    ledger_path : JSON file to persist LLM call records.
    region      : Your local grid region for carbon intensity.
    host        : Ollama server URL (default: http://localhost:11434).
    tags        : Default tags applied to every call.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        experiment: str = "default",
        ledger_path: str = "carbonwatcher_llm_ledger.json",
        region: str = "us-east-1",
        host: str = "http://localhost:11434",
        tags: Optional[List[str]] = None,
    ):
        try:
            import ollama as _ollama
            self._ollama = _ollama
            self._client = _ollama.Client(host=host)
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")

        self.default_model = model
        self.llm_tracker = LLMTracker(
            experiment=experiment, ledger_path=ledger_path, tags=tags or []
        )
        self._region = region
        self._default_tags = tags or []

        # Override tracker's carbon estimator to use local grid
        from carbonwatcher.core.carbon import CarbonEstimator
        self._carbon_est = CarbonEstimator(region=region)

    # ── chat API ──────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict], model: Optional[str] = None, **kwargs) -> Any:
        """
        ollama.chat() equivalent — tracks tokens automatically.
        Token counts come from Ollama's response metadata.
        """
        model = model or self.default_model
        t0 = time.perf_counter()
        response = self._client.chat(model=model, messages=messages, **kwargs)
        latency = time.perf_counter() - t0

        input_tokens, output_tokens, token_source = self._extract_tokens_chat(
            response, messages, model
        )
        self._record(model, input_tokens, output_tokens, latency, token_source)
        return response

    # ── generate API ─────────────────────────────────────────────────────────

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> Any:
        """
        ollama.generate() equivalent — tracks tokens automatically.
        """
        model = model or self.default_model
        t0 = time.perf_counter()
        response = self._client.generate(model=model, prompt=prompt, **kwargs)
        latency = time.perf_counter() - t0

        input_tokens, output_tokens, token_source = self._extract_tokens_generate(
            response, prompt, model
        )
        self._record(model, input_tokens, output_tokens, latency, token_source)
        return response

    # ── token extraction ──────────────────────────────────────────────────────

    def _extract_tokens_chat(self, response, messages, model):
        """Read tokens from Ollama response, fall back to counting input messages."""
        resp_dict = response if isinstance(response, dict) else response.__dict__
        # Ollama reports these in the response object
        input_tokens  = resp_dict.get("prompt_eval_count", 0)
        output_tokens = resp_dict.get("eval_count", 0)

        if input_tokens > 0 or output_tokens > 0:
            return input_tokens, output_tokens, "ollama_reported"

        # Fallback: count the input messages ourselves
        input_tokens, src = count_messages(messages, model=model)
        # Estimate output from response text
        output_text = ""
        msg = resp_dict.get("message", {})
        if isinstance(msg, dict):
            output_text = msg.get("content", "")
        elif hasattr(msg, "content"):
            output_text = msg.content or ""
        output_tokens, _ = count_tokens(output_text, model=model)
        return input_tokens, output_tokens, f"counted:{src}"

    def _extract_tokens_generate(self, response, prompt, model):
        resp_dict = response if isinstance(response, dict) else response.__dict__
        input_tokens  = resp_dict.get("prompt_eval_count", 0)
        output_tokens = resp_dict.get("eval_count", 0)

        if input_tokens > 0 or output_tokens > 0:
            return input_tokens, output_tokens, "ollama_reported"

        input_tokens, src = count_tokens(prompt, model=model)
        output_text = resp_dict.get("response", "")
        output_tokens, _ = count_tokens(output_text, model=model)
        return input_tokens, output_tokens, f"counted:{src}"

    # ── internal record ───────────────────────────────────────────────────────

    def _record(self, model, input_tokens, output_tokens, latency, token_source):
        wh_in, wh_out = _lookup_local_energy(model)
        total_wh = (wh_in * input_tokens / 1000) + (wh_out * output_tokens / 1000)
        intensity, int_source = self._carbon_est._resolver.resolve()
        carbon_g = total_wh / 1000.0 * intensity

        from carbonwatcher.core.carbon import _compute_equivalents
        from carbonwatcher.llm.tracker import LLMCallResult, LLMCallRecord
        import uuid
        from datetime import datetime, timezone

        result = LLMCallResult(
            model_id=model,
            provider="ollama_local",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_energy_wh=wh_in * input_tokens / 1000,
            output_energy_wh=wh_out * output_tokens / 1000,
            total_energy_wh=total_wh,
            carbon_g=carbon_g,
            carbon_kg=carbon_g / 1000,
            grid_intensity_gco2_kwh=intensity,
            intensity_source=int_source,
            input_cost_usd=0.0,    # local = free
            output_cost_usd=0.0,
            total_cost_usd=0.0,
            equivalents=_compute_equivalents(carbon_g),
        )

        record = LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            experiment=self.llm_tracker.experiment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model,
            provider="ollama_local",
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
            metadata={"token_source": token_source},
        )
        self.llm_tracker._records.append(record)
        self.llm_tracker._save()
        return result

    def print_summary(self):
        from carbonwatcher.llm.report import print_llm_report
        print_llm_report(self.llm_tracker)

    def __getattr__(self, name):
        """Proxy anything else (list, show, pull, etc.) to the real client."""
        return getattr(self._client, name)
