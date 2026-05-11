"""
TrackedHuggingFace — wrapper for local HuggingFace transformers models.

Uses the model's own tokenizer for exact token counts.
Energy is estimated from hardware profiling during inference.

Usage:
    from carbonwatcher.llm.huggingface_wrapper import TrackedHuggingFace

    # Wraps any HuggingFace pipeline
    tracker = TrackedHuggingFace(
        model_name="microsoft/phi-3-mini-4k-instruct",
        experiment="my-local-inference",
        region="eu-west-1",
    )

    result = tracker.generate("Explain neural networks in simple terms")
    print(result)

    tracker.print_summary()

    # Or wrap an existing pipeline:
    from transformers import pipeline
    pipe = pipeline("text-generation", model="gpt2")
    tracker = TrackedHuggingFace.from_pipeline(pipe, experiment="gpt2-test")
    result = tracker.generate("Once upon a time")
"""

from __future__ import annotations

import time
from typing import Any, Optional, List

from carbonwatcher.llm.tracker import LLMTracker
from carbonwatcher.llm.tokenizer import count_tokens


# Energy per 1k tokens by model family (Wh) — from HuggingFace/Luccioni 2023
# Input prefill is cheaper than autoregressive generation (~4× ratio)
_HF_MODEL_ENERGY = {
    "gpt2":            (0.0006,  0.0024),
    "gpt-neo":         (0.0010,  0.0040),
    "gpt-j":           (0.0025,  0.0100),
    "llama":           (0.0120,  0.0480),
    "mistral":         (0.0100,  0.0400),
    "phi":             (0.0040,  0.0160),
    "gemma":           (0.0080,  0.0320),
    "falcon":          (0.0100,  0.0400),
    "bloom":           (0.0200,  0.0800),
    "opt":             (0.0060,  0.0240),
    "t5":              (0.0005,  0.0020),
    "bart":            (0.0005,  0.0020),
    "bert":            (0.0003,  0.0003),   # encoder-only, no generation
    "roberta":         (0.0003,  0.0003),
    "default":         (0.0100,  0.0400),
}


def _lookup_hf_energy(model_name: str) -> tuple[float, float]:
    name = model_name.lower().split("/")[-1]   # strip org prefix
    for key, vals in _HF_MODEL_ENERGY.items():
        if key in name:
            return vals
    return _HF_MODEL_ENERGY["default"]


class TrackedHuggingFace:
    """
    Wrapper for local HuggingFace transformers inference.
    Uses the model's tokenizer for exact token counts.

    Parameters
    ----------
    model_name  : HuggingFace model ID (e.g. "gpt2", "microsoft/phi-3-mini-4k-instruct").
    experiment  : Label for grouping calls in the ledger.
    ledger_path : JSON file to persist LLM call records.
    region      : Your local grid region for carbon intensity.
    device      : "cpu", "cuda", "mps" (Apple Silicon), or device index.
    tags        : Default tags for every call.
    **pipeline_kwargs : Extra args passed to transformers.pipeline().
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        experiment: str = "default",
        ledger_path: str = "carbonwatcher_llm_ledger.json",
        region: str = "us-east-1",
        device: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **pipeline_kwargs,
    ):
        try:
            from transformers import pipeline, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        pipe_kwargs = {"model": model_name, **pipeline_kwargs}
        if device is not None:
            pipe_kwargs["device"] = device
        self._pipe = pipeline("text-generation", **pipe_kwargs)

        self.llm_tracker = LLMTracker(
            experiment=experiment, ledger_path=ledger_path, tags=tags or []
        )
        self._default_tags = tags or []

        from carbonwatcher.core.carbon import CarbonEstimator
        self._carbon_est = CarbonEstimator(region=region)

    @classmethod
    def from_pipeline(
        cls,
        pipe,
        experiment: str = "default",
        ledger_path: str = "carbonwatcher_llm_ledger.json",
        region: str = "us-east-1",
        tags: Optional[List[str]] = None,
    ) -> "TrackedHuggingFace":
        """Wrap an already-loaded HuggingFace pipeline."""
        instance = object.__new__(cls)
        instance.model_name = getattr(pipe.model.config, "_name_or_path",
                                      type(pipe.model).__name__)
        instance._pipe = pipe
        instance._tokenizer = pipe.tokenizer
        instance.llm_tracker = LLMTracker(
            experiment=experiment, ledger_path=ledger_path, tags=tags or []
        )
        instance._default_tags = tags or []
        from carbonwatcher.core.carbon import CarbonEstimator
        instance._carbon_est = CarbonEstimator(region=region)
        return instance

    # ── generate ──────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """
        Generate text from prompt. Returns generated text (not including prompt).
        Token counts are exact — uses the model's own tokenizer.
        """
        t0 = time.perf_counter()
        outputs = self._pipe(prompt, max_new_tokens=max_new_tokens,
                             return_full_text=False, **kwargs)
        latency = time.perf_counter() - t0

        generated_text = outputs[0]["generated_text"] if outputs else ""

        # Exact token counts via the model's tokenizer
        input_tokens  = len(self._tokenizer.encode(prompt, add_special_tokens=True))
        output_tokens = len(self._tokenizer.encode(generated_text, add_special_tokens=False))

        self._record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
            token_source="hf_tokenizer",
        )
        return generated_text

    def __call__(self, prompt: str, **kwargs) -> str:
        """Alias for generate — makes the tracker callable like a pipeline."""
        return self.generate(prompt, **kwargs)

    # ── internal ──────────────────────────────────────────────────────────────

    def _record(self, input_tokens, output_tokens, latency, token_source):
        wh_in, wh_out = _lookup_hf_energy(self.model_name)
        total_wh = (wh_in * input_tokens / 1000) + (wh_out * output_tokens / 1000)
        intensity, int_source = self._carbon_est._resolver.resolve()
        carbon_g = total_wh / 1000.0 * intensity

        from carbonwatcher.core.carbon import _compute_equivalents
        from carbonwatcher.llm.tracker import LLMCallRecord
        import uuid
        from datetime import datetime, timezone

        record = LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            experiment=self.llm_tracker.experiment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=self.model_name,
            provider="huggingface_local",
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
            metadata={"token_source": token_source, "model": self.model_name},
        )
        self.llm_tracker._records.append(record)
        self.llm_tracker._save()

    def print_summary(self):
        from carbonwatcher.llm.report import print_llm_report
        print_llm_report(self.llm_tracker)
