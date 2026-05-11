"""
Tracked OpenAI client — drop-in replacement for `openai.OpenAI`.

Usage:
    from carbonwatcher.llm.openai_wrapper import TrackedOpenAI

    client = TrackedOpenAI(experiment="my-rag-pipeline")

    # Exact same API as openai.OpenAI — nothing else changes
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Explain transformers"}],
    )
    print(response.choices[0].message.content)

    client.print_summary()
"""

from __future__ import annotations

import time
from typing import Any, Optional, List

from carbonwatcher.llm.tracker import LLMTracker


class _TrackedCompletions:
    """Wraps openai client.chat.completions and intercepts .create()."""

    def __init__(self, raw_completions, tracker: LLMTracker, tags: List[str]):
        self._raw = raw_completions
        self._tracker = tracker
        self._tags = tags

    def create(self, **kwargs) -> Any:
        stream_mode = kwargs.get("stream", False)
        t0 = time.perf_counter()

        if stream_mode:
            return self._stream_create(**kwargs)

        response = self._raw.create(**kwargs)
        latency = time.perf_counter() - t0

        usage = getattr(response, "usage", None)
        input_tokens  = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        self._tracker.record(
            model=kwargs.get("model", "gpt-4o"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            tags=self._tags,
            metadata={
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "temperature": kwargs.get("temperature"),
            },
        )
        return response

    def _stream_create(self, **kwargs):
        """Generator wrapper that tracks tokens after the stream exhausts."""
        t0 = time.perf_counter()
        stream = self._raw.create(**kwargs)
        input_tokens = output_tokens = 0

        for chunk in stream:
            # OpenAI streams usage in the final chunk when stream_options.include_usage=True
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens  = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
            yield chunk

        latency = time.perf_counter() - t0
        self._tracker.record(
            model=kwargs.get("model", "gpt-4o"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            tags=self._tags + ["stream"],
        )


class _TrackedChat:
    def __init__(self, raw_chat, tracker: LLMTracker, tags: List[str]):
        self.completions = _TrackedCompletions(raw_chat.completions, tracker, tags)

    def __getattr__(self, name):
        return getattr(self._raw_chat, name)


class TrackedOpenAI:
    """
    Drop-in replacement for `openai.OpenAI`.
    Every chat.completions.create() call is automatically tracked.

    Parameters
    ----------
    experiment   : Name for grouping calls in the ledger.
    ledger_path  : JSON file to persist LLM call records.
    tags         : Default tags applied to every call.
    **client_kwargs : Passed directly to openai.OpenAI() (api_key, timeout, etc.)
    """

    def __init__(
        self,
        experiment: str = "default",
        ledger_path: str = "carbonwatcher_llm_ledger.json",
        tags: Optional[List[str]] = None,
        **client_kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._client = openai.OpenAI(**client_kwargs)
        self.llm_tracker = LLMTracker(experiment=experiment, ledger_path=ledger_path, tags=tags or [])
        self.chat = _TrackedChat(self._client.chat, self.llm_tracker, tags or [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def print_summary(self):
        from carbonwatcher.llm.report import print_llm_report
        print_llm_report(self.llm_tracker, experiment=self.llm_tracker.experiment)
