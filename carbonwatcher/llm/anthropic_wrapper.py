"""
TrackedAnthropic — drop-in replacement for anthropic.Anthropic.

Automatically tracks every messages.create() / stream() call:
  - Token counts  (from API response.usage — server-counted, exact)
  - Energy (Wh)
  - Carbon (gCO2e)
  - Cost (USD)
  - Latency (s)
  - Prompt-cache hits/misses

Usage:
    from carbonwatcher.llm.anthropic_wrapper import TrackedAnthropic

    client = TrackedAnthropic(experiment="my-app", print_each=True)

    # Identical to anthropic.Anthropic — nothing else changes
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Explain transformers"}],
    )
    print(response.content[0].text)

    client.print_summary()           # rich table of all calls
    client.llm_tracker.aggregate()  # dict of totals
"""

from __future__ import annotations

import time
from typing import Any, Optional, List

from carbonwatcher.llm.tracker import LLMTracker
from carbonwatcher.llm.tokenizer import count_messages


def _extract_usage(response) -> tuple[int, int, int, int]:
    """Pull token counts out of an Anthropic response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0, 0
    input_tokens  = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read    = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return input_tokens, output_tokens, cache_read, cache_created


class _TrackedMessages:
    """Wraps anthropic.messages and intercepts .create() and .stream()."""

    def __init__(self, raw_messages, tracker: LLMTracker,
                 tags: List[str], print_each: bool):
        self._raw = raw_messages
        self._tracker = tracker
        self._tags = tags
        self._print_each = print_each

    # ── non-streaming ─────────────────────────────────────────────────────────

    def create(self, **kwargs) -> Any:
        t0 = time.perf_counter()
        response = self._raw.create(**kwargs)
        latency = time.perf_counter() - t0

        input_tokens, output_tokens, cache_read, cache_created = _extract_usage(response)
        # Cache-read tokens consumed ~10% energy vs full prefill — track separately
        billable_input = max(input_tokens - cache_read, 0)

        result = self._tracker.record(
            model=kwargs.get("model", "claude-sonnet-4-6"),
            input_tokens=billable_input,
            output_tokens=output_tokens,
            latency_s=latency,
            tags=self._tags,
            metadata={
                "cache_read_tokens":    cache_read,
                "cache_created_tokens": cache_created,
                "total_input_tokens":   input_tokens,
                "max_tokens":           kwargs.get("max_tokens"),
                "stop_reason":          getattr(response, "stop_reason", None),
                "token_source":         "anthropic_api",
            },
        )
        if self._print_each:
            print(result.summary())
        return response

    # ── streaming ─────────────────────────────────────────────────────────────

    def stream(self, **kwargs):
        """Streaming context manager — tracks tokens after stream completes."""
        t0 = time.perf_counter()
        with self._raw.stream(**kwargs) as stream:
            yield from stream
        latency = time.perf_counter() - t0

        try:
            msg = stream.get_final_message()
            input_tokens, output_tokens, cache_read, cache_created = _extract_usage(msg)
            billable_input = max(input_tokens - cache_read, 0)
        except Exception:
            billable_input = output_tokens = cache_read = cache_created = 0

        result = self._tracker.record(
            model=kwargs.get("model", "claude-sonnet-4-6"),
            input_tokens=billable_input,
            output_tokens=output_tokens,
            latency_s=latency,
            tags=self._tags + ["stream"],
            metadata={
                "cache_read_tokens":    cache_read,
                "cache_created_tokens": cache_created,
                "token_source":         "anthropic_api",
            },
        )
        if self._print_each:
            print(result.summary())

    # ── token pre-counting (before sending) ───────────────────────────────────

    def count_tokens(self, messages: list[dict], model: str = "claude-sonnet-4-6") -> tuple[int, str]:
        """
        Estimate input tokens before making the API call.
        Uses tiktoken if available, else char/4 estimate.
        Returns (count, method).
        """
        return count_messages(messages, model=model)


class TrackedAnthropic:
    """
    Drop-in replacement for anthropic.Anthropic.

    Parameters
    ----------
    experiment   : Label for grouping calls in the ledger.
    ledger_path  : JSON file to persist records (default: carbonwatcher_llm_ledger.json).
    tags         : Default tags applied to every call.
    print_each   : Print a one-line carbon summary after every API call.
    **client_kwargs : Forwarded to anthropic.Anthropic() — api_key, timeout, etc.
    """

    def __init__(
        self,
        experiment: str = "default",
        ledger_path: str = "carbonwatcher_llm_ledger.json",
        tags: Optional[List[str]] = None,
        print_each: bool = False,
        **client_kwargs,
    ):
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        self._client = anthropic.Anthropic(**client_kwargs)
        self.llm_tracker = LLMTracker(
            experiment=experiment, ledger_path=ledger_path, tags=tags or []
        )
        self._print_each = print_each
        self.messages = _TrackedMessages(
            self._client.messages,
            self.llm_tracker,
            tags or [],
            print_each,
        )

    # ── convenience ───────────────────────────────────────────────────────────

    def print_summary(self):
        """Print a rich table of all tracked calls for this experiment."""
        from carbonwatcher.llm.report import print_llm_report
        print_llm_report(self.llm_tracker, experiment=self.llm_tracker.experiment)

    def aggregate(self) -> dict:
        """Return aggregate stats dict (total tokens, energy, carbon, cost)."""
        return self.llm_tracker.aggregate()

    # ── proxy everything else to the real client ──────────────────────────────

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
