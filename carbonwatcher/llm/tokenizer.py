"""Token counting utilities for local and online LLMs.

Priority order for counting:
  1. API response usage field  (most accurate — server-counted)
  2. HuggingFace tokenizer      (exact for HF models)
  3. tiktoken                   (exact for OpenAI/GPT family)
  4. character ÷ 4 estimate     (universal fallback, ~±15%)
"""

from typing import Optional, Union


# ── tiktoken (OpenAI tokenizer, works for GPT + many others) ─────────────────

def _tiktoken_count(text: str, model: str = "gpt-4o") -> Optional[int]:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 / Claude compatible
        return len(enc.encode(text))
    except ImportError:
        return None


# ── HuggingFace tokenizer ─────────────────────────────────────────────────────

def _hf_count(text: str, tokenizer) -> int:
    """Count tokens using a HuggingFace tokenizer object."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


# ── Fallback: character estimate ──────────────────────────────────────────────

def _char_estimate(text: str) -> int:
    """Rough estimate: ~4 chars per token (works across most tokenizers)."""
    return max(1, len(text) // 4)


# ── Public API ────────────────────────────────────────────────────────────────

def count_tokens(
    text: str,
    model: str = "gpt-4o",
    tokenizer=None,
) -> tuple[int, str]:
    """
    Count tokens in text. Returns (count, method_used).

    Parameters
    ----------
    text      : The string to count.
    model     : Model name hint (used to pick the right tiktoken encoding).
    tokenizer : A HuggingFace tokenizer instance (overrides tiktoken).

    Returns
    -------
    (token_count, source)  where source is one of:
        "hf_tokenizer" | "tiktoken" | "char_estimate"
    """
    if tokenizer is not None:
        return _hf_count(text, tokenizer), "hf_tokenizer"

    n = _tiktoken_count(text, model)
    if n is not None:
        return n, "tiktoken"

    return _char_estimate(text), "char_estimate"


def count_messages(
    messages: list[dict],
    model: str = "gpt-4o",
    tokenizer=None,
) -> tuple[int, str]:
    """
    Count tokens across a list of chat messages (role+content dicts).
    Adds ~4 tokens per message for role/formatting overhead.
    """
    total = 0
    source = "char_estimate"
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Anthropic-style content blocks
            content = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        n, src = count_tokens(content, model=model, tokenizer=tokenizer)
        total += n + 4   # per-message overhead
        source = src     # all messages use the same method
    return total, source
