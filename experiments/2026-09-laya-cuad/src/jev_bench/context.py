"""Long-contract handling. Many CUAD contracts exceed small-model context windows.

- truncate:       keep the first `max_chars` characters (one call per example)
- chunk_any_yes:  split into overlapping chunks, answer Yes if any chunk says Yes
"""

from __future__ import annotations


def chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    return [text[i : i + max_chars] for i in range(0, len(text), step) if text[i : i + max_chars].strip()]


def contexts_for(text: str, strategy: str, max_chars: int, overlap: int = 0) -> list[str]:
    if strategy == "truncate":
        return [text[:max_chars]]
    if strategy == "chunk_any_yes":
        return chunk_text(text, max_chars, overlap)
    raise ValueError(f"Unknown context strategy: {strategy}")
