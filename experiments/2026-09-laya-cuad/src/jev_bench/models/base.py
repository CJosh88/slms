from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Prediction:
    answer: bool | None  # None = unparseable / error
    raw: str
    latency_s: float
    error: str | None = None


class ModelClient(Protocol):
    """Answers one yes/no question about one context chunk.

    Clients may also define `answer_many(contexts, question) -> list[Prediction]` to score all
    chunks of a contract at once; the runner uses it when present.
    """

    name: str

    def answer(self, context: str, question: str) -> Prediction: ...


def parse_yes_no(text: str) -> bool | None:
    t = text.strip().strip('"').lower()
    if t.startswith("{"):
        import json

        try:
            t = str(json.loads(t).get("answer", "")).lower()
        except (ValueError, AttributeError):
            pass
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return None
