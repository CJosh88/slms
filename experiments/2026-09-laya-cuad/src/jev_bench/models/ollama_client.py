from __future__ import annotations

import os
import time

import httpx

from jev_bench.prompts import SYSTEM, build_user

from .base import Prediction, parse_yes_no

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}},
    "required": ["answer"],
}


class OllamaClient:
    def __init__(self, model: str, temperature: float = 0.0, num_ctx: int = 8192, timeout_s: float = 300, host: str | None = None):
        self.name = f"ollama:{model}"
        self.model = model
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)

    def answer(self, context: str, question: str) -> Prediction:
        start = time.perf_counter()
        try:
            r = self._http.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": build_user(context, question)},
                    ],
                    "format": ANSWER_SCHEMA,
                    "stream": False,
                    "options": self.options,
                },
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            return Prediction(parse_yes_no(raw), raw, time.perf_counter() - start)
        except httpx.HTTPError as e:
            return Prediction(None, "", time.perf_counter() - start, error=str(e))
