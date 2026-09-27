"""Laya client (https://huggingface.co/convaiinnovations/laya).

Laya is a non-autoregressive decision model (ModernBERT-large, 512-token window), not a chat
LLM. It takes a state plus typed questions and returns calibrated probabilities in one forward
pass. Each CUAD question is asked as a `noul` (yes/no) question; the answer is Yes when
P(true) >= `threshold`. The shared chat prompt in `prompts.py` doesn't apply here.
"""

from __future__ import annotations

import time

from .base import Prediction

QID = "clause"
_ANSWER_SUFFIX = " Answer Yes or No."


def instructions_for(question: str) -> str:
    # The CUAD question ends with an instruction aimed at generative models; Laya has typed options.
    return question.removesuffix(_ANSWER_SUFFIX).strip()


class LayaClient:
    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        device: str | None = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        max_len: int | None = None,
        fast: bool = False,
    ):
        import laya

        self.name = f"laya:{model.rsplit('/', 1)[-1]}" + (f"-{subfolder}" if subfolder else "")
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_len = max_len
        self.agent = laya.load(model, device=device, subfolder=subfolder, fast=fast)

    def answer(self, context: str, question: str) -> Prediction:
        return self.answer_many([context], question)[0]

    def answer_many(self, contexts: list[str], question: str) -> list[Prediction]:
        """Score every chunk of a contract in shared forward passes."""
        questions = {QID: {"type": "noul", "instructions": instructions_for(question)}}
        start = time.perf_counter()
        try:
            results = self.agent.predict_batch(
                contexts, questions, batch_size=self.batch_size, max_len=self.max_len
            )
        except (RuntimeError, ValueError) as e:
            per = (time.perf_counter() - start) / max(1, len(contexts))
            return [Prediction(None, "", per, error=str(e)) for _ in contexts]
        per = (time.perf_counter() - start) / max(1, len(contexts))
        preds = []
        for r in results:
            p_yes = r["answers"][QID]["noul"]
            preds.append(Prediction(p_yes >= self.threshold, f"p_yes={p_yes:.4f}", per))
        return preds
