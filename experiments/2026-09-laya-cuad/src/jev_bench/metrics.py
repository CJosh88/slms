from __future__ import annotations

from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _prf(y_true: list[bool], y_pred: list[bool]) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def summarize(records: list[dict]) -> dict:
    """records: dicts with category, label (bool), pred (bool|None), latency_s.

    Unparseable predictions (pred=None) count as wrong (treated as the opposite of the label).
    """
    y_true = [r["label"] for r in records]
    y_pred = [r["pred"] if r["pred"] is not None else (not r["label"]) for r in records]

    by_cat: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r, p in zip(records, y_pred):
        by_cat[r["category"]][0].append(r["label"])
        by_cat[r["category"]][1].append(p)
    per_category = {c: _prf(t, p) for c, (t, p) in sorted(by_cat.items())}

    macro = {k: sum(m[k] for m in per_category.values()) / len(per_category) for k in ("precision", "recall", "f1")} if per_category else {}
    latencies = [r["latency_s"] for r in records]
    return {
        "micro": _prf(y_true, y_pred) if records else {},
        "macro": macro,
        "unparseable": sum(r["pred"] is None for r in records),
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "per_category": per_category,
    }
