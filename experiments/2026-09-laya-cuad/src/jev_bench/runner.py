"""Run a model over the CUAD yes/no set. Resumable: predictions cached per example id.

    uv run python -m jev_bench.runner --model ollama --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from jev_bench import PROJECT_ROOT
from jev_bench.context import contexts_for
from jev_bench.data.cuad import load_cuad_yesno
from jev_bench.metrics import summarize
from jev_bench.models import get_client


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_chunks(client, chunks: list[str], question: str):
    """Yield one Prediction per chunk. Lazy for per-chunk clients so an early Yes stops the calls."""
    if hasattr(client, "answer_many"):
        yield from client.answer_many(chunks, question)
    else:
        for chunk in chunks:
            yield client.answer(chunk, question)


def run(model_name: str, cfg: dict, limit: int | None = None) -> dict:
    ds = cfg["dataset"]
    model_cfg = dict(cfg["models"][model_name])
    # A model may override the shared context settings when its window forces it (Laya: 512 tokens).
    ctx = {**cfg["context"], **model_cfg.pop("context", {})}
    client = get_client(model_name, model_cfg)
    examples = load_cuad_yesno(
        PROJECT_ROOT / ds["processed_path"],
        limit=limit if limit is not None else ds.get("limit"),
        categories_filter=ds.get("categories") or None,
        seed=ds.get("seed", 42),
    )

    out_dir = PROJECT_ROOT / cfg["output_dir"] / re.sub(r"[^\w.-]", "_", client.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "preds.jsonl"
    run_cfg = cfg.get("run", {})
    done: dict[str, dict] = {}
    if preds_path.exists() and not run_cfg.get("overwrite", False):
        with open(preds_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if run_cfg.get("retry_errors", True) and (rec["pred"] is None or rec["errors"]):
                    continue
                done[rec["id"]] = rec
    # Rewrite the file with only the kept records so retried ids aren't duplicated.
    with open(preds_path, "w", encoding="utf-8") as f:
        for rec in done.values():
            f.write(json.dumps(rec) + "\n")

    with open(preds_path, "a", encoding="utf-8") as f:
        for ex in tqdm(examples, desc=client.name):
            if ex.id in done:
                continue
            answer, latency, raws, errors = False, 0.0, [], []
            chunks = contexts_for(ex.context, ctx["strategy"], ctx["max_chars"], ctx.get("chunk_overlap", 0))
            for p in predict_chunks(client, chunks, ex.question):
                latency += p.latency_s
                raws.append(p.raw)
                if p.error:
                    errors.append(p.error)
                if p.answer is None:
                    answer = None if answer is False else answer
                elif p.answer:
                    answer = True
                    break
            rec = {
                "id": ex.id, "category": ex.category, "label": ex.label, "pred": answer,
                "latency_s": latency, "raw": raws, "errors": errors,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[ex.id] = rec

    wanted = {ex.id for ex in examples}
    summary = summarize([r for i, r in done.items() if i in wanted])
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ollama", "laya"])
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    summary = run(args.model, load_config(Path(args.config)), args.limit)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
