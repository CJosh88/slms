# Laya vs a local SLM on CUAD clause detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-cuad/notebook.ipynb)

Benchmarks **[Laya](https://huggingface.co/convaiinnovations/laya)** against a **local SLM (Ollama)** on [CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad). The task is 41 yes/no clause-presence questions for each of 510 contracts, giving 20,910 examples.

> jev signup is closed, so Laya takes its place. Laya (Apache-2.0, 421M params, ModernBERT-large) is a non-autoregressive decision model, not a chat LLM: it scores typed questions against a text in one forward pass and returns calibrated probabilities. Each CUAD question is asked as a `noul` (yes/no) question and counted as Yes when `P(true) >= threshold`.

## Setup
Run these from this folder (`experiments/2026-09-laya-cuad`).

```powershell
uv sync --extra laya   # the laya extra pulls in torch + transformers; plain `uv sync` is enough for Ollama only
copy .env.example .env
uv run python scripts/download_cuad.py      # raw CUAD files (PDFs skipped)
uv run python scripts/build_cuad_yesno.py   # -> data/processed/cuad_yesno.jsonl
uv run pytest
```

## How the 41 yes/no questions are built
CUAD has 41 categories. Only 33 of them have native Yes/No answers; the other 8 are extraction fields such as Parties, dates and Governing Law. So every category is scored as clause presence:
- **Yes** when CUAD annotated at least one span for that category (`is_impossible == false`)
- **No** otherwise

The prompt combines the category name with CUAD's own definition of it. Gold spans are kept in the JSONL for later evidence scoring.

`datasets.load_dataset("theatticusproject/cuad")` can't be used. HF's automatic Parquet conversion is a single `text` column, so the raw files are downloaded with `huggingface_hub` instead.

## Running
```powershell
# local SLM (install Ollama first: https://ollama.com, then)
ollama pull qwen2.5:3b
uv run python -m jev_bench.runner --model ollama --limit 20

# Laya (downloads ~1.7 GB of weights on first run; a GPU helps a lot)
uv run python -m jev_bench.runner --model laya --limit 20
```

To run Laya on a Colab GPU without setting up this repo, use `notebook.ipynb`. It installs its own packages, downloads CUAD, and runs the same evaluation on its own.
Predictions are cached in `results/<model>/preds.jsonl`, so reruns resume where they stopped. Metrics go to `results/<model>/metrics.json`: micro and macro P/R/F1, accuracy, latency, and a per-category breakdown.

## Fairness
Both models get the same question text and the same long-contract strategy (`context.py`, set in `configs/benchmark.yaml`). Many contracts exceed small-model context windows. The default `chunk_any_yes` strategy splits a contract into overlapping chunks and answers Yes if any chunk says Yes.

Two differences are forced by the models themselves:
- **Prompt.** Ollama gets the chat prompt from `prompts.py`. Laya has no chat prompt; it gets the question as `noul` instructions plus the raw chunk.
- **Chunk size.** Laya's 512-token window holds only about 320 tokens of contract after the question is encoded. So `models.laya.context` overrides `max_chars` to 1200 (with 200 overlap) instead of the shared 24000. Laya therefore sees roughly 20x more chunks per contract. Since any Yes chunk makes the answer Yes, expect higher recall and lower precision; `threshold` is the setting to tune.

## Layout
```
configs/benchmark.yaml     run settings
scripts/                   download + preprocessing
notebook.ipynb             standalone Colab notebook for Laya
src/jev_bench/data/cuad.py CUAD parsing -> Example objects
src/jev_bench/models/      ModelClient protocol, Ollama client, Laya client
src/jev_bench/runner.py    resumable evaluation loop
src/jev_bench/metrics.py   scoring
data/raw, data/processed   (gitignored)
results/                   (gitignored)
```
