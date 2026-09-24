# slms

Small experiments comparing small language models (SLMs). Each experiment is a self-contained Jupyter notebook that runs on Google Colab's free GPU.

## Experiments

| Folder | Question | Open |
|---|---|---|
| [`experiments/2026-09-slm-gsm8k`](experiments/2026-09-slm-gsm8k) | Qwen3-1.7B vs SmolLM3-3B vs Phi-4-mini on GSM8K grade-school maths | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-slm-gsm8k/notebook.ipynb) |

## Layout

```
experiments/
  YYYY-MM-<slug>/
    notebook.ipynb     # runnable top to bottom
    requirements.txt   # pinned versions
    README.md          # question, dataset, how to run, results
    figures/           # charts saved by the notebook
```
