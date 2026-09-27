# slms

Small experiments comparing small language models (SLMs). Each experiment is a self-contained Jupyter notebook that runs on Google Colab's free GPU.

## Experiments

| Folder | Question | Open |
|---|---|---|
| [`experiments/2026-09-slm-gsm8k`](experiments/2026-09-slm-gsm8k) | Qwen3-1.7B vs SmolLM3-3B vs Phi-4-mini on GSM8K grade-school maths | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-slm-gsm8k/notebook.ipynb) |
| [`experiments/2026-09-laya-cuad`](experiments/2026-09-laya-cuad) | Laya (421M decision model) vs a local SLM on CUAD's 41 yes/no clause questions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-cuad/notebook.ipynb) |
| [`experiments/2026-09-laya-vs-phi4-boolq`](experiments/2026-09-laya-vs-phi4-boolq) | Laya (421M decision model) vs Phi-4-mini on BoolQ yes/no reading comprehension | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-vs-phi4-boolq/notebook.ipynb) |
| [`experiments/2026-09-laya-finetune-boolq`](experiments/2026-09-laya-finetune-boolq) | Can Laya fine-tuned on 3k BoolQ examples beat zero-shot Phi-4-mini? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-finetune-boolq/notebook.ipynb) |

## Layout

```
experiments/
  YYYY-MM-<slug>/
    notebook.ipynb     # runnable top to bottom
    requirements.txt   # pinned versions
    README.md          # question, dataset, how to run, results
    figures/           # charts saved by the notebook
```
