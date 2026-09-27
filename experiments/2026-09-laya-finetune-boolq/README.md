# Can fine-tuned Laya beat Phi-4-mini on BoolQ?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-finetune-boolq/notebook.ipynb)

## Question

In [`2026-09-laya-vs-phi4-boolq`](../2026-09-laya-vs-phi4-boolq), zero-shot Laya (421M) scored 75.7% on the full BoolQ validation split (3,270 examples), against 81.4% for zero-shot Phi-4-mini-instruct (3.8B). If Laya is fine-tuned on 3,000 BoolQ training examples, does it beat an off-the-shelf Phi-4-mini?

## Setup

| | Examples | Source | Used for |
|---|---|---|---|
| Train | 3,000 | BoolQ `train`, random sample (seed 42) | Fine-tuning Laya |
| Calibration | 300 | BoolQ `train`, disjoint from the above | Temperature fitting and threshold choice for every model |
| Test | 3,270 | BoolQ `validation`, all of it | All reported numbers |

No validation example is used for training, calibration or tuning. Passages over 400 Laya tokens are truncated the same way for all models.

## Models

| Model | Params | Adaptation |
|---|---|---|
| Laya (zero-shot) | 421M | none; re-scored here as the before-fine-tuning baseline |
| Laya (fine-tuned) | 421M | 3 epochs on 3,000 examples |
| Phi-4-mini-instruct | 3.8B | none (zero-shot, same prompt as the previous experiment) |

## Method

- **Fine-tuning** is a single-T4 port of Laya's [official RLCD recipe](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb). Gaussian exploration noise is added to the answer logits and scored with a proper scoring rule. A policy-gradient loss plus a soft cross-entropy term trains the model, with the noise annealing from 0.4 to 0.1. The learning rates are 2.5e-5 (encoder) and 1e-4 (head), the effective batch is 64 and training runs 3 epochs. Training sequences are built with `laya.common.build_sequence`, so they match inference exactly. After training, a `noul` temperature is fitted on the calibration slice and the checkpoint is saved in the layout `laya.Agent` loads.
- **Phi-4-mini:** its 3,270 predictions are reused from the previous run's `results.csv`, committed here as [`prior_results.csv`](prior_results.csv). The notebook fetches the file from GitHub when it isn't present locally, and it only generates examples missing from the file.
- **Metrics:** accuracy with a 95% Wilson CI at threshold 0.5, plus accuracy at a threshold tuned on the calibration slice. Also F1, ROC-AUC, Brier score, ECE, ms per example and peak GPU memory. The headline is an **exact McNemar test** of fine-tuned Laya against Phi-4-mini on the 3,270 paired answers.

## How to run

1. Click **Open in Colab** above and choose **Runtime → Change runtime type → T4 GPU**.
2. Nothing to upload: `prior_results.csv` is fetched automatically, so Phi-4-mini isn't re-run.
3. **Runtime → Run all.** Fine-tuning takes about 10 minutes and Laya evaluation about 5. Predictions, training history and the checkpoint are cached, so a rerun after a disconnect resumes. Set `USE_DRIVE = True` to keep them on Google Drive.
4. Download the results zip and commit `figures/`, `results.csv` and `summary.csv` here. The ~850 MB checkpoint is not committed.

## Results

Not run yet.
