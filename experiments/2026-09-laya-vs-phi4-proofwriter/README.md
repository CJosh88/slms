# Laya vs Phi-4-mini on ProofWriter: accuracy vs reasoning depth

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-vs-phi4-proofwriter/notebook.ipynb)

## Question

[BoolQ](../2026-09-laya-vs-phi4-boolq) asked whether Laya can *find* an answer in a passage. ProofWriter asks whether it can *chain* facts together. Laya decides in one forward pass with no scratchpad, so the hypothesis is that it does well at proof depth 0–1 and falls off as depth grows. Phi-4-mini should degrade more slowly, especially when it reasons step by step before answering. The result is a plot of accuracy against depth, which tests that architectural claim directly instead of reporting one leaderboard number. A fine-tuned Laya arm checks whether training on multi-hop examples fixes the fall-off, or whether it persists and so points at the single forward pass.

## Models

| Arm | Params | How it answers | Hugging Face id |
|---|---|---|---|
| Laya (zero-shot) | 421M | One `choice` question (True / False / Unknown), one forward pass, 512-token window | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (Apache-2.0) |
| Laya (fine-tuned) | 421M | The same, after ~3k ProofWriter *train* questions over depths 0–5 | fine-tuned from the above (not committed) |
| Phi-4-mini (direct) | 3.8B | "Answer with only True, False or Unknown", greedy | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) (MIT) |
| Phi-4-mini (CoT) | 3.8B | "Reason step by step…", then `Answer: …`, greedy, up to 512 new tokens | same |

Neither model requires a Hugging Face login.

## Dataset

The data is the official AI2 release, [ProofWriter](https://allenai.org/data/proofwriter) `proofwriter-dataset-V2020.12.3` (214 MB zip). The notebook downloads it from AI2's S3 bucket, which is where that page links. It uses the **open-world (OWA) `depth-5`** split, because only OWA has the third label, *Unknown*. Each question's depth is the release's own **`QDep`** field, which is never recomputed.

Each test theory is a paragraph of synthetic facts and rules ("Anne is kind. If someone is kind then they are round. …") with several statements to judge. The test sample is **stratified: 50 questions for each (QDep, label) cell, depths 0–5 × {True, False, Unknown}, 900 questions in total** (seed 42). Every depth has all three labels, and chance is exactly 1/3 at every depth. The notebook stops with an error if any cell is empty. The scarcest cell in the test split is depth-5 *Unknown*, with 63 questions. Unknowns at QDep 6–8 (10 test questions) are left out.

Theories are 69–265 Laya tokens long, and the mean barely changes with depth (155 at depth 0, 144 at depth 5). So every input fits Laya's window untruncated, and input length doesn't confound the depth curve. Train, dev and test share no theories.

## Method

- **Laya:** the theory is the `state`, and the question is one `choice` question. Its three options carry the label definitions: True = the statement follows from the facts and rules, False = its negation follows, Unknown = neither can be proved. Laya returns a probability for each option, and the prediction is the argmax. Batch size is 1.
- **Laya fine-tuned:** this uses the single-T4 port of Laya's official RLCD recipe from [`2026-09-laya-finetune-boolq`](../2026-09-laya-finetune-boolq), with the same hyperparameters (3 epochs, effective batch 64, LR 2.5e-5 / 1e-4, σ 0.4→0.1). The question type is `choice` instead of `noul`.
  - Training data: 3,006 questions from `meta-train`, 167 per (depth, label) cell.
  - Calibration: 306 questions from `meta-dev` track accuracy per epoch and fit the `choice` temperature.
- **Phi-4-mini (direct):** the prompt gives the same label definitions and ends "Answer with only True, False or Unknown." Decoding is greedy with 3 new tokens. The label probabilities are a softmax over the first token's logits for the True/False/Unknown variants.
- **Phi-4-mini (CoT):** the same system prompt, but the model is asked to reason step by step and finish with `Answer: True|False|Unknown`. Decoding is greedy with up to 512 new tokens, and the last `Answer:` is parsed. An output with no parseable answer counts as wrong, and the parse rate is reported.
- **Metrics:**
  - Accuracy per depth with 95% Wilson CIs, and per depth × label.
  - Overall accuracy, macro-F1 and parse rate.
  - Latency, CoT output length and peak GPU memory.
- **Degradation test:**
  - A per-arm logistic regression of `correct ~ QDep` gives each arm's slope in log-odds per extra proof step.
  - A GEE model `correct ~ QDep × arm`, clustered on question, tests whether each arm's slope differs from zero-shot Laya's.
  - Exact McNemar tests at each depth for the key pairs.

## How to run

1. Click **Open in Colab** above and choose **Runtime → Change runtime type → T4 GPU**.
2. Click **Runtime → Run all**. A T4 should take roughly 1.5 hours in total:
   - Laya zero-shot: about 1 minute
   - fine-tuning: about 15–20 minutes
   - Phi-4-mini direct: a few minutes
   - Phi-4-mini CoT: about 40–60 minutes

   Model outputs, the dataset zip and the fine-tuned checkpoint are cached, so a rerun after a disconnect resumes. Set `USE_DRIVE = True` to keep them on Google Drive, which is recommended for a run this long.
3. For a quick check first, set `N_PER_CELL = 2`, `N_TRAIN_PER_CELL = 4`, `N_CALIB_PER_CELL = 2` and `EPOCHS = 1`. Before the full run, delete `laya_proofwriter_ft/` and the `*.json`/`*.jsonl` files in `cache/`, or the quick check's checkpoint and training history will be reused.
4. Download the results zip and commit `figures/`, `results.csv`, `by_depth.csv` and `summary.csv` here. The dataset zip and the ~850 MB checkpoint are not committed.

To run locally, use `pip install -r requirements.txt` and a CUDA GPU with at least 12 GB of memory.

## Results

*Pending the first full run.*

The headline figure will be `figures/accuracy_vs_depth.png`. The supporting figures are `confusion_by_model.png` (shallow vs deep), `accuracy_depth_label_heatmap.png` and `cost_and_finetuning.png`.
