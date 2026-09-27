# Laya vs Phi-4-mini on BoolQ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-vs-phi4-boolq/notebook.ipynb)

## Question

On short-passage yes/no reading comprehension, how does Laya, a 421M-param non-autoregressive decision model, compare with the best SLM from [the GSM8K experiment](../2026-09-slm-gsm8k)? The comparison covers accuracy, calibration, speed and memory.

## Models

| Model | Params | Type | Hugging Face id | Licence |
|---|---|---|---|---|
| Laya | 421M | ModernBERT-large decision model, one forward pass, 512-token window | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | Apache-2.0 |
| Phi-4-mini-instruct | 3.8B | Autoregressive chat LLM (88.5% on GSM8K) | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT |

Neither model requires a Hugging Face login.

## Dataset

[BoolQ](https://huggingface.co/datasets/google/boolq) (`google/boolq`, `validation` split, CC-BY-SA-3.0). It has naturally occurring yes/no questions, each paired with a Wikipedia passage (~110 tokens on average), so almost every input fits in Laya's 512-token window. The notebook defaults to 500 randomly chosen examples (seed 42). The results below use all 3,270 (`N_SAMPLES = None`).

## Method

- **Laya:** the passage is the `state` and the question is one `noul` (yes/no) question, so the model returns P(yes).
- **Phi-4-mini:** it gets the same passage and question in a chat prompt ending "Answer with only Yes or No.". Decoding is greedy with `max_new_tokens=3`. P(yes) is the softmax over the first answer token's *Yes*/*No* logits, and the decoded text gives the prediction.
- **Same inputs:** passages longer than 400 Laya tokens are truncated for **both** models, and their accuracy is reported separately.
- **Metrics:** accuracy with a 95% Wilson CI, F1 (Yes), ROC-AUC, Brier score, ECE, ms per example, peak GPU memory and parameter count, plus a majority-class baseline. There is also an exact McNemar test on paired correctness, a threshold sweep and reliability diagrams.
- **Latency caveat:** Laya runs at batch size 1 (each example has its own question), while Phi-4-mini runs at batch size 8 with time split across the batch.

## How to run

1. Click **Open in Colab** above.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**. A T4 should take roughly 10–15 minutes: under a minute for Laya and most of the rest for Phi-4-mini. Model outputs are cached in `cache/`, so a rerun after a disconnect continues where it stopped. Set `USE_DRIVE = True` to keep the cache on Google Drive. For a quick check, set `N_SAMPLES = 20` first.
4. Download the results zip and commit `figures/`, `results.csv` and `summary.csv` here.

To run locally, use `pip install -r requirements.txt` and a CUDA GPU with at least 12 GB of memory.

## Results

The full BoolQ validation split (3,270 examples, run with `N_SAMPLES = None`) on a Colab T4, fp16. Per-example outputs are in [`results.csv`](results.csv).

| Model | Accuracy | 95% CI | F1 (Yes) | ROC-AUC | Brier | ECE | ms / example | Peak GPU memory |
|---|---|---|---|---|---|---|---|---|
| Majority class (always Yes) | 62.2% | | | | | | | |
| Laya (421M) | 75.7% | 74.2–77.1% | 0.811 | 0.812 | 0.170 | **0.065** | **50** (batch 1) | **2.7 GB** |
| Phi-4-mini (3.8B) | **81.4%** | 80.0–82.7% | **0.835** | **0.919** | **0.143** | 0.133 | 118 (batch 8) | 9.0 GB |

![Accuracy, latency and calibration](figures/accuracy_speed_calibration.png)

**Takeaways**

- **Phi-4-mini is more accurate, by 5.7 points.** The gap is real: in the paired comparison, Phi-4-mini alone was right on 572 questions and Laya alone on 386, which gives an exact McNemar p ≈ 2e-9. Its ROC-AUC (0.919 vs 0.812) shows that it also ranks answers much better, not just that it sits at a better threshold.
- **Laya is better calibrated.** Its ECE is half of Phi-4-mini's (0.065 vs 0.133). Phi-4-mini is biased towards *No*: it answers Yes 50% of the time against a true Yes rate of 62%, and its reliability curve sits above the diagonal. With a threshold of 0.3 instead of 0.5, Phi-4-mini would reach about 84%, but that threshold was picked on the test set, so treat it as an upper bound.
- **Laya is much cheaper.** It used under a third of the GPU memory and took under half the time per example, even at batch size 1 against Phi-4-mini's batch of 8. It is also 9× smaller.
- **Agreement:** both models were right on 2,089 questions and both wrong on 223.
- **Input length was not a factor.** Only 20 of the 3,270 passages (0.6%) went over the 400-token budget and were truncated. Phi-4-mini answered in the requested Yes/No format 100% of the time.

The follow-up experiment [`2026-09-laya-finetune-boolq`](../2026-09-laya-finetune-boolq) tests whether fine-tuning Laya on 3,000 BoolQ training examples closes the gap.
