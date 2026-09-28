# Laya vs Phi-4-mini on ProntoQA: accuracy vs reasoning hops

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-laya-vs-phi4-prontoqa/notebook.ipynb)

## Question

[BoolQ](../2026-09-laya-vs-phi4-boolq) asked whether Laya can *find* an answer in a passage. This experiment asks whether it can *chain* facts together. Laya decides in a single forward pass with no scratchpad, so the hypothesis is that it handles 1-hop questions and falls off as the hop count grows. Phi-4-mini, especially when it reasons step by step before answering, should degrade more slowly. The headline is a plot of accuracy against hop count, which tests that architectural claim directly instead of reporting one leaderboard number. A fine-tuned Laya arm checks whether any fall-off is a lack of training on the task or a limit of the single forward pass.

### Why ProntoQA, not ProofWriter

The [ProofWriter run](../2026-09-laya-vs-phi4-proofwriter) was undermined by a surface cue: ~95% of its False statements contain "not" and ~95% of its True ones don't. "False if it says *not*, otherwise True" scored 64% at every depth without any reasoning, and fine-tuned Laya's flat 66% sat right on top of it.

ProntoQA ([Saparov & He, ICLR 2023](https://openreview.net/forum?id=qFVVBzXxR2V)) was built to test whether chain-of-thought helps as the number of hops grows. Three properties make it a cleaner test:
- **Made-up vocabulary** ("Every wumpus is a yumpus"), so world knowledge can't help.
- **Distractor rules** that assign the opposite property to an unrelated concept, so spotting the property in the text isn't enough.
- **Label and negation are independent in the generated data.** The sample is balanced on both anyway, so a "not"-based guess scores exactly 50%.

## Models

| Arm | Params | How it answers | Hugging Face id |
|---|---|---|---|
| Laya (zero-shot) | 421M | One `noul` (yes/no) question, one forward pass, 512-token window | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (Apache-2.0) |
| Laya (fine-tuned) | 421M | The same, after fine-tuning on 3,000 generated training examples over 1–5 hops | fine-tuned from the above (not committed) |
| Phi-4-mini (direct) | 3.8B | "Answer with only True or False", greedy | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) (MIT) |
| Phi-4-mini (CoT) | 3.8B | "Reason step by step…", then `Answer: True/False`, greedy, up to 1,024 new tokens | same |

Neither model requires a Hugging Face login.

## Dataset

The data in [`data/`](data) was generated with the authors' official generator, [`asaparov/prontoqa`](https://github.com/asaparov/prontoqa) (Apache-2.0), pinned to commit `0a6412b`. [`scripts/generate_data.sh`](scripts/generate_data.sh) reproduces it exactly.

**Generator settings:**
- `--ontology fictional`
- `--ordering random`: facts and rules are shuffled, so the proof can't be read off in order.
- Relevant distractors (the generator's default).
- Zero-shot: no in-context examples.
- `--min-hops 1 --max-hops 5`.

**Hop count** is the generator's own setting and is never inferred. As a sanity check, the notebook asserts that each gold proof has `2 × hops + 1` steps.

**Sizes:**
- **Test:** seed 1001, 400 generated examples per hop count. The notebook samples **50 questions for every hop × label × contains-"not" cell**, 1,000 in total, so chance is exactly 50% at every hop count.
- **Train:** seed 2002, 700 per hop count. It supplies 3,000 fine-tuning examples and a disjoint 300-example calibration slice, both balanced over hops × label. The notebook checks that no test example appears in train.

Theories are at most 209 Laya tokens, so nothing is truncated. More hops need more rules, so theories get longer with hop count: a mean of 118 tokens at 1 hop and 195 at 5 hops. Hop count and input length therefore move together. Both stay well inside Laya's 512-token window.

The ProntoQA authors released pre-generated data too, but the True/False version only covers 1, 3 and 5 hops and exists only inside GPT-3 prompt logs. The PrOntoQA-OOD files have "Prove: …" queries with no True/False label. Running the pinned generator was the clean way to get every hop count from 1 to 5.

## Method

- **Laya:** the theory is the `state`, and the statement is one `noul` question ("Using only the facts and rules in the text, is this statement true?"). Laya returns P(true), scored at batch size 1.
- **Laya fine-tuned:** the single-T4 port of Laya's official RLCD recipe, unchanged from [`2026-09-laya-finetune-boolq`](../2026-09-laya-finetune-boolq): the same `noul` target and hyperparameters (3 epochs, effective batch 64, LR 2.5e-5 / 1e-4, σ 0.4→0.1). The `noul` temperature is fitted on the calibration slice.
- **Phi-4-mini direct:** greedy decoding with 3 new tokens. P(true) is the softmax over the first token's True/False logits.
- **Phi-4-mini CoT:** reasons step by step, then the last `Answer:` is parsed. Unparsed or cut-off outputs count as wrong, and the parse rate is reported.
- **Thresholds:** for the three arms that output P(true), accuracy is reported at 0.5 *and* at a threshold tuned on the calibration slice. ROC-AUC per hop count is reported too. Together these separate "no signal" from "signal with a yes/no bias". That distinction was hard to make on ProofWriter.
- **Degradation test:**
  - Per arm, a logistic regression of `correct ~ hops` gives the slope in log-odds per extra hop.
  - Across arms, a GEE model `correct ~ hops × arm`, clustered on question, tests whether the slopes differ. It uses each Laya arm as the reference in turn.
  - Exact McNemar tests at each hop count for the key pairs.
- **Also reported:** answer bias against hops (the share answered True), latency, CoT length and peak GPU memory.

## How to run

1. Click **Open in Colab** above and choose **Runtime → Change runtime type → T4 GPU**.
2. Click **Runtime → Run all**. The data files are fetched from this repo automatically. A T4 should take roughly 1–1.5 hours, most of it Phi-4-mini CoT. Outputs and the fine-tuned checkpoint are cached, so a rerun after a disconnect resumes. Set `USE_DRIVE = True` to keep them on Google Drive.
3. For a quick check, set `N_PER_CELL = 2`, `N_TRAIN_PER_CELL = 4`, `N_CALIB_PER_CELL = 2` and `EPOCHS = 1`. Delete `cache/` and `laya_prontoqa_ft/` before the full run.
4. Download the results zip and commit `figures/`, `results.csv`, `by_hops.csv` and `summary.csv` here.

To regenerate the data, run `bash scripts/generate_data.sh` in an environment with `numpy` and `scipy`. To run the notebook locally, use `pip install -r requirements.txt` and a CUDA GPU with at least 12 GB of memory.

## Results

*Pending the first full run.*

The headline figure will be `figures/accuracy_vs_hops.png`, with accuracy at threshold 0.5 and at the tuned threshold. The supporting figures are `signal_and_bias.png` (ROC-AUC, answer bias and per-label accuracy against hops) and `cost_and_finetuning.png`.
