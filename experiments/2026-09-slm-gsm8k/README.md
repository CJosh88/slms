# Small language models on GSM8K: Qwen3-1.7B vs SmolLM3-3B vs Phi-4-mini

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CJosh88/slms/blob/main/experiments/2026-09-slm-gsm8k/notebook.ipynb)

## Question

Which popular small language model (4B parameters or fewer) solves grade-school maths word problems most accurately, and what does that accuracy cost in speed and memory?

## Models

| Model | Params | Hugging Face id | Licence |
|---|---|---|---|
| Qwen3-1.7B | 1.7B | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0 |
| SmolLM3-3B | 3B | [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Apache-2.0 |
| Phi-4-mini-instruct | 3.8B | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT |

None of these models require a Hugging Face login.

## Dataset

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) (`openai/gsm8k`, `main` config, `test` split), MIT licence. The notebook uses 200 randomly chosen questions (seed 42). Set `N_SAMPLES = 1319` to use the full test set.

## Method

- Every model gets the same zero-shot step-by-step prompt and must end its answer with `#### <number>`.
- Decoding is greedy with `max_new_tokens=512` and batch size 8. Thinking mode is off for Qwen3 and SmolLM3.
- The final number is extracted and compared with the reference answer (exact match).
- **Metrics:** accuracy with a 95% Wilson confidence interval, seconds per question, generated tokens, tokens per second, peak GPU memory, and how often each model used the requested answer format.

## How to run

1. Click **Open in Colab** above.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**. A T4 takes about 20–35 minutes. Model outputs are cached in `cache/`, so if Colab disconnects you can rerun and it continues where it stopped. Set `USE_DRIVE = True` to keep the cache on Google Drive.
4. Download the files in `figures/`, plus `results.csv` and `summary.csv`, and commit them here.

To run locally, use `pip install -r requirements.txt` and a CUDA GPU with at least 12 GB of memory.

## Results

TODO: fill after running
