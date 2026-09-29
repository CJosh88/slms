"""Post-hoc analysis of the ProofWriter run: the "not" shortcut and the provable-vs-Unknown split.

1. In ProofWriter OWA, almost every False statement contains "not" and almost every True one
   doesn't. "False if it says 'not', otherwise True, never Unknown" is scored on the release
   and on the 900-question test sample in results.csv, by depth.
2. How often fine-tuned Laya's True/False answers agree with that shortcut.
3. The part of the task "not" can't help with: provable (True/False) vs Unknown, as balanced
   accuracy (chance = 50%) by depth for every arm.

Downloads the official release zip (214 MB) into cache/ on first run. Run from the experiment
folder:  python scripts/shortcut_analysis.py   (needs pandas)
"""
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ZIP = HERE / "cache" / "proofwriter-dataset-V2020.12.3.zip"
URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
SPLIT_DIR = "proofwriter-dataset-V2020.12.3/OWA/depth-5"
MODELS = ["Laya (zero-shot)", "Laya (fine-tuned)", "Phi-4-mini (direct)", "Phi-4-mini (CoT)"]

if not ZIP.exists():
    ZIP.parent.mkdir(exist_ok=True)
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, ZIP)


def label(answer):
    return "True" if answer is True else "False" if answer is False else "Unknown"


def load(split):
    rows = []
    with zipfile.ZipFile(ZIP) as z, z.open(f"{SPLIT_DIR}/meta-{split}.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            for qid, q in rec["questions"].items():
                if q["QDep"] <= 5:
                    rows.append({"id": f"{rec['id']}:{qid}", "qdep": q["QDep"], "label": label(q["answer"]),
                                 "neg": " not " in f" {q['question']} "})
    return pd.DataFrame(rows)


print("1. Share of statements containing 'not', by gold label (QDep 0-5)")
for split in ["train", "dev", "test"]:
    d = load(split)
    print(f"   {split:5s}", (d.groupby("label").neg.mean() * 100).round(1).to_dict())

res = pd.read_csv(HERE / "results.csv")
test = load("test").set_index("id")
sample = res[res.model == MODELS[0]][["id", "qdep", "label"]].copy()
sample["neg"] = sample.id.map(test.neg)
sample["shortcut"] = sample.neg.map({True: "False", False: "True"})
ok = sample.shortcut == sample.label
print(f"\n   Shortcut accuracy on the test sample: {ok.mean():.1%} overall;",
      "by depth", (ok.groupby(sample.qdep).mean() * 100).round(1).to_dict())

print("\n2. Fine-tuned Laya vs the shortcut (gold True/False questions it answered True or False)")
ft = res[res.model == MODELS[1]].set_index("id").pred
s = sample.set_index("id")
known = s[(s.label != "Unknown") & ft.reindex(s.index).isin(["True", "False"])]
agree = (ft[known.index] == known.shortcut).mean()
odd = known[known.shortcut != known.label]
print(f"   agrees with the shortcut on {agree:.1%} of {len(known)};",
      f"right on {int((ft[odd.index] == odd.label).sum())} of the {len(odd)} where the shortcut is wrong")

print("\n3. Provable vs Unknown, balanced accuracy % (chance 50) by depth")
res["gold_known"] = res.label != "Unknown"
res["pred_known"] = res.pred.isin(["True", "False"])


def balanced(g):
    return 50 * (g[g.gold_known].pred_known.mean() + (~g[~g.gold_known].pred_known).mean())


print(pd.DataFrame({m: {q: round(balanced(g), 1) for q, g in res[res.model == m].groupby("qdep")} for m in MODELS})
      .to_string())
