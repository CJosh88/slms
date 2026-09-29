"""Scores a no-chaining heuristic on the generated ProntoQA test files.

Every theory has exactly two rules ending in the queried property, with opposite polarity: the
real one and a distractor. The distractor's subject concept is mentioned nowhere else in the
theory, while the real rule's subject sits on the proof chain and recurs. So:

    1. find the two rules that end in the statement's property,
    2. keep the one whose subject concept appears most often in the theory,
    3. answer True if its polarity ("not" or not) matches the statement's, else False.

This never looks at the entity or follows a single link. Run from the experiment folder:
    python scripts/shortcut_check.py
"""
import collections
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def stem(word):
    word = word.lower()
    return word[:-2] if word.endswith("uses") else word   # "wumpuses" -> "wumpus"


def subject(rule):
    words = rule.split()
    return stem(words[1]) if words[0] in ("Every", "Each") else stem(words[0])


def negated(sentence):
    return " not " in f" {sentence} "


correct, counts = collections.Counter(), collections.Counter()
for hops in range(1, 6):
    for ex in json.loads((DATA / f"test_{hops}hop.json").read_text(encoding="utf-8")).values():
        t = ex["test_example"]
        statement = t["query"].removeprefix("True or false: ").rstrip(".")
        prop = statement.split()[-1]
        sentences = [s.strip() for s in t["question"].split(".") if s.strip()]
        rules = [s for s in sentences if s.split()[-1] == prop]
        assert len(rules) == 2, rules
        words = [stem(w) for w in re.findall(r"[A-Za-z]+", t["question"])]
        mentions = {r: sum(w == subject(r) for w in words) for r in rules}
        chosen = max(rules, key=mentions.get)
        answer = "True" if negated(chosen) == negated(statement) else "False"
        correct[hops] += answer == t["answer"]
        counts[hops] += 1
        assert sorted(mentions.values())[0] == 1, mentions   # the distractor's subject appears exactly once

for hops in sorted(counts):
    print(f"{hops} hops: {correct[hops]}/{counts[hops]} = {correct[hops] / counts[hops]:.1%}")
total = sum(correct.values()) / sum(counts.values())
print(f"overall: {total:.1%} on {sum(counts.values())} generated test questions")
