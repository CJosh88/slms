#!/usr/bin/env bash
# Regenerates data/ with the official PrOntoQA generator (github.com/asaparov/prontoqa, Apache-2.0),
# pinned to a fixed commit. Hop counts are set by the generator (--min-hops/--max-hops), never inferred.
#   test  : seed 1001, 400 examples per hop count (the notebook samples a balanced subset)
#   train : seed 2002, 700 examples per hop count (fine-tuning + calibration slices)
# Flags: fictional ontology (made-up words, no world knowledge), random sentence order,
# relevant distractors (the generator's default), zero-shot (no in-context examples).
set -euo pipefail
COMMIT=0a6412b
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
git clone -q https://github.com/asaparov/prontoqa.git "$WORK/prontoqa"
git -C "$WORK/prontoqa" checkout -q "$COMMIT"
cd "$WORK/prontoqa"   # run_experiment.py reads bad_patterns.txt from the working directory

gen() {  # split seed trials
  python -W ignore run_experiment.py --model-name json --model-size none --ontology fictional \
    --ordering random --few-shot-examples 0 --min-hops 1 --max-hops 5 \
    --num-trials "$3" --seed "$2" > /dev/null
  for h in 1 2 3 4 5; do
    mv "${h}hop_0shot_random_seed$2.json" "$HERE/data/$1_${h}hop.json"
  done
}
gen test 1001 400
gen train 2002 700
rm -rf "$WORK"
echo "wrote $(ls "$HERE"/data/*.json | wc -l) files to $HERE/data"
