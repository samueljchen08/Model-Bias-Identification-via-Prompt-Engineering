#!/usr/bin/env bash
# Full experimental grid. Every call is cached by DSPy, so re-running is free for finished items.
#   bash scripts/run_all.sh            # everything
#   SEEDS="0" NS="5 10" bash scripts/run_all.sh   # a subset
set -euo pipefail
cd "$(dirname "$0")/.."
NS="${NS:-5 10 25 50}"
SEEDS="${SEEDS:-0 1 2}"
THREADS="${THREADS:-8}"

run() { python -m biasjudge.run --threads "$THREADS" "$@"; }

# 1. zero-shot on test (the baseline) and on the pool (ALLURE needs these predictions)
run --condition zero_shot --n 0 --seed 0 --split test
run --condition zero_shot --n 0 --seed 0 --split pool

# 2. demo-based conditions (cheap: no optimization)
for n in $NS; do for s in $SEEDS; do
  run --condition many_shot --n "$n" --seed "$s" --split test
  run --condition allure    --n "$n" --seed "$s" --split test
done; done

# 3. AutoCalibrate via GEPA (runs the optimizer per (n, seed) first)
for n in $NS; do for s in $SEEDS; do
  run --condition autocalibrate --n "$n" --seed "$s" --split test
done; done

python -m biasjudge.analyze
