#!/usr/bin/env bash
# Run every grid cell that has no summary yet, cheapest and most informative first:
# AutoCalibrate at small N, then the N=50 demo cells the free-model quota corrupted, then AutoCalibrate at large N.
# Finished calls are cached by DSPy, so re-running a partially finished cell only pays for what failed.
#   bash scripts/run_missing.sh            (THREADS as in run_all.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS="${THREADS:-8}"
SEEDS="${SEEDS:-0 1 2}"
python scripts/parley_smoke.py --no-call

cell() {  # cell CONDITION N SEED
  if [ -f "results/$1_n$2_s$3_test.summary.json" ]; then echo "skip $1 n=$2 seed=$3 (done)"; return; fi
  python -m biasjudge.run --threads "$THREADS" --condition "$1" --n "$2" --seed "$3" --split test
}
for n in 5 10; do for s in $SEEDS; do cell autocalibrate "$n" "$s"; done; done
for s in $SEEDS; do cell many_shot 50 "$s"; cell allure 50 "$s"; done
for n in 25 50; do for s in $SEEDS; do cell autocalibrate "$n" "$s"; done; done
python -m biasjudge.analyze
