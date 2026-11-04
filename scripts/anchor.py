"""Sanity anchor: the dataset's own LLM-judge ratings vs human medians, on our exact test split.

StereoTales ships 23 judge models' zero-shot ratings (3 reps each). Comparing our zero-shot
run against the matching model here tells us whether the prompt reconstruction is faithful.

    python scripts/anchor.py                      # all 23 models
    python scripts/anchor.py --model azure_ai/gpt-4o
"""

from __future__ import annotations

import argparse

import pandas as pd

from biasjudge.data import load_human_labels, load_llm_evals, make_splits
from biasjudge.metrics import compute_metrics
from biasjudge.run import POOL_SIZE_TEST, SPLIT_SEED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", choices=("test", "pool", "all"), default="test")
    args = ap.parse_args()
    labels = load_human_labels()
    pool, test = make_splits(labels, test_size=POOL_SIZE_TEST, seed=SPLIT_SEED)
    target = {"test": test, "pool": pool, "all": labels}[args.split]
    evals = load_llm_evals()
    rows = []
    for model, d in evals.groupby("evaluator_model"):
        if args.model and model != args.model:
            continue
        med = d.groupby("value_association_id").harmful_rating.median()
        j = target.set_index("value_association_id").join(med.rename("pred"), how="inner")
        m = compute_metrics(j.human_median.tolist(), j.pred.round().tolist())
        rows.append({"model": model, **{k: m[k] for k in ("n", "spearman", "pearson", "mae", "f1", "kappa", "accuracy", "pred_five_rate")}})
    out = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    print(f"split={args.split} human_five_rate={float((target.human_median == 5).mean()):.3f}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
