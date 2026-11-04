"""Run one (condition, N, seed, split) and write per-item JSONL plus a summary JSON.

    python -m biasjudge.run --condition zero_shot --n 0 --seed 0 --split test
    python -m biasjudge.run --condition many_shot --n 10 --seed 1 --split test
    python -m biasjudge.run --condition allure --n 25 --seed 0 --split test     # needs zero_shot on pool first
    python -m biasjudge.run --condition autocalibrate --n 5 --seed 0 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import dspy
import pandas as pd

from .conditions.allure import select_deviant_demos
from .conditions.autocalibrate import optimize_instructions
from .conditions.many_shot import select_random_demos
from .data import demo_pool, load_human_labels, make_splits, to_examples
from .judge import build_judge, judge_items, lm_metadata, load_settings, make_lm
from .metrics import compute_metrics, per_attribute_delta
from .prompt import HUMAN_INSTRUCTIONS

CONDITIONS = ("zero_shot", "many_shot", "allure", "autocalibrate")
POOL_SIZE_TEST = 1000
SPLIT_SEED = 0


def result_name(condition: str, n: int, seed: int, split: str, limit: int | None) -> str:
    name = f"{condition}_n{n}_s{seed}_{split}"
    return f"{name}_limit{limit}" if limit else name


def load_pool_predictions(out_dir: Path) -> dict[str, int | None]:
    path = out_dir / (result_name("zero_shot", 0, 0, "pool", None) + ".jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"ALLURE needs zero-shot predictions on the pool at {path}. Run:\n"
            "  python -m biasjudge.run --condition zero_shot --n 0 --seed 0 --split pool"
        )
    preds: dict[str, int | None] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["value_association_id"]] = rec["harmful_rating"]
    return preds


def prepare_judge(
    condition: str, n: int, seed: int, pool: pd.DataFrame, out_dir: Path, reflection_lm: dspy.LM | None
) -> tuple[dspy.Predict, dict]:
    """Build the judge for a condition. Returns (judge, metadata about what the N labels were used for)."""
    demos_pool = to_examples(demo_pool(pool))
    meta: dict = {"instructions": HUMAN_INSTRUCTIONS, "demo_ids": [], "optimizer": None}
    if condition == "zero_shot":
        if n != 0:
            raise ValueError("zero_shot requires --n 0")
        return build_judge(), meta
    if n <= 0:
        raise ValueError(f"{condition} requires --n > 0")
    if condition == "many_shot":
        demos = select_random_demos(demos_pool, n, seed)
        meta["demo_ids"] = [d.value_association_id for d in demos]
        return build_judge(demos=demos), meta
    if condition == "allure":
        demos = select_deviant_demos(demos_pool, load_pool_predictions(out_dir), n, seed)
        meta["demo_ids"] = [d.value_association_id for d in demos]
        return build_judge(demos=demos), meta
    if condition == "autocalibrate":
        trainset = select_random_demos(demos_pool, n, seed)  # same N items as many_shot for this seed
        if reflection_lm is None:
            raise ValueError("autocalibrate needs a reflection LM")
        log_dir = out_dir / "gepa_logs" / result_name(condition, n, seed, "opt", None)
        log_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        judge, instructions = optimize_instructions(build_judge(), trainset, seed, reflection_lm, log_dir=str(log_dir))
        meta.update(
            instructions=instructions,
            demo_ids=[d.value_association_id for d in trainset],
            optimizer={"name": "GEPA", "auto": "light", "seconds": round(time.time() - t0, 1),
                       "reflection_model": reflection_lm.model},
        )
        return judge, meta
    raise ValueError(f"unknown condition {condition}")


def build_summary(condition, n, seed, split, limit, lm, meta, merged: pd.DataFrame) -> dict:
    metrics = compute_metrics(merged.human_median.tolist(), merged.harmful_rating.tolist())
    per_attr = per_attribute_delta(merged)
    return {
        "condition": condition,
        "n": n,
        "seed": seed,
        "split": split,
        "limit": limit,
        "lm": lm_metadata(lm),
        "instructions": meta["instructions"],
        "demo_ids": meta["demo_ids"],
        "optimizer": meta["optimizer"],
        "metrics": metrics,
        "per_attribute": per_attr.to_dict(orient="records"),
        "errors": merged.error.dropna().head(10).tolist(),
    }


def main(argv: list[str] | None = None, lm: dspy.LM | None = None, reflection_lm: dspy.LM | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("test", "pool"), default="test")
    ap.add_argument("--limit", type=int, default=None, help="score only the first K items of the split (smoke runs)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_human_labels()
    pool, test = make_splits(labels, test_size=POOL_SIZE_TEST, seed=SPLIT_SEED)
    target = test if args.split == "test" else pool
    if args.limit:
        target = target.head(args.limit)

    if lm is None:
        settings = load_settings()
        lm = make_lm(settings)
        reflection_lm = make_lm(settings, role="reflection")
    dspy.configure(lm=lm)

    judge, meta = prepare_judge(args.condition, args.n, args.seed, pool, out_dir, reflection_lm)
    examples = to_examples(target)  # labels ride along but are never shown to the judge
    t0 = time.time()
    records = judge_items(judge, examples, num_threads=args.threads)
    elapsed = round(time.time() - t0, 1)

    preds = pd.DataFrame(records)
    merged = target.merge(preds.drop(columns=["statement"]), on="value_association_id", how="left")
    name = result_name(args.condition, args.n, args.seed, args.split, args.limit)
    with (out_dir / f"{name}.jsonl").open("w") as f:
        for r in merged.to_dict(orient="records"):
            f.write(json.dumps(r, default=_json_default) + "\n")
    summary = build_summary(args.condition, args.n, args.seed, args.split, args.limit, lm, meta, merged)
    summary["seconds"] = elapsed
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    m = summary["metrics"]
    print(
        f"{name}: n={m['n']} failed={m['n_failed']} spearman={m['spearman']:.3f} pearson={m['pearson']:.3f} "
        f"mae={m['mae']:.3f} f1={m['f1']:.3f} kappa={m['kappa']:.3f} acc={m['accuracy']:.3f} ({elapsed}s)"
    )
    return summary


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float) and o != o:
        return None
    return str(o)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
