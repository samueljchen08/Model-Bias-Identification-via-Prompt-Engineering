"""Aggregate results/*.summary.json into tables and a metric-vs-N plot.

    python -m biasjudge.analyze [--results results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import compute_metrics  # noqa: E402

N_GRID = [0, 5, 10, 25, 50]
PLOT_METRICS = [("spearman", "Spearman ρ with human median"), ("f1", "F1 on harmful (median ≥ 4)"), ("mae", "MAE (1–5 scale)")]
ORDER = ["zero_shot", "many_shot", "allure", "autocalibrate"]


def _test_summaries(results: Path) -> list[dict]:
    out = []
    for p in sorted(results.glob("*.summary.json")):
        s = json.loads(p.read_text())
        if s.get("limit") or s["split"] != "test":
            continue
        out.append(s)
    return out


def load_summaries(results: Path) -> pd.DataFrame:
    rows = [
        {"condition": s["condition"], "n": s["n"], "seed": s["seed"], "model": s["lm"]["model"],
         **{("n_items" if k == "n" else k): v for k, v in s["metrics"].items()}}
        for s in _test_summaries(results)
    ]
    return pd.DataFrame(rows)


def per_attribute_table(results: Path) -> pd.DataFrame:
    """Mean signed error (judge - human) per base attribute, averaged over seeds, for every (condition, n)."""
    rows = []
    for s in _test_summaries(results):
        for r in s["per_attribute"]:
            rows.append({"condition": s["condition"], "n": s["n"], "seed": s["seed"],
                         "base_attribute": r["base_attribute"], "n_items": r["n"],
                         "mean_signed_error": r["mean_signed_error"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    wide = df.groupby(["base_attribute", "condition", "n"]).mean_signed_error.mean().unstack(["condition", "n"])
    wide.columns = [f"{c}_n{n}" for c, n in wide.columns]
    return wide.reset_index()


def dump_instructions(results: Path, path: Path) -> None:
    parts = ["# Learned AutoCalibrate (GEPA) instructions\n"]
    for s in _test_summaries(results):
        if s["condition"] != "autocalibrate":
            continue
        m = s["metrics"]
        parts.append(
            f"## N={s['n']}, seed={s['seed']}  (test Spearman {m['spearman']:.3f}, F1 {m['f1']:.3f}, "
            f"MAE {m['mae']:.3f}; optimizer {s['optimizer']})\n\n```\n{s['instructions']}\n```\n"
        )
    path.write_text("\n".join(parts))


def common_items_table(results: Path) -> tuple[pd.DataFrame, int]:
    """Every test-split run re-scored on the items *all* runs rated.

    Runs made after Parley enabled its output guardrail (Nov 2026) lose a few percent of items to
    content_filter; runs made before it scored every item. Restricting to the shared items makes the
    conditions comparable. Returns (per-run metrics, number of shared items)."""
    runs: dict[str, pd.DataFrame] = {}
    for p in sorted(results.glob("*_test.jsonl")):
        if "_limit" in p.name:
            continue
        runs[p.stem] = pd.read_json(p, lines=True)
    if not runs:
        return pd.DataFrame(), 0
    common = set.intersection(*(set(d[d.harmful_rating.notna()].value_association_id) for d in runs.values()))
    rows = []
    for name, d in runs.items():
        s = json.loads((results / f"{name}.summary.json").read_text())
        sub = d[d.value_association_id.isin(common)]
        m = compute_metrics(sub.human_median.tolist(), sub.harmful_rating.tolist())
        rows.append({"condition": s["condition"], "n": s["n"], "seed": s["seed"], "model": s["lm"]["model"],
                     **{("n_items" if k == "n" else k): v for k, v in m.items()}})
    return pd.DataFrame(rows), len(common)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in ("condition", "n", "seed", "model")]
    g = df.groupby(["condition", "n"])[metric_cols]
    out = g.mean().join(g.std().add_suffix("_sd")).join(g.size().rename("runs")).reset_index()
    out["condition"] = pd.Categorical(out.condition, ORDER)
    return out.sort_values(["condition", "n"]).reset_index(drop=True)


def plot(agg: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(5 * len(PLOT_METRICS), 4))
    zero = agg[agg.condition == "zero_shot"]
    for ax, (metric, label) in zip(axes, PLOT_METRICS):
        if len(zero):
            ax.axhline(zero[metric].iloc[0], color="grey", linestyle="--", label="zero-shot")
        for cond in ("many_shot", "allure", "autocalibrate"):
            d = agg[agg.condition == cond].sort_values("n")
            if len(d):
                ax.errorbar(d.n, d[metric], yerr=d[f"{metric}_sd"].fillna(0), marker="o", capsize=3, label=cond)
        ax.set_xlabel("N human-labeled examples")
        ax.set_ylabel(label)
        ax.set_xticks([n for n in N_GRID if n > 0])
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    results = Path(args.results)
    df = load_summaries(results)
    if df.empty:
        print(f"no test-split summaries in {results}")
        return
    df.to_csv(results / "summary.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(results / "summary_by_n.csv", index=False)
    plot(agg, results / "metric_vs_n.png")
    pa = per_attribute_table(results)
    if not pa.empty:
        pa.to_csv(results / "per_attribute.csv", index=False)
    dump_instructions(results, results / "learned_instructions.md")
    cols = ["condition", "n", "runs", "spearman", "spearman_sd", "f1", "f1_sd", "mae", "kappa", "pred_five_rate"]
    print("All rated items per run:")
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    common, n_common = common_items_table(results)
    if not common.empty:
        common.to_csv(results / "summary_common.csv", index=False)
        agg_c = aggregate(common)
        agg_c.to_csv(results / "summary_by_n_common.csv", index=False)
        plot(agg_c, results / "metric_vs_n_common.png")
        print(f"\nItems rated in every run ({n_common} of 1000):")
        print(agg_c[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
