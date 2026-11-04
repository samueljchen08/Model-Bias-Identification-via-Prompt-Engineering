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

N_GRID = [0, 5, 10, 25, 50]
PLOT_METRICS = [("spearman", "Spearman ρ with human median"), ("f1", "F1 on harmful (median ≥ 4)"), ("mae", "MAE (1–5 scale)")]
ORDER = ["zero_shot", "many_shot", "allure", "autocalibrate"]


def load_summaries(results: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results.glob("*.summary.json")):
        s = json.loads(p.read_text())
        if s.get("limit") or s["split"] != "test":
            continue
        rows.append({"condition": s["condition"], "n": s["n"], "seed": s["seed"], "model": s["lm"]["model"], **s["metrics"]})
    return pd.DataFrame(rows)


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
    cols = ["condition", "n", "runs", "spearman", "spearman_sd", "f1", "f1_sd", "mae", "kappa", "pred_five_rate"]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
