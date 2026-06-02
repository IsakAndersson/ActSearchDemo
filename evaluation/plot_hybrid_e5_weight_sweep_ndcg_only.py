from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS_CSV = Path(
    "/home/isak/Repos/ActSearchDemo/evaluation/plots/hybrid_e5_weight_sweep/"
    "hybrid_e5_weight_sweep_metrics.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/isak/Repos/ActSearchDemo/evaluation/plots/hybrid_e5_weight_sweep/"
    "hybrid_e5_weight_sweep_ndcg10.png"
)
VARIANT_LINESTYLES = {
    "query": "-",
    "information_need": "--",
}
VARIANT_LABELS = {
    "query": "Query",
    "information_need": "Information need",
}


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"variant", "e5_weight_percent", "weight_label", "ndcg@10"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")

    df = df.copy()
    df["e5_weight_percent"] = pd.to_numeric(df["e5_weight_percent"], errors="coerce")
    df["ndcg@10"] = pd.to_numeric(df["ndcg@10"], errors="coerce")
    df = df.dropna(subset=["e5_weight_percent", "ndcg@10"])
    return df.sort_values(["variant", "e5_weight_percent"]).reset_index(drop=True)


def _plot_ndcg(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for variant_name in ("query", "information_need"):
        variant_df = results_df[results_df["variant"] == variant_name]
        if variant_df.empty:
            continue
        ax.plot(
            variant_df["e5_weight_percent"],
            variant_df["ndcg@10"],
            marker="o",
            linewidth=2,
            linestyle=VARIANT_LINESTYLES[variant_name],
            color="#1f77b4",
            label=VARIANT_LABELS[variant_name],
        )

    ax.set_title("nDCG@10")
    ax.set_ylabel("Value")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_xlabel("E5 weight / BM25 weight")

    query_df = results_df[results_df["variant"] == "query"]
    if not query_df.empty:
        ax.set_xticks(query_df["e5_weight_percent"])
        ax.set_xticklabels(query_df["weight_label"])

    fig.suptitle("Hybrid E5/BM25 weight sweep on form submissions", y=0.995)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot only nDCG@10 from an existing hybrid_e5 weight sweep metrics CSV."
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS_CSV),
        help="Path to hybrid_e5_weight_sweep_metrics.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv)
    output_path = Path(args.output)
    results_df = _load_metrics(metrics_csv)
    _plot_ndcg(results_df, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
