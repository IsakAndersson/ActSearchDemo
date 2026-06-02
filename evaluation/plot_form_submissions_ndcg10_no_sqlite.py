from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .plot_form_submissions_grouped_metrics import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_QRELS_PATH,
        METHOD_LABELS,
        _compute_scores,
        _load_clean_form_submissions_qrels,
    )
except ImportError:
    from plot_form_submissions_grouped_metrics import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_QRELS_PATH,
        METHOD_LABELS,
        _compute_scores,
        _load_clean_form_submissions_qrels,
    )


METHOD_ORDER: Sequence[str] = ("bm25", "dense_e5", "hybrid_e5", "docplus_live")
METRIC_ORDER: Sequence[str] = ("NDCG@10 (query)",)
COLOR_MAP = {
    "bm25": "#4c78a8",
    "dense_e5": "#f58518",
    "hybrid_e5": "#54a24b",
    "docplus_live": "#72b7b2",
}


def _plot_ndcg_bars(
    scores_df: pd.DataFrame,
    output_path: Path,
    qrels_count: int,
    information_need_count: int,
) -> None:
    x_positions = list(range(len(METRIC_ORDER)))
    bar_width = min(0.18, 0.78 / max(1, len(METHOD_ORDER)))
    center_offset = (len(METHOD_ORDER) - 1) / 2
    offsets = {
        method_key: (index - center_offset) * bar_width
        for index, method_key in enumerate(METHOD_ORDER)
    }

    fig, ax = plt.subplots(figsize=(12.5, 7.2))

    for method_key in METHOD_ORDER:
        method_scores = (
            scores_df[scores_df["method"] == method_key]
            .set_index("metric_label")
            .reindex(METRIC_ORDER)
        )
        values = method_scores["value"].astype(float).tolist()
        positions = [x + offsets[method_key] for x in x_positions]
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=METHOD_LABELS[method_key],
            color=COLOR_MAP[method_key],
            edgecolor="#1f2933",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(METRIC_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Form submissions: nDCG@10 for queries by retrieval method")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    fig.text(
        0.5,
        0.01,
        f"Using {qrels_count} unique qrels across {information_need_count} information needs from form submissions.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a grouped nDCG@10 bar chart without SQLite FTS."
    )
    parser.add_argument(
        "--qrels-path",
        default=str(DEFAULT_QRELS_PATH),
        help="Path to qrels_from_form_submissions_clean.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the CSV and PNG are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k documents to retrieve for each method.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qrels_bundle, query_queries, info_need_queries = _load_clean_form_submissions_qrels(
        Path(args.qrels_path).resolve()
    )
    qrels_df, qrels_count, information_need_count = qrels_bundle

    scores_df = _compute_scores(
        qrels_df=qrels_df,
        query_queries=query_queries,
        info_need_queries=info_need_queries,
        top_k=args.top_k,
        qrels_count=qrels_count,
        information_need_count=information_need_count,
        method_order=METHOD_ORDER,
    )
    ndcg_scores_df = (
        scores_df[scores_df["metric_label"].isin(METRIC_ORDER)]
        .copy()
        .assign(
            method=lambda df: pd.Categorical(
                df["method"], categories=list(METHOD_ORDER), ordered=True
            ),
            metric_label=lambda df: pd.Categorical(
                df["metric_label"], categories=list(METRIC_ORDER), ordered=True
            ),
        )
        .sort_values(["metric_label", "method"])
        .reset_index(drop=True)
    )

    csv_path = output_dir / "form_submissions_ndcg10_no_sqlite.csv"
    png_path = output_dir / "form_submissions_ndcg10_no_sqlite.png"
    ndcg_scores_df.to_csv(csv_path, index=False)
    _plot_ndcg_bars(
        scores_df=ndcg_scores_df,
        output_path=png_path,
        qrels_count=qrels_count,
        information_need_count=information_need_count,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
