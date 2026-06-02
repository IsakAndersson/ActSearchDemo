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
    from .search_adapter import DEFAULT_CONFIG
    from .plot_form_submissions_grouped_metrics import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_QRELS_PATH,
        METHOD_LABELS,
        _compute_scores,
        _load_clean_form_submissions_qrels,
    )
except ImportError:
    from search_adapter import DEFAULT_CONFIG
    from plot_form_submissions_grouped_metrics import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_QRELS_PATH,
        METHOD_LABELS,
        _compute_scores,
        _load_clean_form_submissions_qrels,
    )


METHOD_ORDER: Sequence[str] = ("bm25", "dense_e5", "hybrid_e5", "docplus_live")
COLOR_MAP = {
    "bm25": "#4c78a8",
    "dense_e5": "#f58518",
    "hybrid_e5": "#54a24b",
    "docplus_live": "#72b7b2",
}


def _plot_scores(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    x_positions = list(range(len(scores_df)))
    values = scores_df["value"].astype(float).tolist()
    labels = scores_df["method"].map(METHOD_LABELS).tolist()
    colors = [COLOR_MAP[str(method)] for method in scores_df["method"]]

    bars = ax.bar(
        x_positions,
        values,
        color=colors,
        edgecolor="#1f2933",
        linewidth=0.8,
        width=0.62,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 1.08)
    ax.set_title("Form submissions: nDCG@10 for queries")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a bar chart for nDCG@10 on form-submissions queries using "
            "BM25, Dense E5, Hybrid E5, and Docplus Live."
        )
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
    csv_path = output_dir / "form_submissions_ndcg10_query_four_methods.csv"
    png_path = output_dir / "form_submissions_ndcg10_query_four_methods.png"

    print(f"Using E5 index: {DEFAULT_CONFIG.e5_index_path}", flush=True)
    print(f"Using E5 metadata: {DEFAULT_CONFIG.e5_metadata_path}", flush=True)
    if os.getenv("DOCPLUS_E5_INDEX_PATH"):
        print(
            f"DOCPLUS_E5_INDEX_PATH override is set: {os.getenv('DOCPLUS_E5_INDEX_PATH')}",
            flush=True,
        )
    if os.getenv("DOCPLUS_E5_METADATA_PATH"):
        print(
            f"DOCPLUS_E5_METADATA_PATH override is set: {os.getenv('DOCPLUS_E5_METADATA_PATH')}",
            flush=True,
        )

    qrels_bundle, query_queries, info_need_queries = _load_clean_form_submissions_qrels(
        Path(args.qrels_path).resolve()
    )
    qrels_df, _, _ = qrels_bundle

    scores_df = _compute_scores(
        qrels_df=qrels_df,
        query_queries=query_queries,
        info_need_queries=info_need_queries,
        top_k=args.top_k,
        qrels_count=0,
        information_need_count=0,
        method_order=METHOD_ORDER,
    )
    filtered_scores_df = (
        scores_df[
            (scores_df["query_variant"] == "query")
            & (scores_df["metric_slug"] == "ndcg10_query")
            & (scores_df["method"].isin(METHOD_ORDER))
        ]
        .copy()
        .assign(
            method=lambda df: pd.Categorical(
                df["method"], categories=list(METHOD_ORDER), ordered=True
            )
        )
        .sort_values("method")
        .reset_index(drop=True)
    )

    if csv_path.exists():
        csv_path.unlink()
    if png_path.exists():
        png_path.unlink()

    filtered_scores_df.to_csv(csv_path, index=False)
    _plot_scores(filtered_scores_df, png_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
