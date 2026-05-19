from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import ir_measures
import matplotlib.pyplot as plt
import pandas as pd
from ir_measures import RR, nDCG

try:
    from .evaluation import EVALUATION_DIR
    from .search_adapter import (
        bm25_search,
        dense_e5_search,
        docplus_live_search,
        hybrid_e5_search,
        sqlite_fts_search,
    )
except ImportError:
    from evaluation import EVALUATION_DIR
    from search_adapter import (
        bm25_search,
        dense_e5_search,
        docplus_live_search,
        hybrid_e5_search,
        sqlite_fts_search,
    )


DEFAULT_QRELS_PATH = EVALUATION_DIR / "qrels_from_form_submissions_clean.csv"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots" / "form_submissions_grouped_metrics"

METHODS: Dict[str, Callable[[str, int], List[Tuple[str, float]]]] = {
    "bm25": bm25_search,
    "dense_e5": dense_e5_search,
    "hybrid_e5": hybrid_e5_search,
    "sqlite_fts": sqlite_fts_search,
    "docplus_live": docplus_live_search,
}

METHOD_LABELS = {
    "bm25": "BM25",
    "dense_e5": "Dense E5",
    "hybrid_e5": "Hybrid E5",
    "sqlite_fts": "SQLite FTS",
    "docplus_live": "Docplus Live",
}

CORE_METHOD_ORDER = ["bm25", "dense_e5", "hybrid_e5"]
EXTENDED_METHOD_ORDER = ["bm25", "dense_e5", "hybrid_e5", "sqlite_fts", "docplus_live"]

METRIC_SPECS: Sequence[Tuple[str, object, str, str]] = (
    ("NDCG@10 (query)", nDCG @ 10, "query", "ndcg10_query"),
    ("NDCG@10 (info need)", nDCG @ 10, "information_need", "ndcg10_information_need"),
    ("RR@100 (query)", RR @ 100, "query", "rr100_query"),
    ("RR@100 (info need)", RR @ 100, "information_need", "rr100_information_need"),
)


def _load_clean_form_submissions_qrels(path: Path) -> tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(path)
    required_columns = {"query_id", "doc_id", "information_need"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")

    filtered = df.copy()
    if "relevance" in filtered.columns:
        filtered["relevance"] = pd.to_numeric(filtered["relevance"], errors="coerce").fillna(0)
        filtered = filtered[filtered["relevance"] > 0]
    else:
        filtered["relevance"] = 1

    filtered["query_id"] = filtered["query_id"].fillna("").astype(str).str.strip()
    filtered["information_need"] = filtered["information_need"].fillna("").astype(str).str.strip()
    filtered["doc_id"] = filtered["doc_id"].fillna("").astype(str).str.strip()
    filtered = filtered[
        (filtered["query_id"] != "")
        & (filtered["information_need"] != "")
        & (filtered["doc_id"] != "")
    ].copy()

    filtered = filtered.drop_duplicates(subset=["query_id", "information_need", "doc_id"]).reset_index(drop=True)
    information_need_count = filtered["information_need"].nunique()
    qrels_count = len(filtered)

    query_rows: List[dict[str, object]] = []
    info_need_rows: List[dict[str, object]] = []
    query_queries: List[Tuple[str, str]] = []
    info_need_queries: List[Tuple[str, str]] = []

    dedup_pairs = filtered[["query_id", "information_need"]].drop_duplicates().reset_index(drop=True)
    for row_number, pair in enumerate(dedup_pairs.itertuples(index=False), start=1):
        query_qid = f"query_{row_number:02d}"
        info_need_qid = f"info_need_{row_number:02d}"
        query_text = str(pair.query_id)
        info_need_text = str(pair.information_need)
        query_queries.append((query_qid, query_text))
        info_need_queries.append((info_need_qid, info_need_text))

        matching = filtered[
            (filtered["query_id"] == query_text)
            & (filtered["information_need"] == info_need_text)
        ]
        for doc_id in matching["doc_id"]:
            query_rows.append({"query_id": query_qid, "doc_id": str(doc_id), "relevance": 1})
            info_need_rows.append({"query_id": info_need_qid, "doc_id": str(doc_id), "relevance": 1})

    query_qrels = pd.DataFrame(query_rows)
    info_need_qrels = pd.DataFrame(info_need_rows)
    return (
        pd.concat(
            [
                query_qrels.assign(query_variant="query"),
                info_need_qrels.assign(query_variant="information_need"),
            ],
            ignore_index=True,
        ),
        qrels_count,
        information_need_count,
    ), query_queries, info_need_queries


def _build_run_df(
    queries: Sequence[Tuple[str, str]],
    search_fn: Callable[[str, int], List[Tuple[str, float]]],
    top_k: int,
) -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    for query_id, query_text in queries:
        results = search_fn(query_text, top_k)
        for doc_id, score in results:
            rows.append({"query_id": query_id, "doc_id": str(doc_id), "score": float(score)})
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])


def _compute_scores(
    qrels_df: pd.DataFrame,
    query_queries: Sequence[Tuple[str, str]],
    info_need_queries: Sequence[Tuple[str, str]],
    top_k: int,
    qrels_count: int,
    information_need_count: int,
    method_order: Sequence[str],
) -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    for method_key in method_order:
        search_fn = METHODS[method_key]
        query_run = _build_run_df(query_queries, search_fn, top_k=top_k)
        info_need_run = _build_run_df(info_need_queries, search_fn, top_k=top_k)

        runs = {
            "query": query_run,
            "information_need": info_need_run,
        }
        for metric_label, metric, variant, metric_slug in METRIC_SPECS:
            variant_qrels = qrels_df[qrels_df["query_variant"] == variant][["query_id", "doc_id", "relevance"]]
            value = ir_measures.calc_aggregate([metric], variant_qrels, runs[variant])[metric]
            rows.append(
                {
                    "method": method_key,
                    "method_label": METHOD_LABELS[method_key],
                    "metric_label": metric_label,
                    "metric_slug": metric_slug,
                    "query_variant": variant,
                    "value": float(value),
                    "qrels_count": qrels_count,
                    "information_need_count": information_need_count,
                }
            )
    return pd.DataFrame(rows)


def _plot_grouped_bars(
    scores_df: pd.DataFrame,
    output_path: Path,
    qrels_count: int,
    information_need_count: int,
    method_order: Sequence[str],
    title: str,
) -> None:
    metric_order = [label for label, _, _, _ in METRIC_SPECS]
    color_map = {
        "bm25": "#4c78a8",
        "dense_e5": "#f58518",
        "hybrid_e5": "#54a24b",
        "sqlite_fts": "#e45756",
        "docplus_live": "#72b7b2",
    }

    x_positions = list(range(len(metric_order)))
    bar_width = min(0.18, 0.78 / max(1, len(method_order)))
    center_offset = (len(method_order) - 1) / 2
    offsets = {
        method_key: (index - center_offset) * bar_width
        for index, method_key in enumerate(method_order)
    }

    fig_width = 12.5 if len(method_order) <= 3 else 14.5
    fig, ax = plt.subplots(figsize=(fig_width, 7.2))

    for method_key in method_order:
        method_scores = (
            scores_df[scores_df["method"] == method_key]
            .set_index("metric_label")
            .reindex(metric_order)
        )
        values = method_scores["value"].astype(float).tolist()
        positions = [x + offsets[method_key] for x in x_positions]
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=METHOD_LABELS[method_key],
            color=color_map[method_key],
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
    ax.set_xticklabels(metric_order, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
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
        description="Create a grouped bar chart for form submissions metrics across BM25, Dense, and Hybrid."
    )
    parser.add_argument(
        "--qrels-path",
        default=str(DEFAULT_QRELS_PATH),
        help="Path to qrels_from_form_submissions_clean.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the chart and CSV are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-k documents to retrieve for each method. Must be >= 100 for RR@100.",
    )
    args = parser.parse_args()

    if args.top_k < 100:
        raise ValueError("--top-k must be at least 100 to compute RR@100 correctly.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qrels_bundle, query_queries, info_need_queries = _load_clean_form_submissions_qrels(Path(args.qrels_path).resolve())
    qrels_df, qrels_count, information_need_count = qrels_bundle

    scores_df = _compute_scores(
        qrels_df=qrels_df,
        query_queries=query_queries,
        info_need_queries=info_need_queries,
        top_k=args.top_k,
        qrels_count=qrels_count,
        information_need_count=information_need_count,
        method_order=EXTENDED_METHOD_ORDER,
    )

    core_scores_df = scores_df[scores_df["method"].isin(CORE_METHOD_ORDER)].copy()
    extended_scores_df = scores_df[scores_df["method"].isin(EXTENDED_METHOD_ORDER)].copy()

    csv_path = output_dir / "form_submissions_grouped_metrics.csv"
    png_path = output_dir / "form_submissions_grouped_metrics.png"
    extended_csv_path = output_dir / "form_submissions_grouped_metrics_extended.csv"
    extended_png_path = output_dir / "form_submissions_grouped_metrics_extended.png"

    core_scores_df.to_csv(csv_path, index=False)
    extended_scores_df.to_csv(extended_csv_path, index=False)
    _plot_grouped_bars(
        scores_df=core_scores_df,
        output_path=png_path,
        qrels_count=qrels_count,
        information_need_count=information_need_count,
        method_order=CORE_METHOD_ORDER,
        title="Form submissions: grouped metrics by retrieval method",
    )
    _plot_grouped_bars(
        scores_df=extended_scores_df,
        output_path=extended_png_path,
        qrels_count=qrels_count,
        information_need_count=information_need_count,
        method_order=EXTENDED_METHOD_ORDER,
        title="Form submissions: grouped metrics by retrieval method (extended)",
    )

    print(f"Using {qrels_count} unique qrels across {information_need_count} information needs.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {extended_csv_path}")
    print(f"Wrote {extended_png_path}")


if __name__ == "__main__":
    main()
