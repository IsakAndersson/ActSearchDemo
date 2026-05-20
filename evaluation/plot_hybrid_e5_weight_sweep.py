from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import ir_measures
import matplotlib.pyplot as plt
import pandas as pd
from ir_measures import RR, Recall, nDCG

try:
    from .evaluation import EVALUATION_DIR
    from .search_adapter import DEFAULT_CONFIG, SearchConfig, hybrid_e5_search
except ImportError:
    from evaluation import EVALUATION_DIR
    from search_adapter import DEFAULT_CONFIG, SearchConfig, hybrid_e5_search


DEFAULT_QRELS_PATH = EVALUATION_DIR / "qrels_from_form_submissions_clean.csv"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots" / "hybrid_e5_weight_sweep"
DEFAULT_WEIGHTS: Sequence[Tuple[float, float]] = (
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20),
    (0.90, 0.10),
)
VARIANT_SPECS: Sequence[Tuple[str, str]] = (
    ("query", "query_id"),
    ("information_need", "information_need"),
)
METRIC_SPECS: Sequence[Tuple[str, object, str, str]] = (
    ("ndcg@10", nDCG @ 10, "nDCG@10", "#1f77b4"),
    ("rr@100", RR @ 100, "RR@100", "#d62728"),
    ("recall@20", Recall @ 20, "Recall@20", "#2ca02c"),
)
VARIANT_LINESTYLES = {
    "query": "-",
    "information_need": "--",
}
VARIANT_LABELS = {
    "query": "Query",
    "information_need": "Information need",
}


def _load_form_submission_data(qrels_path: str) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    qrels_df = pd.read_csv(qrels_path)
    required_columns = {"query_id", "information_need", "doc_id"}
    missing = required_columns - set(qrels_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in qrels CSV: {', '.join(sorted(missing))}")

    filtered = qrels_df.copy()
    if "relevance" in filtered.columns:
        filtered["relevance"] = pd.to_numeric(filtered["relevance"], errors="coerce").fillna(0)
        filtered = filtered[filtered["relevance"] > 0]
    else:
        filtered["relevance"] = 1

    for column in ("query_id", "information_need", "doc_id"):
        filtered[column] = filtered[column].fillna("").astype(str).str.strip()
    filtered = filtered[
        (filtered["query_id"] != "")
        & (filtered["information_need"] != "")
        & (filtered["doc_id"] != "")
    ].copy()
    filtered = filtered.drop_duplicates(
        subset=["query_id", "information_need", "doc_id"]
    ).reset_index(drop=True)

    queries_by_variant: dict[str, pd.Series] = {}
    for variant_name, query_column in VARIANT_SPECS:
        queries_by_variant[variant_name] = (
            filtered[query_column].drop_duplicates().reset_index(drop=True)
        )

    return filtered[["query_id", "information_need", "doc_id", "relevance"]], queries_by_variant


def _weight_label(e5_weight: float, bm25_weight: float) -> str:
    return f"{int(round(e5_weight * 100))}/{int(round(bm25_weight * 100))}"


def _build_run_df(
    queries: pd.Series,
    search_fn,
    top_k: int,
    progress_label: str,
) -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    total_queries = len(queries)
    for index, query in enumerate(queries, start=1):
        if index == 1 or index % 5 == 0 or index == total_queries:
            print(f"[evaluation] {progress_label}: query {index}/{total_queries}", flush=True)
        results = search_fn(str(query), top_k)
        for doc_id, score in results:
            rows.append(
                {
                    "query_id": str(query),
                    "doc_id": str(doc_id),
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])


def _compute_metrics(
    qrels_df: pd.DataFrame,
    queries_by_variant: dict[str, pd.Series],
    top_k: int,
    weights: Iterable[Tuple[float, float]],
    base_config: SearchConfig,
) -> pd.DataFrame:
    rows: List[dict[str, float | int | str]] = []

    for e5_weight, bm25_weight in weights:
        config = replace(
            base_config,
            hybrid_bm25_weight=float(bm25_weight),
            hybrid_e5_weight=float(e5_weight),
        )

        for variant_name, query_column in VARIANT_SPECS:
            variant_queries = queries_by_variant[variant_name]
            variant_qrels = qrels_df[[query_column, "doc_id", "relevance"]].rename(
                columns={query_column: "query_id"}
            )
            run = _build_run_df(
                queries=variant_queries,
                search_fn=lambda query, current_top_k: hybrid_e5_search(
                    query=query,
                    top_k=current_top_k,
                    config=config,
                ),
                top_k=top_k,
                progress_label=(
                    f"hybrid_e5 {variant_name} "
                    f"{int(round(e5_weight * 100))}/{int(round(bm25_weight * 100))}"
                ),
            )
            metric_values = ir_measures.calc_aggregate(
                [metric for _, metric, _, _ in METRIC_SPECS],
                variant_qrels,
                run,
            )
            row: dict[str, float | int | str] = {
                "variant": variant_name,
                "e5_weight": float(e5_weight),
                "bm25_weight": float(bm25_weight),
                "weight_label": _weight_label(e5_weight, bm25_weight),
                "e5_weight_percent": int(round(e5_weight * 100)),
                "bm25_weight_percent": int(round(bm25_weight * 100)),
            }
            for column_name, metric, _, _ in METRIC_SPECS:
                row[column_name] = float(metric_values[metric])
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["variant", "e5_weight"]).reset_index(drop=True)


def _plot_metrics(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 14), sharex=True)

    for ax, (column_name, metric, title, color) in zip(axes, METRIC_SPECS):
        for variant_name, _ in VARIANT_SPECS:
            variant_df = results_df[results_df["variant"] == variant_name]
            ax.plot(
                variant_df["e5_weight_percent"],
                variant_df[column_name],
                marker="o",
                linewidth=2,
                linestyle=VARIANT_LINESTYLES[variant_name],
                color=color,
                label=VARIANT_LABELS[variant_name],
            )
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("E5 weight / BM25 weight")
    query_df = results_df[results_df["variant"] == "query"]
    axes[-1].set_xticks(query_df["e5_weight_percent"])
    axes[-1].set_xticklabels(query_df["weight_label"])

    fig.suptitle("Hybrid E5/BM25 weight sweep on form submissions", y=0.995)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run hybrid_e5 weight sweeps on form-submissions qrels for both query and "
            "information_need, then plot the metrics."
        )
    )
    parser.add_argument(
        "--qrels-path",
        default=str(DEFAULT_QRELS_PATH),
        help="Path to form submissions qrels CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV results and plot image are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-k documents to retrieve. Must be at least 100 for RR@100.",
    )
    args = parser.parse_args()

    if args.top_k < 100:
        raise ValueError("--top-k must be at least 100 to compute RR@100 correctly.")

    output_dir = Path(args.output_dir).resolve()
    qrels_path = str(Path(args.qrels_path).resolve())

    qrels_df, queries_by_variant = _load_form_submission_data(qrels_path)
    results_df = _compute_metrics(
        qrels_df=qrels_df,
        queries_by_variant=queries_by_variant,
        top_k=args.top_k,
        weights=DEFAULT_WEIGHTS,
        base_config=DEFAULT_CONFIG,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hybrid_e5_weight_sweep_metrics.csv"
    png_path = output_dir / "hybrid_e5_weight_sweep.png"
    results_df.to_csv(csv_path, index=False)
    _plot_metrics(results_df, png_path)

    print(f"Saved metrics to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
