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
    from .evaluation import EVALUATION_DIR, build_run_df, load_qrels_dataset
    from .search_adapter import DEFAULT_CONFIG, SearchConfig, hybrid_e5_search
except ImportError:
    from evaluation import EVALUATION_DIR, build_run_df, load_qrels_dataset
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


def _load_form_submission_queries(qrels_path: str) -> tuple[pd.DataFrame, pd.Series]:
    frames = load_qrels_dataset(qrels_source="form_submissions", qrels_path=qrels_path)
    qrels_frames: List[pd.DataFrame] = []
    query_frames: List[pd.DataFrame] = []
    for frame in frames.values():
        qrels_frames.append(frame["qrels"][["query_id", "doc_id", "relevance"]].copy())
        query_frames.append(frame["queries"][["query_id"]].copy())

    qrels = pd.concat(qrels_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    queries = pd.concat(query_frames, ignore_index=True)["query_id"].drop_duplicates().reset_index(drop=True)
    return qrels, queries


def _weight_label(e5_weight: float, bm25_weight: float) -> str:
    return f"{int(round(e5_weight * 100))}/{int(round(bm25_weight * 100))}"


def _compute_metrics(
    qrels: pd.DataFrame,
    queries: pd.Series,
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
        run = build_run_df(
            queries=queries,
            search_fn=lambda query, current_top_k: hybrid_e5_search(
                query=query,
                top_k=current_top_k,
                config=config,
            ),
            top_k=top_k,
            progress_label=f"hybrid_e5 {int(round(e5_weight * 100))}/{int(round(bm25_weight * 100))}",
        )
        metrics = ir_measures.calc_aggregate(
            [nDCG @ 10, RR @ 100, Recall @ 20],
            qrels,
            run,
        )
        rows.append(
            {
                "e5_weight": float(e5_weight),
                "bm25_weight": float(bm25_weight),
                "weight_label": _weight_label(e5_weight, bm25_weight),
                "e5_weight_percent": int(round(e5_weight * 100)),
                "bm25_weight_percent": int(round(bm25_weight * 100)),
                "ndcg@10": float(metrics[nDCG @ 10]),
                "rr@100": float(metrics[RR @ 100]),
                "recall@20": float(metrics[Recall @ 20]),
            }
        )

    return pd.DataFrame(rows).sort_values("e5_weight").reset_index(drop=True)


def _plot_metrics(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    line_specs = [
        ("ndcg@10", "nDCG@10", "#1f77b4"),
        ("rr@100", "RR@100", "#d62728"),
        ("recall@20", "Recall@20", "#2ca02c"),
    ]

    for column, label, color in line_specs:
        ax.plot(
            results_df["e5_weight_percent"],
            results_df[column],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

    ax.set_title("Hybrid E5/BM25 weight sweep on form submissions")
    ax.set_xlabel("E5 weight (%)")
    ax.set_ylabel("Metric value")
    ax.set_xticks(results_df["e5_weight_percent"])
    ax.set_xticklabels(results_df["weight_label"])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hybrid_e5 weight sweeps on form-submissions qrels and plot metrics."
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

    qrels, queries = _load_form_submission_queries(qrels_path)
    results_df = _compute_metrics(
        qrels=qrels,
        queries=queries,
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
