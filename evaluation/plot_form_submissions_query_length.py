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
from ir_measures import nDCG

try:
    from .evaluation import EVALUATION_DIR
    from .search_adapter import bm25_search, dense_e5_search, hybrid_e5_search
except ImportError:
    from evaluation import EVALUATION_DIR
    from search_adapter import bm25_search, dense_e5_search, hybrid_e5_search


DEFAULT_QRELS_PATH = EVALUATION_DIR / "qrels_from_form_submissions_clean.csv"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots" / "form_submissions_query_length"

METHODS: Dict[str, Callable[[str, int], List[Tuple[str, float]]]] = {
    "bm25": bm25_search,
    "dense_e5": dense_e5_search,
    "hybrid_e5": hybrid_e5_search,
}

METHOD_LABELS = {
    "bm25": "BM25",
    "dense_e5": "Dense E5",
    "hybrid_e5": "Hybrid E5",
}

METHOD_ORDER = ["bm25", "dense_e5", "hybrid_e5"]


def _load_clean_form_submissions_qrels(path: Path) -> tuple[pd.DataFrame, list[tuple[str, str]], list[tuple[str, str]]]:
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

    query_rows: list[dict[str, object]] = []
    info_need_rows: list[dict[str, object]] = []
    query_queries: list[tuple[str, str]] = []
    info_need_queries: list[tuple[str, str]] = []

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

    qrels_df = pd.concat(
        [
            pd.DataFrame(query_rows).assign(source_text_type="query"),
            pd.DataFrame(info_need_rows).assign(source_text_type="information_need"),
        ],
        ignore_index=True,
    )
    return qrels_df, query_queries, info_need_queries


def _build_combined_queries(
    query_queries: Sequence[tuple[str, str]],
    info_need_queries: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    combined_queries: list[dict[str, str]] = []
    for query_id, query_text in query_queries:
        combined_queries.append(
            {"query_id": query_id, "query_text": query_text, "source_text_type": "query"}
        )
    for query_id, query_text in info_need_queries:
        combined_queries.append(
            {"query_id": query_id, "query_text": query_text, "source_text_type": "information_need"}
        )
    return combined_queries


def _build_run_df(
    queries: Sequence[tuple[str, str]],
    search_fn: Callable[[str, int], List[Tuple[str, float]]],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_id, query_text in queries:
        results = search_fn(query_text, top_k)
        for doc_id, score in results:
            rows.append({"query_id": query_id, "doc_id": str(doc_id), "score": float(score)})
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])


def _compute_per_query_ndcg(
    qrels_df: pd.DataFrame,
    combined_queries: Sequence[dict[str, str]],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    query_frame = pd.DataFrame(combined_queries)

    for method_key in METHOD_ORDER:
        search_fn = METHODS[method_key]
        runs = _build_run_df(
            [(row["query_id"], row["query_text"]) for row in combined_queries],
            search_fn,
            top_k=top_k,
        )
        metric_results = list(
            ir_measures.iter_calc(
                [nDCG @ 10],
                qrels_df[["query_id", "doc_id", "relevance"]],
                runs,
            )
        )
        ndcg_by_query_id = {str(result.query_id): float(result.value) for result in metric_results}

        for row in query_frame.itertuples(index=False):
            query_text = str(row.query_text)
            rows.append(
                {
                    "method": method_key,
                    "method_label": METHOD_LABELS[method_key],
                    "query_id": str(row.query_id),
                    "query_text": query_text,
                    "source_text_type": str(row.source_text_type),
                    "word_count": len(query_text.split()),
                    "character_count": len(query_text),
                    "ndcg@10": ndcg_by_query_id.get(str(row.query_id), 0.0),
                }
            )
    return pd.DataFrame(rows)


def _plot_ndcg_vs_length(
    per_query_df: pd.DataFrame,
    output_path: Path,
    x_column: str,
    x_label: str,
    title: str,
) -> None:
    color_map = {
        "bm25": "#4c78a8",
        "dense_e5": "#f58518",
        "hybrid_e5": "#54a24b",
    }
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    for method_key in METHOD_ORDER:
        method_df = per_query_df[per_query_df["method"] == method_key].copy()
        if method_df.empty:
            continue
        ax.scatter(
            method_df[x_column],
            method_df["ndcg@10"],
            s=70,
            alpha=0.75,
            color=color_map[method_key],
            edgecolors="#1f2933",
            linewidths=0.5,
            label=METHOD_LABELS[method_key],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(axis="both", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_ndcg_vs_length_quantiles(
    per_query_df: pd.DataFrame,
    output_path: Path,
    x_column: str,
    x_label: str,
    title: str,
    quantiles: int = 5,
) -> None:
    color_map = {
        "bm25": "#4c78a8",
        "dense_e5": "#f58518",
        "hybrid_e5": "#54a24b",
    }
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    for method_key in METHOD_ORDER:
        method_df = per_query_df[per_query_df["method"] == method_key].copy()
        if method_df.empty:
            continue
        unique_count = method_df[x_column].nunique()
        q = min(quantiles, unique_count)
        if q < 2:
            continue

        method_df["length_bin"] = pd.qcut(method_df[x_column], q=q, duplicates="drop")
        if method_df["length_bin"].nunique() < 2:
            continue

        grouped = (
            method_df.groupby("length_bin", observed=True)
            .agg(
                mean_length=(x_column, "mean"),
                mean_ndcg=("ndcg@10", "mean"),
                count=("ndcg@10", "size"),
            )
            .reset_index()
            .sort_values("mean_length", kind="stable")
        )

        ax.plot(
            grouped["mean_length"],
            grouped["mean_ndcg"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=color_map[method_key],
            label=METHOD_LABELS[method_key],
        )

        for row in grouped.itertuples(index=False):
            ax.annotate(
                f"n={int(row.count)}\n{str(row.length_bin)}",
                (row.mean_length, row.mean_ndcg),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
            )

    ax.set_xlabel(f"{x_label} (quantile-bin average)")
    ax.set_ylabel("Mean nDCG@10")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(axis="both", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create form submissions query-length plots for BM25, Dense E5, and Hybrid E5."
    )
    parser.add_argument(
        "--qrels-path",
        default=str(DEFAULT_QRELS_PATH),
        help="Path to qrels_from_form_submissions_clean.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots and CSV are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-k documents to retrieve for each method.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qrels_df, query_queries, info_need_queries = _load_clean_form_submissions_qrels(
        Path(args.qrels_path).resolve()
    )
    combined_queries = _build_combined_queries(query_queries, info_need_queries)
    per_query_ndcg_df = _compute_per_query_ndcg(
        qrels_df=qrels_df,
        combined_queries=combined_queries,
        top_k=args.top_k,
    )

    per_query_csv_path = output_dir / "form_submissions_ndcg_per_query.csv"
    ndcg_words_png_path = output_dir / "form_submissions_ndcg_vs_word_count.png"
    ndcg_characters_png_path = output_dir / "form_submissions_ndcg_vs_character_count.png"
    ndcg_words_quantiles_png_path = output_dir / "form_submissions_ndcg_vs_word_count_quantiles.png"
    ndcg_characters_quantiles_png_path = output_dir / "form_submissions_ndcg_vs_character_count_quantiles.png"

    per_query_ndcg_df.to_csv(per_query_csv_path, index=False)
    _plot_ndcg_vs_length(
        per_query_df=per_query_ndcg_df,
        output_path=ndcg_words_png_path,
        x_column="word_count",
        x_label="Number of words in query / information need",
        title="Form submissions: nDCG@10 vs word count",
    )
    _plot_ndcg_vs_length(
        per_query_df=per_query_ndcg_df,
        output_path=ndcg_characters_png_path,
        x_column="character_count",
        x_label="Number of characters in query / information need",
        title="Form submissions: nDCG@10 vs character count",
    )
    _plot_ndcg_vs_length_quantiles(
        per_query_df=per_query_ndcg_df,
        output_path=ndcg_words_quantiles_png_path,
        x_column="word_count",
        x_label="Number of words in query / information need",
        title="Form submissions: mean nDCG@10 vs word count quantile bins",
    )
    _plot_ndcg_vs_length_quantiles(
        per_query_df=per_query_ndcg_df,
        output_path=ndcg_characters_quantiles_png_path,
        x_column="character_count",
        x_label="Number of characters in query / information need",
        title="Form submissions: mean nDCG@10 vs character count quantile bins",
    )

    print(f"Wrote {per_query_csv_path}")
    print(f"Wrote {ndcg_words_png_path}")
    print(f"Wrote {ndcg_characters_png_path}")
    print(f"Wrote {ndcg_words_quantiles_png_path}")
    print(f"Wrote {ndcg_characters_quantiles_png_path}")


if __name__ == "__main__":
    main()
