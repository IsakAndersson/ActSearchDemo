from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import ir_measures
import matplotlib.pyplot as plt
import pandas as pd
from ir_measures import RR, Recall, nDCG

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
FLASK_DIR = PROJECT_ROOT / "flask"

if str(FLASK_DIR) not in sys.path:
    sys.path.insert(0, str(FLASK_DIR))

try:
    from .doc_id import normalize_doc_id
    from .search_adapter import DEFAULT_CONFIG, SearchConfig, bm25_search
except ImportError:
    from doc_id import normalize_doc_id
    from search_adapter import DEFAULT_CONFIG, SearchConfig, bm25_search


METRICS = [nDCG @ 10, RR @ 100, Recall @ 20]
METRIC_COLUMNS = {
    "nDCG@10": nDCG @ 10,
    "RR@100": RR @ 100,
    "Recall@20": Recall @ 20,
}
DEFAULT_QRELS_PATH = EVALUATION_DIR / "qrels_from_form_submissions_clean.csv"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots"


def _load_form_submission_qrels(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"query_id", "doc_id", "relevance", "information_need"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Qrels CSV saknar obligatoriska kolumner: {', '.join(sorted(missing))}"
            )
        for row in reader:
            query_text = str(row.get("query_id") or "").strip()
            info_need = str(row.get("information_need") or "").strip()
            doc_id = normalize_doc_id(str(row.get("doc_id") or ""))
            relevance_raw = str(row.get("relevance") or "").strip()
            if not query_text or not info_need or not doc_id or not relevance_raw:
                continue
            rows.append(
                {
                    "query_id": query_text,
                    "doc_id": doc_id,
                    "relevance": int(relevance_raw),
                    "information_need": info_need,
                }
            )

    if not rows:
        raise ValueError(f"Inga giltiga qrels-rader kunde läsas från {path}")

    qrels_df = pd.DataFrame(rows).drop_duplicates()
    return qrels_df


def _build_query_sets(qrels_df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    query_qrels = qrels_df[["query_id", "doc_id", "relevance"]].drop_duplicates().reset_index(drop=True)
    query_queries = pd.DataFrame({"query_id": query_qrels["query_id"].drop_duplicates().reset_index(drop=True)})

    info_need_qrels = (
        qrels_df[["information_need", "doc_id", "relevance"]]
        .rename(columns={"information_need": "query_id"})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    info_need_queries = pd.DataFrame(
        {"query_id": info_need_qrels["query_id"].drop_duplicates().reset_index(drop=True)}
    )

    return {
        "query": {"queries": query_queries, "qrels": query_qrels},
        "information_need": {"queries": info_need_queries, "qrels": info_need_qrels},
    }


def _build_run_df(
    queries: Iterable[str],
    search_fn,
    top_k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for query in queries:
        results = search_fn(str(query), top_k)
        for doc_id, score in results:
            normalized_doc_id = normalize_doc_id(doc_id)
            if not normalized_doc_id:
                continue
            rows.append(
                {
                    "query_id": str(query),
                    "doc_id": normalized_doc_id,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "score"])


def _calculate_metrics(qrels: pd.DataFrame, run: pd.DataFrame) -> Dict[str, float]:
    results = ir_measures.calc_aggregate(METRICS, qrels, run)
    return {
        "nDCG@10": float(results[nDCG @ 10]),
        "RR@100": float(results[RR @ 100]),
        "Recall@20": float(results[Recall @ 20]),
    }


def _make_bm25_config(
    *,
    parsed_dir: str,
    chunk_size: int,
    overlap: int,
    include_title_chunk: bool,
    use_chunking: bool,
) -> SearchConfig:
    return replace(
        DEFAULT_CONFIG,
        parsed_dir=parsed_dir,
        bm25_max_chars=chunk_size,
        bm25_overlap=overlap,
        bm25_include_title_chunk=include_title_chunk,
        bm25_use_chunking=use_chunking,
    )


def _evaluate_variant(
    *,
    label: str,
    config: SearchConfig,
    query_sets: Dict[str, Dict[str, pd.DataFrame]],
    top_k: int,
    chunk_size: int,
    overlap: int,
    use_chunking: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    search_fn = lambda query, k: bm25_search(query=query, top_k=k, config=config)

    for query_input, payload in query_sets.items():
        queries = payload["queries"]["query_id"].tolist()
        qrels = payload["qrels"]
        run = _build_run_df(queries=queries, search_fn=search_fn, top_k=top_k)
        metrics = _calculate_metrics(qrels=qrels, run=run)
        rows.append(
            {
                "variant": label,
                "query_input": query_input,
                "use_chunking": use_chunking,
                "chunk_size": chunk_size if use_chunking else None,
                "chunk_overlap": overlap if use_chunking else None,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def _plot_metrics(results_df: pd.DataFrame, output_path: Path) -> None:
    variants = results_df["variant"].drop_duplicates().tolist()
    query_order = ["query", "information_need"]
    metric_order = ["nDCG@10", "RR@100", "Recall@20"]
    colors = {
        variants[0]: "#1f77b4",
        variants[1]: "#ff7f0e" if len(variants) > 1 else "#ff7f0e",
    }

    fig, axes = plt.subplots(1, len(metric_order), figsize=(13, 4.8), sharey=False)
    bar_width = 0.34
    x_positions = list(range(len(query_order)))

    for metric_index, metric_name in enumerate(metric_order):
        ax = axes[metric_index]
        for variant_index, variant in enumerate(variants):
            metric_values: List[float] = []
            for query_input in query_order:
                match = results_df[
                    (results_df["variant"] == variant) & (results_df["query_input"] == query_input)
                ]
                value = float(match.iloc[0][metric_name]) if not match.empty else 0.0
                metric_values.append(value)
            offsets = [x + (variant_index - (len(variants) - 1) / 2) * bar_width for x in x_positions]
            bars = ax.bar(
                offsets,
                metric_values,
                width=bar_width,
                label=variant,
                color=colors.get(variant, "#4c78a8"),
                edgecolor="#2f2f2f",
                linewidth=0.7,
            )
            for bar, value in zip(bars, metric_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Query", "Information need"])
        ax.set_title(metric_name)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    fig.suptitle("BM25: chunking vs hela dokumentet", fontsize=13)
    fig.legend(loc="upper center", ncol=max(1, len(variants)), bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plotta BM25-resultat för nDCG@10, RR@100 och Recall@20 "
            "med chunking respektive hela dokumentet för query och information need."
        )
    )
    parser.add_argument(
        "--qrels-path",
        default=str(DEFAULT_QRELS_PATH),
        help="CSV med form submissions-qrels som innehåller query_id, information_need och doc_id.",
    )
    parser.add_argument(
        "--parsed-dir",
        default=str(FLASK_DIR / "output" / "parsed"),
        help="Katalog med parsade dokument.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Katalog där CSV och figur sparas.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Chunkstorlek för BM25-varianten med chunking.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap för BM25-varianten med chunking.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Antal dokument att hämta per fråga. Bör vara minst 100 för RR@100.",
    )
    parser.add_argument(
        "--include-title-chunk",
        action="store_true",
        default=True,
        help="Behåll title chunks för BM25-körningarna.",
    )
    parser.add_argument(
        "--no-title-chunk",
        action="store_false",
        dest="include_title_chunk",
        help="Stäng av title chunks.",
    )
    args = parser.parse_args()

    if args.top_k < 100:
        raise ValueError("--top-k måste vara minst 100 för att RR@100 ska bli korrekt.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size måste vara > 0.")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap måste vara >= 0.")

    qrels_path = Path(args.qrels_path).resolve()
    parsed_dir = str(Path(args.parsed_dir).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qrels_df = _load_form_submission_qrels(qrels_path)
    query_sets = _build_query_sets(qrels_df)

    chunked_config = _make_bm25_config(
        parsed_dir=parsed_dir,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        include_title_chunk=args.include_title_chunk,
        use_chunking=True,
    )
    full_doc_config = _make_bm25_config(
        parsed_dir=parsed_dir,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        include_title_chunk=args.include_title_chunk,
        use_chunking=False,
    )

    chunked_label = f"Chunking ({args.chunk_size}/{args.chunk_overlap})"
    full_doc_label = "Hela dokumentet"

    results_df = pd.concat(
        [
            _evaluate_variant(
                label=chunked_label,
                config=chunked_config,
                query_sets=query_sets,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                use_chunking=True,
            ),
            _evaluate_variant(
                label=full_doc_label,
                config=full_doc_config,
                query_sets=query_sets,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
                use_chunking=False,
            ),
        ],
        ignore_index=True,
    )

    csv_path = output_dir / "bm25_chunking_vs_document_metrics.csv"
    png_path = output_dir / "bm25_chunking_vs_document_metrics.png"
    results_df.to_csv(csv_path, index=False)
    _plot_metrics(results_df=results_df, output_path=png_path)

    print(f"Skrev metrics till: {csv_path}")
    print(f"Skrev figur till:   {png_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
