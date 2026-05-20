from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import ir_measures
import matplotlib.pyplot as plt
import pandas as pd
from ir_measures import nDCG

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
FLASK_DIR = PROJECT_ROOT / "flask"

if str(FLASK_DIR) not in sys.path:
    sys.path.insert(0, str(FLASK_DIR))

try:
    from .doc_id import normalize_doc_id
    from .search_adapter import DEFAULT_CONFIG, SearchConfig
except ImportError:
    from doc_id import normalize_doc_id
    from search_adapter import DEFAULT_CONFIG, SearchConfig

import document_structure as document_structure_module
from search import bm25_search as bm25_module


DEFAULT_QRELS_PATH = EVALUATION_DIR / "qrels_from_form_submissions_clean.csv"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots"


def _load_form_submission_qrels(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str | int]] = []
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
    return pd.DataFrame(rows).drop_duplicates()


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


def _payload_text(payload: dict) -> str:
    value = payload.get("text")
    return value if isinstance(value, str) else ""


def _whole_document_section(payload: dict, fallback_title: str | None) -> List[dict]:
    text = _payload_text(payload)
    cleaned = document_structure_module.clean_text(text)
    if not cleaned:
        return []
    heading = str(fallback_title or "Document").strip() or "Document"
    section = {
        "index": 0,
        "heading": heading,
        "title": heading,
        "level": 1,
        "page": None,
        "source": "whole_document",
        "raw_text": text.strip(),
        "text": cleaned,
        "path": [heading],
        "path_text": heading,
    }
    return [section]


def _get_sections_whole_document(payload: dict, fallback_title: str | None = None) -> List[dict]:
    return _whole_document_section(payload, fallback_title)


def _get_sections_toc_or_heuristic(payload: dict, fallback_title: str | None = None) -> List[dict]:
    return document_structure_module.get_document_sections(payload, fallback_title=fallback_title)


def _get_sections_toc_else_whole_document(payload: dict, fallback_title: str | None = None) -> List[dict]:
    text = _payload_text(payload)
    if not text.strip():
        return []
    lines = document_structure_module._parse_text_lines(text)
    toc_sections = document_structure_module._derive_sections_from_toc(lines, fallback_title=fallback_title)
    if toc_sections:
        return toc_sections
    return _whole_document_section(payload, fallback_title)


@contextmanager
def _patched_section_builder(section_builder: Callable[[dict, str | None], List[dict]]):
    original_builder = bm25_module.get_document_sections
    original_cache = dict(bm25_module._BM25_INDEX_CACHE)
    bm25_module.get_document_sections = section_builder
    bm25_module._BM25_INDEX_CACHE.clear()
    try:
        yield
    finally:
        bm25_module.get_document_sections = original_builder
        bm25_module._BM25_INDEX_CACHE.clear()
        bm25_module._BM25_INDEX_CACHE.update(original_cache)


def _build_run_df(
    queries: Iterable[str],
    search_fn: Callable[[str, int], List[tuple[str, float]]],
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


def _result_to_doc_id(result: dict) -> str:
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    for key in (
        "title",
        "document_title",
        "doc_title",
        "document_name",
        "name",
        "filename",
        "file_name",
    ):
        value = metadata_dict.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_doc_id(value)

    source_path = result.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        candidate = Path(source_path).name
        normalized = normalize_doc_id(candidate)
        if normalized:
            return normalized

    return ""


def _calculate_ndcg_at_10(qrels: pd.DataFrame, run: pd.DataFrame) -> float:
    results = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
    return float(results[nDCG @ 10])


def _make_bm25_config(
    *,
    parsed_dir: str,
    chunk_size: int,
    overlap: int,
    include_title_chunk: bool,
) -> SearchConfig:
    return replace(
        DEFAULT_CONFIG,
        parsed_dir=parsed_dir,
        bm25_max_chars=chunk_size,
        bm25_overlap=overlap,
        bm25_include_title_chunk=include_title_chunk,
        bm25_use_chunking=True,
    )


def _evaluate_scenario(
    *,
    label: str,
    section_builder: Callable[[dict, str | None], List[dict]],
    config: SearchConfig,
    query_sets: Dict[str, Dict[str, pd.DataFrame]],
    top_k: int,
    chunk_size: int,
    overlap: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def search_fn(query: str, k: int) -> List[tuple[str, float]]:
        rows: List[tuple[str, float]] = []
        for result in bm25_module.bm25_search(
            parsed_dir=config.parsed_dir,
            query=query,
            top_k=k,
            max_chars=config.bm25_max_chars,
            overlap=config.bm25_overlap,
            include_title_chunk=config.bm25_include_title_chunk,
            use_chunking=config.bm25_use_chunking,
        ):
            doc_id = _result_to_doc_id(result)
            if not doc_id:
                continue
            rows.append((doc_id, float(result.get("score") or 0.0)))
        return rows

    with _patched_section_builder(section_builder):
        for query_input, payload in query_sets.items():
            queries = payload["queries"]["query_id"].tolist()
            qrels = payload["qrels"]
            run = _build_run_df(queries=queries, search_fn=search_fn, top_k=top_k)
            ndcg_value = _calculate_ndcg_at_10(qrels=qrels, run=run)
            rows.append(
                {
                    "scenario": label,
                    "query_input": query_input,
                    "nDCG@10": ndcg_value,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                }
            )

    return pd.DataFrame(rows)


def _plot_ndcg(results_df: pd.DataFrame, output_path: Path, chunk_size: int, overlap: int) -> None:
    scenario_order = results_df["scenario"].drop_duplicates().tolist()
    query_order = ["query", "information_need"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x_positions = list(range(len(query_order)))
    bar_width = 0.23

    for scenario_index, scenario in enumerate(scenario_order):
        values: List[float] = []
        for query_input in query_order:
            match = results_df[
                (results_df["scenario"] == scenario) & (results_df["query_input"] == query_input)
            ]
            values.append(float(match.iloc[0]["nDCG@10"]) if not match.empty else 0.0)
        offsets = [x + (scenario_index - (len(scenario_order) - 1) / 2) * bar_width for x in x_positions]
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            color=colors[scenario_index % len(colors)],
            edgecolor="#2f2f2f",
            linewidth=0.7,
            label=scenario,
        )
        for bar, value in zip(bars, values):
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
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"BM25-scenarier, chunking {chunk_size}/{overlap}")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Jämför BM25 nDCG@10 för tre segmenteringsscenarier: "
            "hela dokumentet, TOC+heuristik, och TOC-annars-hela-dokumentet."
        )
    )
    parser.add_argument(
        "--qrels-path",
        default=str(DEFAULT_QRELS_PATH),
        help="CSV med query_id, information_need, doc_id och relevance.",
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
        help="Chunkstorlek för BM25.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap för BM25.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Antal chunkar att hämta per fråga.",
    )
    parser.add_argument(
        "--include-title-chunk",
        action="store_true",
        default=True,
        help="Behåll title chunks i BM25-indexet.",
    )
    parser.add_argument(
        "--no-title-chunk",
        action="store_false",
        dest="include_title_chunk",
        help="Stäng av title chunks.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k måste vara > 0.")
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
    config = _make_bm25_config(
        parsed_dir=parsed_dir,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        include_title_chunk=args.include_title_chunk,
    )

    results_df = pd.concat(
        [
            _evaluate_scenario(
                label="1. Hela dokumentet utan kapiteluppdelning",
                section_builder=_get_sections_whole_document,
                config=config,
                query_sets=query_sets,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            ),
            _evaluate_scenario(
                label="2. Kapiteluppdelning med TOC eller heuristik",
                section_builder=_get_sections_toc_or_heuristic,
                config=config,
                query_sets=query_sets,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            ),
            _evaluate_scenario(
                label="3. TOC-kapitel, annars hela dokumentet",
                section_builder=_get_sections_toc_else_whole_document,
                config=config,
                query_sets=query_sets,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            ),
        ],
        ignore_index=True,
    )

    csv_path = output_dir / "bm25_segmentation_scenarios_ndcg10.csv"
    png_path = output_dir / "bm25_segmentation_scenarios_ndcg10.png"
    results_df.to_csv(csv_path, index=False)
    _plot_ndcg(results_df=results_df, output_path=png_path, chunk_size=args.chunk_size, overlap=args.chunk_overlap)

    print(f"Skrev metrics till: {csv_path}")
    print(f"Skrev figur till:   {png_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
