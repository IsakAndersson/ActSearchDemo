"""Search adapters for evaluation.

Exposes a unified search signature:
    search(query: str, top_k: int = 20) -> List[Tuple[str, float]]

Return format is List[(doc_id, score)].
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

# Make flask/search importable from evaluation/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASK_DIR = PROJECT_ROOT / "flask"
if str(FLASK_DIR) not in sys.path:
    sys.path.insert(0, str(FLASK_DIR))

from search.bm25_search import bm25_search as _bm25_impl  # noqa: E402
try:
    from .doc_id import normalize_doc_id  # type: ignore[attr-defined] # noqa: E402
except ImportError:
    from doc_id import normalize_doc_id  # noqa: E402

SearchResults = List[Tuple[str, float]]


@dataclass(frozen=True)
class SearchConfig:
    parsed_dir: str = str(FLASK_DIR / "output" / "parsed")
    index_path: str = str(FLASK_DIR / "output" / "vector_index" / "docplus.faiss")
    metadata_path: str = str(FLASK_DIR / "output" / "vector_index" / "docplus_metadata.jsonl")
    model_name: str = "KBLab/bert-base-swedish-cased"
    e5_index_path: str = str(FLASK_DIR / "output" / "vector_index_e5" / "docplus.faiss")
    e5_metadata_path: str = str(FLASK_DIR / "output" / "vector_index_e5" / "docplus_metadata.jsonl")
    e5_model_name: str = "intfloat/multilingual-e5-large-instruct"
    device: str = "auto"


DEFAULT_CONFIG = SearchConfig(
    parsed_dir=os.getenv("DOCPLUS_PARSED_DIR", SearchConfig.parsed_dir),
    index_path=os.getenv("DOCPLUS_INDEX_PATH", SearchConfig.index_path),
    metadata_path=os.getenv("DOCPLUS_METADATA_PATH", SearchConfig.metadata_path),
    model_name=os.getenv("DOCPLUS_MODEL_NAME", SearchConfig.model_name),
    e5_index_path=os.getenv("DOCPLUS_E5_INDEX_PATH", SearchConfig.e5_index_path),
    e5_metadata_path=os.getenv("DOCPLUS_E5_METADATA_PATH", SearchConfig.e5_metadata_path),
    e5_model_name=os.getenv("DOCPLUS_E5_MODEL_NAME", SearchConfig.e5_model_name),
    device=os.getenv("DOCPLUS_DEVICE", SearchConfig.device),
)


def _metadata_value(metadata: Dict[str, object], keys: List[str]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    for key in ("filename", "file", "name"):
        values = query_params.get(key)
        if values:
            return unquote(values[0]).strip()

    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        return ""
    candidate = unquote(segments[-1]).strip()
    if candidate.lower() == "getdocument":
        return ""
    return candidate


def _extract_doc_id(result: Dict[str, object]) -> Optional[str]:
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    title = _metadata_value(
        metadata_dict,
        [
            "title",
            "document_title",
            "doc_title",
            "document_name",
            "name",
            "filename",
            "file_name",
        ],
    )
    if title:
        normalized = normalize_doc_id(title)
        return normalized if normalized else None

    source_url = _metadata_value(metadata_dict, ["source_url", "url", "link", "href", "source"])
    if source_url:
        filename = _filename_from_url(source_url)
        if filename:
            normalized = normalize_doc_id(filename)
            return normalized if normalized else None

    source_path = result.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        normalized = normalize_doc_id(Path(source_path).name)
        return normalized if normalized else None

    return None


def _dedupe_and_sort(results: List[Dict[str, object]], top_k: int) -> SearchResults:
    best_by_doc: Dict[str, float] = {}
    for item in results:
        doc_id = _extract_doc_id(item)
        score_raw = item.get("score")
        if not doc_id or not isinstance(score_raw, (int, float)):
            continue
        score = float(score_raw)
        prev = best_by_doc.get(doc_id)
        if prev is None or score > prev:
            best_by_doc[doc_id] = score

    ranked = sorted(best_by_doc.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def bm25_search(query: str, top_k: int = 20, config: SearchConfig = DEFAULT_CONFIG) -> SearchResults:
    if not query.strip() or top_k <= 0:
        return []
    raw_results = _bm25_impl(parsed_dir=config.parsed_dir, query=query, top_k=top_k)
    return _dedupe_and_sort(raw_results, top_k=top_k)


def dense_search(query: str, top_k: int = 20, config: SearchConfig = DEFAULT_CONFIG) -> SearchResults:
    if not query.strip() or top_k <= 0:
        return []
    try:
        from search.vector_index import query_index
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dense search requires vector dependencies (faiss/torch/transformers). "
            "Install requirements and rebuild index."
        ) from exc

    raw_results = query_index(
        index_path=config.index_path,
        metadata_path=config.metadata_path,
        query=query,
        model_name=config.model_name,
        top_k=top_k,
        device_preference=config.device,
    )
    return _dedupe_and_sort(raw_results, top_k=top_k)


def dense_e5_search(query: str, top_k: int = 20, config: SearchConfig = DEFAULT_CONFIG) -> SearchResults:
    if not query.strip() or top_k <= 0:
        return []
    try:
        from search.vector_index import query_index
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dense search requires vector dependencies (faiss/torch/transformers). "
            "Install requirements and rebuild index."
        ) from exc

    raw_results = query_index(
        index_path=config.e5_index_path,
        metadata_path=config.e5_metadata_path,
        query=query,
        model_name=config.e5_model_name,
        top_k=top_k,
        device_preference=config.device,
    )
    return _dedupe_and_sort(raw_results, top_k=top_k)


def hybrid_search(
    query: str,
    top_k: int = 20,
    config: SearchConfig = DEFAULT_CONFIG,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> SearchResults:
    if not query.strip() or top_k <= 0:
        return []

    candidate_k = max(top_k * 3, top_k)
    bm25_results = bm25_search(query=query, top_k=candidate_k, config=config)
    dense_results = dense_search(query=query, top_k=candidate_k, config=config)

    fused_scores: Dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + bm25_weight / (rrf_k + rank)

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + dense_weight / (rrf_k + rank)

    ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def hybrid_e5_search(
    query: str,
    top_k: int = 20,
    config: SearchConfig = DEFAULT_CONFIG,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> SearchResults:
    if not query.strip() or top_k <= 0:
        return []

    candidate_k = max(top_k * 3, top_k)
    bm25_results = bm25_search(query=query, top_k=candidate_k, config=config)
    dense_results = dense_e5_search(query=query, top_k=candidate_k, config=config)

    fused_scores: Dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + bm25_weight / (rrf_k + rank)

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + dense_weight / (rrf_k + rank)

    ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def search(query: str, top_k: int = 20) -> SearchResults:
    """Default search entrypoint for evaluation (hybrid)."""
    return hybrid_search(query=query, top_k=top_k)
