"""Search adapters for evaluation.

Exposes a unified search signature:
    search(query: str, top_k: int = 20) -> List[Tuple[str, float]]

Return format is List[(doc_id, score)].
"""
from __future__ import annotations

import os
import re
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

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
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".rtf", ".txt"}
LOG = logging.getLogger(__name__)


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
    live_base_url: str = "https://publikdocplus.regionuppsala.se/"
    live_search_path: str = "/Home/Search"
    live_timeout_seconds: int = 20
    live_max_pages: int = 1
    live_user_agent: str = "ActSearchEvaluation/1.0"
    sts_live_base_url: str = "https://sts.search.datatovalue.se/"
    sts_live_search_path: str = "/"
    sts_live_timeout_seconds: int = 20
    sts_live_max_pages: int = 1
    sts_live_user_agent: str = "ActSearchEvaluation/1.0"


DEFAULT_CONFIG = SearchConfig(
    parsed_dir=os.getenv("DOCPLUS_PARSED_DIR", SearchConfig.parsed_dir),
    index_path=os.getenv("DOCPLUS_INDEX_PATH", SearchConfig.index_path),
    metadata_path=os.getenv("DOCPLUS_METADATA_PATH", SearchConfig.metadata_path),
    model_name=os.getenv("DOCPLUS_MODEL_NAME", SearchConfig.model_name),
    e5_index_path=os.getenv("DOCPLUS_E5_INDEX_PATH", SearchConfig.e5_index_path),
    e5_metadata_path=os.getenv("DOCPLUS_E5_METADATA_PATH", SearchConfig.e5_metadata_path),
    e5_model_name=os.getenv("DOCPLUS_E5_MODEL_NAME", SearchConfig.e5_model_name),
    device=os.getenv("DOCPLUS_DEVICE", SearchConfig.device),
    live_base_url=os.getenv("DOCPLUS_LIVE_BASE_URL", SearchConfig.live_base_url),
    live_search_path=os.getenv("DOCPLUS_LIVE_SEARCH_PATH", SearchConfig.live_search_path),
    live_timeout_seconds=int(os.getenv("DOCPLUS_LIVE_TIMEOUT_SECONDS", str(SearchConfig.live_timeout_seconds))),
    live_max_pages=max(1, int(os.getenv("DOCPLUS_LIVE_MAX_PAGES", str(SearchConfig.live_max_pages)))),
    live_user_agent=os.getenv("DOCPLUS_LIVE_USER_AGENT", SearchConfig.live_user_agent),
    sts_live_base_url=os.getenv("DOCPLUS_STS_LIVE_BASE_URL", SearchConfig.sts_live_base_url),
    sts_live_search_path=os.getenv("DOCPLUS_STS_LIVE_SEARCH_PATH", SearchConfig.sts_live_search_path),
    sts_live_timeout_seconds=int(
        os.getenv("DOCPLUS_STS_LIVE_TIMEOUT_SECONDS", str(SearchConfig.sts_live_timeout_seconds))
    ),
    sts_live_max_pages=max(1, int(os.getenv("DOCPLUS_STS_LIVE_MAX_PAGES", str(SearchConfig.sts_live_max_pages)))),
    sts_live_user_agent=os.getenv("DOCPLUS_STS_LIVE_USER_AGENT", SearchConfig.sts_live_user_agent),
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


def _is_document_link(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    for ext in DOCUMENT_EXTENSIONS:
        if path.endswith(ext):
            return True
    if "getdocument" in path:
        return True

    query = parse_qs(parsed.query)
    for values in query.values():
        for value in values:
            candidate = value.lower()
            for ext in DOCUMENT_EXTENSIONS:
                if candidate.endswith(ext):
                    return True
    return False


def _extract_live_docplus_links(html: str, page_url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[Dict[str, str]] = []
    seen_urls: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute_url = urljoin(page_url, str(anchor["href"]).strip())
        if not _is_document_link(absolute_url):
            continue
        if absolute_url in seen_urls:
            continue
        seen_urls.add(absolute_url)
        title = anchor.get_text(" ", strip=True) or _filename_from_url(absolute_url)
        links.append({"source_url": absolute_url, "title": title})
    return links


def _extract_live_sts_links(html: str, page_url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[Dict[str, str]] = []
    seen_urls: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        absolute_url = urljoin(page_url, str(anchor["href"]).strip())
        is_docplus_link = "docplus" in absolute_url.lower() or "getdocument" in absolute_url.lower()
        if not _is_document_link(absolute_url) and not is_docplus_link:
            continue
        if absolute_url in seen_urls:
            continue
        seen_urls.add(absolute_url)
        title = anchor.get_text(" ", strip=True) or _filename_from_url(absolute_url)
        links.append({"source_url": absolute_url, "title": title})
    return links


def _sanitize_live_query(query: str) -> str:
    cleaned = re.sub(r"[^\w\sÅÄÖåäö-]", " ", query, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()
    return cleaned


def _fetch_live_search_page(
    session: requests.Session,
    search_url: str,
    query: str,
    page: int,
    timeout_seconds: int,
) -> Optional[str]:
    # Docplus can intermittently return 500 for some encoded inputs.
    # Try a few compatible query variants before giving up.
    variants: List[str] = [query]
    query_with_equals = f"{query}="
    if query_with_equals not in variants:
        variants.append(query_with_equals)
    sanitized = _sanitize_live_query(query)
    if sanitized and sanitized not in variants:
        variants.append(sanitized)
        sanitized_with_equals = f"{sanitized}="
        if sanitized_with_equals not in variants:
            variants.append(sanitized_with_equals)

    for candidate in variants:
        try:
            response = session.get(
                search_url,
                params={
                    "searchValue": candidate,
                    "oldFilter": "",
                    "facet": "",
                    "facetVal": "",
                    "page": page,
                },
                timeout=timeout_seconds,
            )
            if response.status_code >= 500:
                continue
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            continue

    LOG.warning("Docplus live search failed for query on page %s: %r", page, query)
    return None


def _fetch_sts_live_search_page(
    session: requests.Session,
    search_url: str,
    query: str,
    page: int,
    timeout_seconds: int,
) -> Optional[str]:
    query_variants: List[str] = [query]
    sanitized = _sanitize_live_query(query)
    if sanitized and sanitized not in query_variants:
        query_variants.append(sanitized)
    query_with_equals = f"{query}="
    if query_with_equals not in query_variants:
        query_variants.append(query_with_equals)

    param_templates = [
        {"q": "{query}", "page": "{page}"},
        {"query": "{query}", "page": "{page}"},
        {"search": "{query}", "page": "{page}"},
        {"searchValue": "{query}", "page": "{page}"},
        {"q": "{query}", "p": "{page}"},
        {"query": "{query}", "p": "{page}"},
    ]

    for candidate_query in query_variants:
        for template in param_templates:
            params = {
                key: (
                    str(page)
                    if value == "{page}"
                    else candidate_query
                )
                for key, value in template.items()
            }
            try:
                response = session.get(search_url, params=params, timeout=timeout_seconds)
                if response.status_code >= 500:
                    continue
                response.raise_for_status()
                if response.text.strip():
                    return response.text
            except requests.RequestException:
                continue

    LOG.warning("STS live search failed for query on page %s: %r", page, query)
    return None


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


def docplus_live_search(
    query: str,
    top_k: int = 20,
    config: SearchConfig = DEFAULT_CONFIG,
    session: Optional[requests.Session] = None,
) -> SearchResults:
    if not query.strip() or top_k <= 0:
        return []

    search_url = urljoin(config.live_base_url.rstrip("/") + "/", config.live_search_path.lstrip("/"))
    max_pages = max(config.live_max_pages, (top_k + 19) // 20)

    active_session = session or requests.Session()
    active_session.headers.update({"User-Agent": config.live_user_agent})

    ranked: SearchResults = []
    seen_doc_ids: set[str] = set()
    for page in range(1, max_pages + 1):
        html = _fetch_live_search_page(
            session=active_session,
            search_url=search_url,
            query=query,
            page=page,
            timeout_seconds=config.live_timeout_seconds,
        )
        if html is None:
            if page == 1:
                return []
            break
        page_links = _extract_live_docplus_links(html, search_url)
        if not page_links:
            break

        new_docs_on_page = 0
        for item in page_links:
            score = float(max(1, top_k - len(ranked)))
            result_payload: Dict[str, object] = {
                "metadata": {
                    "title": item.get("title", ""),
                    "source_url": item.get("source_url", ""),
                },
                "score": score,
            }
            doc_id = _extract_doc_id(result_payload)
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            ranked.append((doc_id, score))
            new_docs_on_page += 1
            if len(ranked) >= top_k:
                return ranked
        if new_docs_on_page == 0:
            break

    return ranked[:top_k]


def sts_live_search(
    query: str,
    top_k: int = 20,
    config: SearchConfig = DEFAULT_CONFIG,
    session: Optional[requests.Session] = None,
) -> SearchResults:
    if not query.strip() or top_k <= 0:
        return []

    search_url = urljoin(config.sts_live_base_url.rstrip("/") + "/", config.sts_live_search_path.lstrip("/"))
    max_pages = max(config.sts_live_max_pages, (top_k + 19) // 20)

    active_session = session or requests.Session()
    active_session.headers.update({"User-Agent": config.sts_live_user_agent})

    ranked: SearchResults = []
    seen_doc_ids: set[str] = set()
    for page in range(1, max_pages + 1):
        html = _fetch_sts_live_search_page(
            session=active_session,
            search_url=search_url,
            query=query,
            page=page,
            timeout_seconds=config.sts_live_timeout_seconds,
        )
        if html is None:
            if page == 1:
                return []
            break
        page_links = _extract_live_sts_links(html, search_url)
        if not page_links:
            break

        new_docs_on_page = 0
        for item in page_links:
            score = float(max(1, top_k - len(ranked)))
            result_payload: Dict[str, object] = {
                "metadata": {
                    "title": item.get("title", ""),
                    "source_url": item.get("source_url", ""),
                },
                "score": score,
            }
            doc_id = _extract_doc_id(result_payload)
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            ranked.append((doc_id, score))
            new_docs_on_page += 1
            if len(ranked) >= top_k:
                return ranked
        if new_docs_on_page == 0:
            break

    return ranked[:top_k]


def search(query: str, top_k: int = 20) -> SearchResults:
    """Default search entrypoint for evaluation (hybrid)."""
    return hybrid_search(query=query, top_k=top_k)
