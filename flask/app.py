"""Flask API server for Docplus BM25 and vector search."""
from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

from flask import Flask, jsonify, request

from search.bm25_search import bm25_search
from search.vector_index import DEFAULT_MODEL, VECTOR_MODEL_PROFILES, query_index


E5_PROFILE = VECTOR_MODEL_PROFILES["e5_large_instruct"]


def _get_env_default(name: str, fallback: str) -> str:
    value = os.getenv(name)
    return value if value else fallback


def _search_log_path() -> str:
    return _get_env_default("DOCPLUS_SEARCH_LOG_PATH", "output/logs/search_events.csv")


def _click_log_path() -> str:
    return _get_env_default("DOCPLUS_CLICK_LOG_PATH", "output/logs/click_events.csv")


def _rating_log_path() -> str:
    return _get_env_default("DOCPLUS_RATING_LOG_PATH", "output/logs/rating_events.csv")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _metadata_value(metadata: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        text = _to_text(metadata.get(key))
        if text:
            return text
    return ""


def _extract_result_url(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    value = _metadata_value(metadata_dict, ["source_url", "url", "link", "href", "source"])
    if value.lower().startswith(("http://", "https://")):
        return value
    return ""


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    for key in ("filename", "file", "name"):
        values = query_params.get(key)
        if values:
            return unquote(values[0])

    segments = [segment for segment in parsed.path.split("/") if segment]
    if segments:
        candidate = unquote(segments[-1])
        if candidate.lower() != "getdocument":
            return candidate
    return ""


def _extract_result_title(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    title = _metadata_value(
        metadata_dict,
        ["title", "document_title", "doc_title", "name", "filename", "file_name"],
    )
    if title:
        return title

    url = _extract_result_url(result)
    if url:
        title = _filename_from_url(url)
        if title:
            return title

    source_path = _to_text(result.get("source_path"))
    if source_path:
        pieces = [segment for segment in source_path.split("/") if segment]
        if pieces:
            return pieces[-1]
    return ""


def _extract_score(result: Dict[str, Any]) -> str:
    score = result.get("score")
    if isinstance(score, (int, float)):
        return f"{float(score):.12g}"
    if score is None:
        return ""
    return _to_text(score)


def _append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with LOG_WRITE_LOCK:
        needs_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _result_key(result: Dict[str, Any]) -> Tuple[str, int]:
    source_path = _to_text(result.get("source_path"))
    chunk_id_raw = result.get("chunk_id")
    if isinstance(chunk_id_raw, int):
        chunk_id = chunk_id_raw
    else:
        chunk_id = -1
        chunk_id_text = _to_text(chunk_id_raw)
        if chunk_id_text:
            try:
                chunk_id = int(chunk_id_text)
            except ValueError:
                chunk_id = -1
    return source_path, chunk_id


def _rrf_hybrid(
    bm25_results: List[Dict[str, Any]],
    e5_results: List[Dict[str, Any]],
    top_k: int,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    e5_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    fused_scores: Dict[Tuple[str, int], float] = {}
    result_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for rank, result in enumerate(bm25_results, start=1):
        key = _result_key(result)
        fused_scores[key] = fused_scores.get(key, 0.0) + bm25_weight / (rrf_k + rank)
        result_lookup.setdefault(key, result)

    for rank, result in enumerate(e5_results, start=1):
        key = _result_key(result)
        fused_scores[key] = fused_scores.get(key, 0.0) + e5_weight / (rrf_k + rank)
        result_lookup.setdefault(key, result)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    merged: List[Dict[str, Any]] = []
    for key, fused_score in ranked[:top_k]:
        item = dict(result_lookup[key])
        item["score"] = float(fused_score)
        merged.append(item)
    return merged


def _log_search(
    search_id: str,
    query: str,
    requested_method: str,
    top_k: int,
    results: List[Dict[str, Any]],
    results_by_method: Dict[str, List[Dict[str, Any]]],
    errors: List[str],
) -> None:
    base = {
        "timestamp_utc": _utc_now(),
        "search_id": search_id,
        "query": query,
        "requested_method": requested_method,
        "top_k": top_k,
        "had_errors": "1" if errors else "0",
        "errors": " | ".join(errors),
        "client_ip": request.headers.get("X-Forwarded-For", request.remote_addr or ""),
        "user_agent": request.headers.get("User-Agent", ""),
    }

    methods_to_log: Dict[str, List[Dict[str, Any]]] = {}
    if results_by_method:
        methods_to_log = results_by_method
    elif requested_method in {"bm25", "vector", "vector_e5", "hybrid_e5"}:
        methods_to_log = {requested_method: results}

    if not methods_to_log:
        _append_csv_row(
            _search_log_path(),
            SEARCH_LOG_FIELDS,
            {
                **base,
                "result_method": requested_method,
                "result_count_for_method": 0,
            },
        )
        return

    for result_method, method_results in methods_to_log.items():
        if not method_results:
            _append_csv_row(
                _search_log_path(),
                SEARCH_LOG_FIELDS,
                {
                    **base,
                    "result_method": result_method,
                    "result_count_for_method": 0,
                },
            )
            continue

        result_count = len(method_results)
        for index, item in enumerate(method_results, start=1):
            _append_csv_row(
                _search_log_path(),
                SEARCH_LOG_FIELDS,
                {
                    **base,
                    "result_method": result_method,
                    "rank": index,
                    "score": _extract_score(item),
                    "title": _extract_result_title(item),
                    "url": _extract_result_url(item),
                    "chunk_type": _to_text(item.get("chunk_type")),
                    "source_path": _to_text(item.get("source_path")),
                    "result_count_for_method": result_count,
                },
            )


app = Flask(__name__)
LOG_WRITE_LOCK = threading.Lock()
SEARCH_LOG_FIELDS = [
    "timestamp_utc",
    "search_id",
    "query",
    "requested_method",
    "result_method",
    "top_k",
    "rank",
    "score",
    "title",
    "url",
    "chunk_type",
    "source_path",
    "result_count_for_method",
    "had_errors",
    "errors",
    "client_ip",
    "user_agent",
]
CLICK_LOG_FIELDS = [
    "timestamp_utc",
    "search_id",
    "query",
    "requested_method",
    "result_method",
    "rank",
    "score",
    "title",
    "url",
    "chunk_type",
    "source_path",
    "client_ip",
    "user_agent",
]
RATING_LOG_FIELDS = [
    "timestamp_utc",
    "search_id",
    "query",
    "requested_method",
    "result_method",
    "document",
    "title",
    "url",
    "source_path",
    "user_score",
    "client_ip",
    "user_agent",
]


@app.after_request
def add_cors_headers(response):  # type: ignore[no-untyped-def]
    response.headers["Access-Control-Allow-Origin"] = os.getenv(
        "DOCPLUS_ALLOWED_ORIGIN",
        "*",
    )
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _defaults_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    return {
        "parsed_dir": str(
            payload.get("parsed_dir")
            or _get_env_default("DOCPLUS_PARSED_DIR", "output/parsed")
        ),
        "index_path": str(
            payload.get("index_path")
            or _get_env_default("DOCPLUS_INDEX_PATH", "output/vector_index/docplus.faiss")
        ),
        "metadata_path": str(
            payload.get("metadata_path")
            or _get_env_default(
                "DOCPLUS_METADATA_PATH", "output/vector_index/docplus_metadata.jsonl"
            )
        ),
        "e5_index_path": str(
            payload.get("e5_index_path")
            or _get_env_default(
                "DOCPLUS_E5_INDEX_PATH",
                "output/vector_index_e5/docplus.faiss",
            )
        ),
        "e5_metadata_path": str(
            payload.get("e5_metadata_path")
            or _get_env_default(
                "DOCPLUS_E5_METADATA_PATH",
                "output/vector_index_e5/docplus_metadata.jsonl",
            )
        ),
        "model_name": str(
            payload.get("model_name")
            or _get_env_default("DOCPLUS_MODEL_NAME", DEFAULT_MODEL)
        ),
        "e5_model_name": str(
            payload.get("e5_model_name")
            or _get_env_default("DOCPLUS_E5_MODEL_NAME", E5_PROFILE.model_name)
        ),
        "device": str(payload.get("device") or _get_env_default("DOCPLUS_DEVICE", "auto")),
        "top_k": str(payload.get("top_k") or _get_env_default("DOCPLUS_TOP_K", "5")),
    }


@app.get("/")
def health() -> Any:
    return jsonify(
        {
            "ok": True,
            "message": "Docplus API is running.",
            "endpoints": {
                "search": "/search (POST)",
                "search_click": "/search/click (POST)",
                "search_rating": "/search/rating (POST)",
            },
        }
    )


@app.route("/search", methods=["POST", "OPTIONS"])
def search() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or request.form.to_dict()
    search_id = str(uuid4())
    method = str(payload.get("method", "hybrid_e5")).lower()
    query = str(payload.get("query") or "").strip()

    defaults = _defaults_from_payload(payload)

    errors: List[str] = []
    top_k = 5
    results: List[Dict[str, Any]] | None = None
    results_by_method: Dict[str, List[Dict[str, Any]]] | None = None

    if not query:
        errors.append("Query cannot be empty.")
    else:
        try:
            top_k = int(defaults["top_k"])
        except ValueError:
            top_k = 5
            errors.append("Top-k must be an integer; defaulted to 5.")

        if method == "bm25":
            try:
                results = bm25_search(
                    parsed_dir=defaults["parsed_dir"],
                    query=query,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"BM25 search failed: {exc}")
        elif method == "vector":
            try:
                results = query_index(
                    index_path=defaults["index_path"],
                    metadata_path=defaults["metadata_path"],
                    query=query,
                    model_name=defaults["model_name"],
                    top_k=top_k,
                    device_preference=defaults["device"],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Vector search failed: {exc}")
        elif method == "vector_e5":
            try:
                results = query_index(
                    index_path=defaults["e5_index_path"],
                    metadata_path=defaults["e5_metadata_path"],
                    query=query,
                    model_name=defaults["e5_model_name"],
                    top_k=top_k,
                    device_preference=defaults["device"],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Vector E5 search failed: {exc}")
        elif method == "hybrid_e5":
            candidate_k = max(top_k * 3, top_k)
            bm25_results: List[Dict[str, Any]] = []
            e5_results: List[Dict[str, Any]] = []
            try:
                bm25_results = bm25_search(
                    parsed_dir=defaults["parsed_dir"],
                    query=query,
                    top_k=candidate_k,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"BM25 search failed: {exc}")
            try:
                e5_results = query_index(
                    index_path=defaults["e5_index_path"],
                    metadata_path=defaults["e5_metadata_path"],
                    query=query,
                    model_name=defaults["e5_model_name"],
                    top_k=candidate_k,
                    device_preference=defaults["device"],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Vector E5 search failed: {exc}")
            results = _rrf_hybrid(
                bm25_results=bm25_results,
                e5_results=e5_results,
                top_k=top_k,
            )
        elif method == "all":
            results_by_method = {}
            try:
                results_by_method["bm25"] = bm25_search(
                    parsed_dir=defaults["parsed_dir"],
                    query=query,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"BM25 search failed: {exc}")
                results_by_method["bm25"] = []
            try:
                results_by_method["vector"] = query_index(
                    index_path=defaults["index_path"],
                    metadata_path=defaults["metadata_path"],
                    query=query,
                    model_name=defaults["model_name"],
                    top_k=top_k,
                    device_preference=defaults["device"],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Vector search failed: {exc}")
                results_by_method["vector"] = []
            try:
                results_by_method["vector_e5"] = query_index(
                    index_path=defaults["e5_index_path"],
                    metadata_path=defaults["e5_metadata_path"],
                    query=query,
                    model_name=defaults["e5_model_name"],
                    top_k=top_k,
                    device_preference=defaults["device"],
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Vector E5 search failed: {exc}")
                results_by_method["vector_e5"] = []
            results_by_method["hybrid_e5"] = _rrf_hybrid(
                bm25_results=results_by_method.get("bm25", []),
                e5_results=results_by_method.get("vector_e5", []),
                top_k=top_k,
            )
        else:
            errors.append(f"Unknown method '{method}'.")

    _log_search(
        search_id=search_id,
        query=query,
        requested_method=method,
        top_k=top_k,
        results=results or [],
        results_by_method=results_by_method or {},
        errors=errors,
    )

    status_code = 400 if errors else 200
    return jsonify(
        {
            "search_id": search_id,
            "query": query,
            "method": method,
            "defaults": defaults,
            "results": results or [],
            "results_by_method": results_by_method or {},
            "errors": errors,
        }
    ), status_code


@app.route("/search/click", methods=["POST", "OPTIONS"])
def search_click() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or request.form.to_dict()
    search_id = _to_text(payload.get("search_id"))
    if not search_id:
        return jsonify({"ok": False, "errors": ["search_id is required."]}), 400

    _append_csv_row(
        _click_log_path(),
        CLICK_LOG_FIELDS,
        {
            "timestamp_utc": _utc_now(),
            "search_id": search_id,
            "query": _to_text(payload.get("query")),
            "requested_method": _to_text(payload.get("requested_method")),
            "result_method": _to_text(payload.get("result_method")),
            "rank": _to_text(payload.get("rank")),
            "score": _to_text(payload.get("score")),
            "title": _to_text(payload.get("title")),
            "url": _to_text(payload.get("url")),
            "chunk_type": _to_text(payload.get("chunk_type")),
            "source_path": _to_text(payload.get("source_path")),
            "client_ip": request.headers.get("X-Forwarded-For", request.remote_addr or ""),
            "user_agent": request.headers.get("User-Agent", ""),
        },
    )
    return jsonify({"ok": True})


@app.route("/search/rating", methods=["POST", "OPTIONS"])
def search_rating() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or request.form.to_dict()
    search_id = _to_text(payload.get("search_id"))
    if not search_id:
        return jsonify({"ok": False, "errors": ["search_id is required."]}), 400

    try:
        user_score = int(_to_text(payload.get("user_score")))
    except ValueError:
        return jsonify({"ok": False, "errors": ["user_score must be an integer between 1 and 5."]}), 400

    if user_score < 1 or user_score > 5:
        return jsonify({"ok": False, "errors": ["user_score must be an integer between 1 and 5."]}), 400

    title = _to_text(payload.get("title"))
    url = _to_text(payload.get("url"))
    source_path = _to_text(payload.get("source_path"))
    document = _to_text(payload.get("document")) or title or source_path or url

    _append_csv_row(
        _rating_log_path(),
        RATING_LOG_FIELDS,
        {
            "timestamp_utc": _utc_now(),
            "search_id": search_id,
            "query": _to_text(payload.get("query")),
            "requested_method": _to_text(payload.get("requested_method")),
            "result_method": _to_text(payload.get("result_method")),
            "document": document,
            "title": title,
            "url": url,
            "source_path": source_path,
            "user_score": str(user_score),
            "client_ip": request.headers.get("X-Forwarded-For", request.remote_addr or ""),
            "user_agent": request.headers.get("User-Agent", ""),
        },
    )
    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.getenv("DOCPLUS_HOST", "127.0.0.1")
    port = int(os.getenv("DOCPLUS_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
