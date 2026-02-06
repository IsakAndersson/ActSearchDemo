"""Flask API server for Docplus BM25 and vector search."""
from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from search.bm25_search import bm25_search
from search.vector_index import DEFAULT_MODEL, query_index


def _get_env_default(name: str, fallback: str) -> str:
    value = os.getenv(name)
    return value if value else fallback


app = Flask(__name__)


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
        "model_name": str(
            payload.get("model_name")
            or _get_env_default("DOCPLUS_MODEL_NAME", DEFAULT_MODEL)
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
            "endpoints": {"search": "/search (POST)"},
        }
    )


@app.route("/search", methods=["POST", "OPTIONS"])
def search() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or request.form.to_dict()
    method = str(payload.get("method", "bm25")).lower()
    query = str(payload.get("query") or "").strip()

    defaults = _defaults_from_payload(payload)

    errors: List[str] = []
    results: List[Dict[str, Any]] | None = None

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
        else:
            errors.append(f"Unknown method '{method}'.")

    status_code = 400 if errors else 200
    return jsonify(
        {
            "query": query,
            "method": method,
            "defaults": defaults,
            "results": results or [],
            "errors": errors,
        }
    ), status_code


if __name__ == "__main__":
    app.run(debug=True)
