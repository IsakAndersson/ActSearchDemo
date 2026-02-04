"""Flask server for testing Docplus search methods."""
from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, render_template, request

from search.bm25_search import bm25_search
from search.vector_index import DEFAULT_MODEL, query_index


def _get_env_default(name: str, fallback: str) -> str:
    value = os.getenv(name)
    return value if value else fallback


app = Flask(__name__)


@app.get("/")
def index() -> str:
    defaults = {
        "parsed_dir": _get_env_default("DOCPLUS_PARSED_DIR", "output/parsed"),
        "index_path": _get_env_default("DOCPLUS_INDEX_PATH", "output/vector_index/docplus.faiss"),
        "metadata_path": _get_env_default(
            "DOCPLUS_METADATA_PATH", "output/vector_index/docplus_metadata.jsonl"
        ),
        "model_name": _get_env_default("DOCPLUS_MODEL_NAME", DEFAULT_MODEL),
        "device": _get_env_default("DOCPLUS_DEVICE", "auto"),
        "top_k": _get_env_default("DOCPLUS_TOP_K", "5"),
    }
    return render_template("index.html", defaults=defaults, results=None, errors=None)


@app.post("/search")
def search() -> str:
    method = request.form.get("method", "bm25")
    query = (request.form.get("query") or "").strip()

    defaults = {
        "parsed_dir": request.form.get("parsed_dir", "output/parsed"),
        "index_path": request.form.get("index_path", "output/vector_index/docplus.faiss"),
        "metadata_path": request.form.get("metadata_path", "output/vector_index/docplus_metadata.jsonl"),
        "model_name": request.form.get("model_name", DEFAULT_MODEL),
        "device": request.form.get("device", "auto"),
        "top_k": request.form.get("top_k", "5"),
    }

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

    return render_template(
        "index.html",
        defaults=defaults,
        results=results,
        errors=errors or None,
        query=query,
        method=method,
    )


if __name__ == "__main__":
    app.run(debug=True)
