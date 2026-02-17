"""Backfill document titles into an existing scrape summary.json."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse


def extract_title_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    query = parse_qs(parsed.query)
    for key in ("filename", "file", "name"):
        values = query.get(key)
        if values:
            candidate = unquote(values[0]).strip()
            if candidate:
                return candidate

    last = parsed.path.split("/")[-1].strip()
    if last and last.lower() != "getdocument":
        return unquote(last)
    return None


def normalize_title(title: str) -> str:
    stem, ext = os.path.splitext(title)
    return stem if ext else title


def enrich_summary(summary_path: str) -> tuple[int, int]:
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError("summary.json is missing a valid 'documents' list.")

    updated = 0
    skipped = 0
    for document in documents:
        if not isinstance(document, dict):
            skipped += 1
            continue

        url = document.get("url")
        if not isinstance(url, str) or not url.strip():
            skipped += 1
            continue

        title = extract_title_from_url(url.strip())
        if not title:
            skipped += 1
            continue

        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            document["metadata"] = metadata

        cleaned_title = normalize_title(title)
        metadata["title"] = cleaned_title
        metadata["title_source"] = "url_filename"
        updated += 1

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return updated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add document titles to summary.json metadata.")
    parser.add_argument(
        "--summary-path",
        default="output/summary.json",
        help="Path to the scraper summary JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    updated, skipped = enrich_summary(args.summary_path)
    print(f"Updated metadata.title for {updated} documents (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
