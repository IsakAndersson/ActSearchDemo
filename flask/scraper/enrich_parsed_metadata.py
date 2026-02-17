"""Backfill document name/title metadata into parsed JSON files."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, unquote, urlparse


def extract_document_name_from_url(url: str) -> str:
    parsed = urlparse(url)
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
    return ""


def title_from_document_name(document_name: str) -> str:
    stem, ext = os.path.splitext(document_name.strip())
    if stem and ext:
        return stem
    return document_name.strip()


def enrich_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        payload["metadata"] = metadata

    source_url = metadata.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        return False

    document_name = extract_document_name_from_url(source_url.strip())
    if not document_name:
        return False

    metadata["document_name"] = document_name
    metadata["title"] = title_from_document_name(document_name)
    metadata["title_source"] = "url_filename"

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return True


def enrich_directory(parsed_dir: str) -> Tuple[int, int]:
    updated = 0
    skipped = 0
    for filename in sorted(os.listdir(parsed_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(parsed_dir, filename)
        if enrich_file(path):
            updated += 1
        else:
            skipped += 1
    return updated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill document metadata in parsed JSON files.")
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Path to parsed JSON directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    updated, skipped = enrich_directory(args.parsed_dir)
    print(f"Updated {updated} parsed files (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
