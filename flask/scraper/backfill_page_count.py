"""Backfill page_count metadata for existing parsed document JSON files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from scraper.parsers import extract_page_count


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate metadata.page_count for parsed Docplus JSON files.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip files that already have a metadata.page_count value.",
    )
    return parser.parse_args(argv)


def backfill_page_counts(parsed_dir: str, only_missing: bool = False) -> tuple[int, int]:
    updated = 0
    scanned = 0

    for path in sorted(Path(parsed_dir).glob("*.json")):
        scanned += 1
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            payload["metadata"] = metadata

        if only_missing and metadata.get("page_count") is not None:
            continue

        binary_path = payload.get("binary_path")
        if not isinstance(binary_path, str) or not binary_path.strip():
            continue
        if not os.path.exists(binary_path):
            continue

        page_count = extract_page_count(binary_path)
        metadata["page_count"] = page_count

        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        updated += 1

    return updated, scanned


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    updated, scanned = backfill_page_counts(
        parsed_dir=args.parsed_dir,
        only_missing=args.only_missing,
    )
    print(f"Updated {updated} parsed JSON files out of {scanned} scanned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
