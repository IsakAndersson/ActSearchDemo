"""Backfill cleaned_text into parsed Docplus JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scraper.storage import clean_text


DEFAULT_PARSED_DIR = Path(__file__).resolve().parent / "output" / "parsed"


def update_file(path: Path, dry_run: bool = False) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    text = payload.get("text")
    if not isinstance(text, str):
        return False

    cleaned = clean_text(text)
    if payload.get("cleaned_text") == cleaned:
        return False

    updated_payload = {}
    inserted = False
    for key, value in payload.items():
        updated_payload[key] = value
        if key == "text":
            updated_payload["cleaned_text"] = cleaned
            inserted = True
    if not inserted:
        updated_payload["cleaned_text"] = cleaned

    if not dry_run:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(updated_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return True


def iter_json_files(parsed_dir: Path):
    yield from sorted(path for path in parsed_dir.iterdir() if path.suffix.lower() == ".json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add cleaned_text to parsed JSON files by removing embedded newlines and collapsing whitespace."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help=f"Directory containing parsed JSON files. Default: {DEFAULT_PARSED_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would change without writing them.",
    )
    args = parser.parse_args()

    parsed_dir = args.parsed_dir.resolve()
    if not parsed_dir.exists() or not parsed_dir.is_dir():
        parser.error(f"Parsed directory does not exist: {parsed_dir}")

    updated_count = 0
    scanned_count = 0
    for path in iter_json_files(parsed_dir):
        scanned_count += 1
        if update_file(path, dry_run=args.dry_run):
            updated_count += 1
            print(f"updated {path}")

    print(f"scanned={scanned_count} updated={updated_count} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
