"""Export selected Docplus metadata fields to a CSV table."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Optional

APPROVED_DATE_REGEX = re.compile(r"Godkänt den:\s*(?:\n\s*)*(\d{4}-\d{2}-\d{2})")

CSV_COLUMNS: list[tuple[str, str]] = [
    ("URL", "source_url"),
    ("Titel", "title"),
    ("Dokumentsamling", "document_collection"),
    ("Process", "process"),
    ("Publiceringsdatum", "publish_date"),
    ("Ämnesområde", "subject_area"),
    ("Handlingstyp", "type_of_action"),
    ("Gäller för verksamhet", "valid_for_area"),
    ("Dokumentversion", "version"),
    ("Företagsnyckelord", "tax_keyword"),
    ("Antal sidor", "page_count"),
    ("Godkänt den", "approved_date"),
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export selected metadata fields to one CSV row per document.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="output/metadata",
        help="Directory containing metadata-only JSON files.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files used for page_count and approved date.",
    )
    parser.add_argument(
        "--output-path",
        default="output/metadata_documents.csv",
        help="Path where the CSV file will be written.",
    )
    return parser.parse_args(argv)


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sorted_json_paths(directory: str) -> list[Path]:
    return sorted(Path(directory).glob("*.json"))


def _extract_approved_date_from_text(text: object, head_chars: int = 500) -> str:
    if not isinstance(text, str):
        return ""
    match = APPROVED_DATE_REGEX.search(text[:head_chars])
    if not match:
        return ""
    return match.group(1)


def _load_parsed_extras(parsed_dir: str) -> dict[str, dict[str, str]]:
    parsed_extras: dict[str, dict[str, str]] = {}
    for path in _sorted_json_paths(parsed_dir):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            continue

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        page_count = metadata_dict.get("page_count")
        parsed_extras[path.stem] = {
            "page_count": _normalize_cell(page_count),
            "approved_date": _extract_approved_date_from_text(payload.get("text")),
        }

    return parsed_extras


def _build_row(
    payload: dict[str, object],
    parsed_extra: dict[str, str],
) -> dict[str, str]:
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    source_values = {
        "source_url": payload.get("source_url"),
        "title": metadata_dict.get("title"),
        "document_collection": metadata_dict.get("document_collection"),
        "process": metadata_dict.get("process"),
        "publish_date": metadata_dict.get("publish_date"),
        "subject_area": metadata_dict.get("subject_area"),
        "type_of_action": metadata_dict.get("type_of_action"),
        "valid_for_area": metadata_dict.get("valid_for_area"),
        "version": metadata_dict.get("version"),
        "tax_keyword": metadata_dict.get("tax_keyword"),
        "page_count": parsed_extra.get("page_count", ""),
        "approved_date": parsed_extra.get("approved_date", ""),
    }

    return {
        column_name: _normalize_cell(source_values[source_key])
        for column_name, source_key in CSV_COLUMNS
    }


def export_metadata_csv(metadata_dir: str, parsed_dir: str, output_path: str) -> int:
    json_paths = _sorted_json_paths(metadata_dir)
    parsed_extras = _load_parsed_extras(parsed_dir)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [column_name for column_name, _ in CSV_COLUMNS]

    row_count = 0
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in json_paths:
            with path.open("r", encoding="utf-8") as metadata_handle:
                payload = json.load(metadata_handle)

            if not isinstance(payload, dict):
                continue

            writer.writerow(_build_row(payload, parsed_extras.get(path.stem, {})))
            row_count += 1

    return row_count


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    row_count = export_metadata_csv(args.metadata_dir, args.parsed_dir, args.output_path)
    print(f"Wrote {row_count} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
