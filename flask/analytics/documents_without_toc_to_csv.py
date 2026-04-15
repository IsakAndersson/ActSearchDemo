"""Export documents without a detected table of contents to CSV."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from document_structure import _extract_toc_entries, _parse_text_lines


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export documents without a detected table of contents to CSV.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="output/metadata",
        help="Directory containing metadata-only JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default="output/documents_without_toc.csv",
        help="Path where the CSV file will be written.",
    )
    return parser.parse_args(argv)


def _parse_publish_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        return datetime.strptime(trimmed, "%Y-%m-%d").date()
    except ValueError:
        return None


def _has_table_of_contents(text: object) -> bool:
    if not isinstance(text, str):
        return False
    toc_entries, _ = _extract_toc_entries(_parse_text_lines(text))
    return bool(toc_entries)


def _age_years_from_publish_date(publish_date: date | None, today: date) -> str:
    if publish_date is None:
        return ""
    return f"{((today - publish_date).days / 365.25):.2f}"


def _load_metadata_by_id(metadata_dir: str) -> dict[str, dict[str, object]]:
    metadata_by_id: dict[str, dict[str, object]] = {}
    for path in sorted(Path(metadata_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            metadata_by_id[path.stem] = payload
    return metadata_by_id


def export_documents_without_toc(parsed_dir: str, metadata_dir: str, output_path: str) -> int:
    today = date.today()
    metadata_by_id = _load_metadata_by_id(metadata_dir)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["URL", "Titel", "Antal sidor", "Ålder (år)"],
        )
        writer.writeheader()

        for path in sorted(Path(parsed_dir).glob("*.json")):
            with path.open("r", encoding="utf-8") as parsed_handle:
                payload = json.load(parsed_handle)

            if not isinstance(payload, dict):
                continue

            if _has_table_of_contents(payload.get("raw_text")):
                continue

            parsed_metadata = payload.get("metadata")
            parsed_metadata_dict = parsed_metadata if isinstance(parsed_metadata, dict) else {}

            metadata_payload = metadata_by_id.get(path.stem, {})
            metadata = metadata_payload.get("metadata") if isinstance(metadata_payload, dict) else {}
            metadata_dict = metadata if isinstance(metadata, dict) else {}

            publish_date = _parse_publish_date(metadata_dict.get("publish_date"))
            writer.writerow(
                {
                    "URL": metadata_payload.get("source_url", ""),
                    "Titel": metadata_dict.get("title") or parsed_metadata_dict.get("title") or "",
                    "Antal sidor": parsed_metadata_dict.get("page_count", ""),
                    "Ålder (år)": _age_years_from_publish_date(publish_date, today),
                }
            )
            row_count += 1

    return row_count


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    row_count = export_documents_without_toc(args.parsed_dir, args.metadata_dir, args.output_path)
    print(f"Wrote {row_count} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
