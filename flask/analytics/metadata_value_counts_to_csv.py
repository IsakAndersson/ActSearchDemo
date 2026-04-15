"""Export selected Docplus metadata value counts to CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

SELECTED_FIELDS: list[tuple[str, str]] = [
    ("Dokumentsamling", "document_collection"),
    ("Process", "process"),
    ("Ämnesområde", "subject_area"),
    ("Handlingstyp", "type_of_action"),
    ("Gäller för verksamhet", "valid_for_area"),
    ("Företagsnyckelord", "tax_keyword"),
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unique selected Docplus metadata values and their counts to CSV.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="output/metadata",
        help="Directory containing metadata-only JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default="output/metadata_value_counts.csv",
        help="Path where the CSV file will be written.",
    )
    return parser.parse_args(argv)


def _split_values(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def collect_value_counts(metadata_dir: str) -> dict[str, Counter[str]]:
    counts_by_field: dict[str, Counter[str]] = {
        field_key: Counter() for _, field_key in SELECTED_FIELDS
    }

    for path in sorted(Path(metadata_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        for _, field_key in SELECTED_FIELDS:
            for value in _split_values(metadata_dict.get(field_key)):
                counts_by_field[field_key][value] += 1

    return counts_by_field


def write_csv(counts_by_field: dict[str, Counter[str]], output_path: str) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Kategori", "Värde", "Antal dokument"],
        )
        writer.writeheader()

        for field_label, field_key in SELECTED_FIELDS:
            counts = counts_by_field[field_key]
            for value in sorted(counts):
                writer.writerow(
                    {
                        "Kategori": field_label,
                        "Värde": value,
                        "Antal dokument": counts[value],
                    }
                )
                row_count += 1

    return row_count


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    counts_by_field = collect_value_counts(args.metadata_dir)
    row_count = write_csv(counts_by_field, args.output_path)
    print(f"Wrote {row_count} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
