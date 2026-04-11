"""Print unique Docplus metadata values grouped by field."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

DOCPLUS_METADATA_FIELDS = (
    "document_collection",
    "process",
    "publish_date",
    "subject_area",
    "type_of_action",
    "valid_for_area",
    "version",
    "comment",
    "document_type",
    "metadata_url",
    "tax_keyword",
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print unique non-empty Docplus metadata values per field.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="output/metadata",
        help="Directory containing metadata-only JSON files.",
    )
    return parser.parse_args(argv)


def _normalize_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def collect_unique_values(metadata_dir: str) -> dict[str, Counter[str]]:
    values_by_field: dict[str, Counter[str]] = {
        field_name: Counter() for field_name in DOCPLUS_METADATA_FIELDS
    }

    for path in sorted(Path(metadata_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        for field_name in DOCPLUS_METADATA_FIELDS:
            value = _normalize_value(metadata_dict.get(field_name))
            if value is not None:
                values_by_field[field_name][value] += 1

    return values_by_field


def print_unique_values(values_by_field: dict[str, Counter[str]]) -> None:
    print("Unique Docplus metadata values")
    for field_name, counts in values_by_field.items():
        print(f"{field_name} ({len(counts)} unique values)")
        if not counts:
            print("  (none)")
            continue
        for value in sorted(counts):
            print(f"  - {value}: {counts[value]}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    values_by_field = collect_unique_values(args.metadata_dir)
    print_unique_values(values_by_field)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
