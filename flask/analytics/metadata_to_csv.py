"""Export metadata-only Docplus JSON files to one CSV row per document."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Optional


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export metadata JSON files to a CSV table.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="output/metadata",
        help="Directory containing metadata-only JSON files.",
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
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sorted_json_paths(metadata_dir: str) -> list[Path]:
    return sorted(Path(metadata_dir).glob("*.json"))


def _build_row(path: Path, payload: dict[str, object], metadata_keys: list[str]) -> dict[str, str]:
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    row = {
        "metadata_file": path.name,
        "document_id": path.stem,
    }

    for key, value in payload.items():
        if key == "metadata":
            continue
        row[f"root_{key}"] = _normalize_cell(value)

    for key in metadata_keys:
        row[key] = _normalize_cell(metadata_dict.get(key))

    return row


def export_metadata_csv(metadata_dir: str, output_path: str) -> int:
    json_paths = _sorted_json_paths(metadata_dir)
    metadata_key_set: set[str] = set()
    payloads_by_path: dict[Path, dict[str, object]] = {}
    root_keys: set[str] = set()

    for path in json_paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            continue

        payloads_by_path[path] = payload
        root_keys.update(key for key in payload if key != "metadata")

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata_key_set.update(str(key) for key in metadata.keys())

    metadata_keys = sorted(metadata_key_set)
    fieldnames = [
        "metadata_file",
        "document_id",
        *[f"root_{key}" for key in sorted(root_keys)],
        *metadata_keys,
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in json_paths:
            payload = payloads_by_path.get(path)
            if payload is None:
                continue
            writer.writerow(_build_row(path, payload, metadata_keys))

    return len(payloads_by_path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    row_count = export_metadata_csv(args.metadata_dir, args.output_path)
    print(f"Wrote {row_count} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
