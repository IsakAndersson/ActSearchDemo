"""Combine parsed Docplus JSON files into a single JSON file."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def combine_parsed_json(input_dir: str, output_file: str) -> int:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    entries: List[Dict[str, Any]] = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(input_dir, filename)
        try:
            payload = _load_json(path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        payload.setdefault("_source_file", filename)
        entries.append(payload)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine output/parsed JSON files into a single JSON array.",
    )
    parser.add_argument(
        "--input-dir",
        default="/home/isak/Repos/ActSearchDemo/flask/output/parsed",
        help="Directory containing parsed JSON files (default: output/parsed).",
    )
    parser.add_argument(
        "--output-file",
        default="output/combined.json",
        help="Path to combined JSON file (default: output/combined.json).",
    )
    args = parser.parse_args()

    count = combine_parsed_json(args.input_dir, args.output_file)
    print(f"Wrote {count} entries to {args.output_file}")


if __name__ == "__main__":
    main()
