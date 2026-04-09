"""Print summary statistics for parsed Docplus JSON files."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print document, page-count, and content-type statistics for parsed JSON files.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files.",
    )
    return parser.parse_args(argv)


def _normalize_content_type(value: object) -> str:
    if not isinstance(value, str):
        return "(missing)"
    trimmed = value.strip()
    return trimmed if trimmed else "(missing)"


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def collect_stats(parsed_dir: str) -> dict:
    parsed_path = Path(parsed_dir)
    document_count = 0
    page_counts: list[int] = []
    content_types: Counter[str] = Counter()

    for path in sorted(parsed_path.glob("*.json")):
        document_count += 1
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        page_count = metadata_dict.get("page_count")
        if isinstance(page_count, int):
            page_counts.append(page_count)

        content_types[_normalize_content_type(metadata_dict.get("content_type"))] += 1

    average_pages = statistics.mean(page_counts) if page_counts else None
    median_pages = statistics.median(page_counts) if page_counts else None
    max_pages = max(page_counts) if page_counts else None
    percentiles = {
        "p10": _percentile(page_counts, 0.10),
        "p25": _percentile(page_counts, 0.25),
        "p75": _percentile(page_counts, 0.75),
        "p90": _percentile(page_counts, 0.90),
        "p95": _percentile(page_counts, 0.95),
    }

    return {
        "document_count": document_count,
        "page_count_documents": len(page_counts),
        "average_pages": average_pages,
        "median_pages": median_pages,
        "max_pages": max_pages,
        "percentiles": percentiles,
        "content_types": content_types,
    }


def print_stats(stats: dict) -> None:
    print("Parsed document statistics")
    print(f"Documents: {stats['document_count']}")

    page_count_documents = stats["page_count_documents"]
    if page_count_documents == 0:
        print("Page counts: no documents with metadata.page_count")
    else:
        average_pages = stats["average_pages"]
        median_pages = stats["median_pages"]
        max_pages = stats["max_pages"]
        percentiles = stats["percentiles"]
        print(f"Documents with page_count: {page_count_documents}")
        print(f"Average pages: {average_pages:.2f}")
        print(f"Median pages: {median_pages}")
        print(f"Max pages: {max_pages}")
        print("Page-count percentiles:")
        for label in ("p10", "p25", "p75", "p90", "p95"):
            value = percentiles.get(label)
            if value is not None:
                print(f"  {label}: {value:.2f}")

    print("Content types:")
    for content_type, count in stats["content_types"].most_common():
        print(f"  {content_type}: {count}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    stats = collect_stats(args.parsed_dir)
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
