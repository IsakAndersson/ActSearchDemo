"""Print summary statistics for parsed Docplus JSON files."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from document_structure import _extract_toc_entries, _parse_text_lines

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
APPROVED_DATE_PREFIX = "Godkänt den:"
APPROVED_DATE_REGEX = re.compile(r"Godkänt den:\s*(?:\n\s*)*(\d{4}-\d{2}-\d{2})")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print document, page-count, and content-type statistics for parsed JSON files.",
    )
    parser.add_argument(
        "--parsed-dir",
        default="output/parsed",
        help="Directory containing parsed JSON files.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="output/metadata",
        help="Directory containing metadata-only JSON files for Docplus metadata coverage.",
    )
    parser.add_argument(
        "--metadata-values-output-dir",
        default="output/metadata_field_values",
        help="Directory where one .txt file per Docplus metadata field will be written.",
    )
    parser.add_argument(
        "--approved-dates-output-path",
        default="output/approved_dates.txt",
        help="Path where approved-date counts from parsed text will be written.",
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


def _has_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip() != ""
    return value is not None


def _split_metadata_values(value: object) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    if value is None:
        return []
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return [serialized]


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


def _extract_approved_date_from_text(text: object, head_chars: int = 500) -> str | None:
    if not isinstance(text, str):
        return None
    text_head = text[:head_chars]
    match = APPROVED_DATE_REGEX.search(text_head)
    if not match:
        return None
    return match.group(1)


def _has_table_of_contents(text: object) -> bool:
    if not isinstance(text, str):
        return False
    toc_entries, _ = _extract_toc_entries(_parse_text_lines(text))
    return bool(toc_entries)


def collect_stats(parsed_dir: str, metadata_dir: str) -> dict:
    parsed_path = Path(parsed_dir)
    metadata_path = Path(metadata_dir)
    today = date.today()
    document_count = 0
    page_counts: list[int] = []
    content_types: Counter[str] = Counter()
    approved_date_counts: Counter[str] = Counter()
    approved_dates_by_document: dict[str, date] = {}
    approved_prefix_missing = 0
    approved_date_missing_after_prefix = 0
    approved_dates: list[date] = []
    approved_age_days: list[int] = []
    approved_dates_older_than_two_years = 0
    documents_with_toc = 0
    documents_without_toc = 0
    documents_without_toc_over_3_pages = 0
    metadata_coverage: dict[str, dict[str, object]] = {
        field_name: {
            "documents_with_value": 0,
            "unique_values": set(),
        }
        for field_name in DOCPLUS_METADATA_FIELDS
    }
    metadata_value_counts: dict[str, Counter[str]] = {
        field_name: Counter() for field_name in DOCPLUS_METADATA_FIELDS
    }

    for path in sorted(parsed_path.glob("*.json")):
        document_count += 1
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        text = payload.get("text")
        has_toc = _has_table_of_contents(text)

        page_count = metadata_dict.get("page_count")
        if isinstance(page_count, int):
            page_counts.append(page_count)
            if not has_toc and page_count > 3:
                documents_without_toc_over_3_pages += 1

        if has_toc:
            documents_with_toc += 1
        else:
            documents_without_toc += 1

        content_types[_normalize_content_type(metadata_dict.get("content_type"))] += 1

        text_head = text[:500] if isinstance(text, str) else ""
        if APPROVED_DATE_PREFIX not in text_head:
            approved_prefix_missing += 1
        else:
            approved_date = _extract_approved_date_from_text(text)
            if approved_date is None:
                approved_date_missing_after_prefix += 1
            else:
                approved_date_counts[approved_date] += 1
                approved_date_obj = datetime.strptime(approved_date, "%Y-%m-%d").date()
                approved_dates_by_document[path.stem] = approved_date_obj
                approved_dates.append(approved_date_obj)
                approved_age = (today - approved_date_obj).days
                approved_age_days.append(approved_age)
                if approved_age > 365 * 2:
                    approved_dates_older_than_two_years += 1
    metadata_document_count = 0
    publish_date_documents = 0
    older_than_two_years_documents = 0
    approved_dates_differing_from_publish_date = 0
    publish_dates: list[date] = []
    document_ages_days: list[int] = []
    for path in sorted(metadata_path.glob("*.json")):
        metadata_document_count += 1
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        publish_date = _parse_publish_date(metadata_dict.get("publish_date"))
        if publish_date is not None:
            publish_date_documents += 1
            publish_dates.append(publish_date)
            age_in_days = (today - publish_date).days
            document_ages_days.append(age_in_days)
            if age_in_days > 365 * 2:
                older_than_two_years_documents += 1
            approved_date = approved_dates_by_document.get(path.stem)
            if approved_date is not None and approved_date != publish_date:
                approved_dates_differing_from_publish_date += 1

        for field_name in DOCPLUS_METADATA_FIELDS:
            field_value = metadata_dict.get(field_name)
            field_stats = metadata_coverage[field_name]
            if _has_value(field_value):
                field_stats["documents_with_value"] = int(field_stats["documents_with_value"]) + 1
                split_values = _split_metadata_values(field_value)
                for normalized_value in split_values:
                    field_stats["unique_values"].add(normalized_value)
                    metadata_value_counts[field_name][normalized_value] += 1

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
    age_percentiles_days = {
        "p10": _percentile(document_ages_days, 0.10),
        "p25": _percentile(document_ages_days, 0.25),
        "p75": _percentile(document_ages_days, 0.75),
        "p90": _percentile(document_ages_days, 0.90),
        "p95": _percentile(document_ages_days, 0.95),
    }
    average_age_days = statistics.mean(document_ages_days) if document_ages_days else None
    median_age_days = statistics.median(document_ages_days) if document_ages_days else None
    newest_publish_date = max(publish_dates) if publish_dates else None
    oldest_publish_date = min(publish_dates) if publish_dates else None
    average_approved_age_days = statistics.mean(approved_age_days) if approved_age_days else None
    median_approved_age_days = statistics.median(approved_age_days) if approved_age_days else None
    newest_approved_date = max(approved_dates) if approved_dates else None
    oldest_approved_date = min(approved_dates) if approved_dates else None

    return {
        "document_count": document_count,
        "metadata_document_count": metadata_document_count,
        "publish_date_documents": publish_date_documents,
        "older_than_two_years_documents": older_than_two_years_documents,
        "older_than_two_years_percent": (
            (older_than_two_years_documents / publish_date_documents) * 100
            if publish_date_documents > 0
            else 0.0
        ),
        "average_age_days": average_age_days,
        "median_age_days": median_age_days,
        "newest_publish_date": newest_publish_date.isoformat() if newest_publish_date else None,
        "oldest_publish_date": oldest_publish_date.isoformat() if oldest_publish_date else None,
        "age_percentiles_days": age_percentiles_days,
        "page_count_documents": len(page_counts),
        "documents_over_50_pages": sum(1 for value in page_counts if value > 50),
        "documents_over_100_pages": sum(1 for value in page_counts if value > 100),
        "approved_date_counts": approved_date_counts,
        "approved_date_documents": sum(approved_date_counts.values()),
        "approved_date_missing_total": approved_prefix_missing + approved_date_missing_after_prefix,
        "approved_prefix_missing": approved_prefix_missing,
        "approved_date_missing_after_prefix": approved_date_missing_after_prefix,
        "approved_dates_older_than_two_years": approved_dates_older_than_two_years,
        "approved_dates_older_than_two_years_percent": (
            (approved_dates_older_than_two_years / sum(approved_date_counts.values())) * 100
            if sum(approved_date_counts.values()) > 0
            else 0.0
        ),
        "approved_dates_differing_from_publish_date": approved_dates_differing_from_publish_date,
        "approved_dates_differing_from_publish_date_percent": (
            (approved_dates_differing_from_publish_date / sum(approved_date_counts.values())) * 100
            if sum(approved_date_counts.values()) > 0
            else 0.0
        ),
        "average_approved_age_days": average_approved_age_days,
        "median_approved_age_days": median_approved_age_days,
        "newest_approved_date": newest_approved_date.isoformat() if newest_approved_date else None,
        "oldest_approved_date": oldest_approved_date.isoformat() if oldest_approved_date else None,
        "documents_with_toc": documents_with_toc,
        "documents_with_toc_percent": (
            (documents_with_toc / document_count) * 100 if document_count > 0 else 0.0
        ),
        "documents_without_toc": documents_without_toc,
        "documents_without_toc_percent": (
            (documents_without_toc / document_count) * 100 if document_count > 0 else 0.0
        ),
        "documents_without_toc_over_3_pages": documents_without_toc_over_3_pages,
        "documents_without_toc_over_3_pages_percent": (
            (documents_without_toc_over_3_pages / documents_without_toc) * 100
            if documents_without_toc > 0
            else 0.0
        ),
        "average_pages": average_pages,
        "median_pages": median_pages,
        "max_pages": max_pages,
        "percentiles": percentiles,
        "content_types": content_types,
        "metadata_coverage": {
            field_name: {
                "documents_with_value": int(field_stats["documents_with_value"]),
                "coverage_percent": (
                    (int(field_stats["documents_with_value"]) / metadata_document_count) * 100
                    if metadata_document_count > 0
                    else 0.0
                ),
                "unique_value_count": len(field_stats["unique_values"]),
            }
            for field_name, field_stats in metadata_coverage.items()
        },
        "metadata_value_counts": metadata_value_counts,
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
        documents_over_50_pages = stats["documents_over_50_pages"]
        documents_over_100_pages = stats["documents_over_100_pages"]
        print(f"Documents with page_count: {page_count_documents}")
        print(f"Average pages: {average_pages:.2f}")
        print(f"Median pages: {median_pages}")
        print(f"Max pages: {max_pages}")
        print(
            "Documents over 50 pages: "
            f"{documents_over_50_pages} "
            f"({(documents_over_50_pages / page_count_documents) * 100:.2f}%)"
        )
        print(
            "Documents over 100 pages: "
            f"{documents_over_100_pages} "
            f"({(documents_over_100_pages / page_count_documents) * 100:.2f}%)"
        )
        print("Page-count percentiles:")
        for label in ("p10", "p25", "p75", "p90", "p95"):
            value = percentiles.get(label)
            if value is not None:
                print(f"  {label}: {value:.2f}")

    print("Content types:")
    for content_type, count in stats["content_types"].most_common():
        print(f"  {content_type}: {count}")

    print("Table-of-contents summary:")
    print(
        "Documents with table of contents: "
        f"{stats['documents_with_toc']} "
        f"({stats['documents_with_toc_percent']:.2f}%)"
    )
    print(
        "Documents without table of contents: "
        f"{stats['documents_without_toc']} "
        f"({stats['documents_without_toc_percent']:.2f}%)"
    )
    print(
        "Documents without table of contents over 3 pages: "
        f"{stats['documents_without_toc_over_3_pages']} "
        f"({stats['documents_without_toc_over_3_pages_percent']:.2f}%)"
    )

    print("Docplus metadata field coverage:")
    print(f"Metadata documents: {stats['metadata_document_count']}")
    for field_name, field_stats in stats["metadata_coverage"].items():
        print(
            "  "
            f"{field_name}: "
            f"{field_stats['documents_with_value']} documents "
            f"({field_stats['coverage_percent']:.2f}%), "
            f"{field_stats['unique_value_count']} unique values"
        )

    print("Publish-date age summary:")
    print(f"Documents with publish_date: {stats['publish_date_documents']}")
    print(
        "Older than 2 years: "
        f"{stats['older_than_two_years_documents']} documents "
        f"({stats['older_than_two_years_percent']:.2f}%)"
    )
    average_age_days = stats["average_age_days"]
    median_age_days = stats["median_age_days"]
    if average_age_days is not None:
        print(f"Average document age (years): {average_age_days / 365.25:.2f}")
    if median_age_days is not None:
        print(f"Median document age (years): {median_age_days / 365.25:.2f}")
    print(f"Newest publish_date: {stats['newest_publish_date'] or '(missing)'}")
    print(f"Oldest publish_date: {stats['oldest_publish_date'] or '(missing)'}")
    print("Document-age percentiles (years):")
    for label in ("p10", "p25", "p75", "p90", "p95"):
        value = stats["age_percentiles_days"].get(label)
        if value is not None:
            print(f"  {label}: {value / 365.25:.2f}")

    print("Approved-date summary:")
    total_documents = stats["document_count"]
    approved_date_documents = stats["approved_date_documents"]
    approved_date_missing_total = stats["approved_date_missing_total"]
    approved_prefix_missing = stats["approved_prefix_missing"]
    approved_date_missing_after_prefix = stats["approved_date_missing_after_prefix"]
    print(
        "Documents with approved date: "
        f"{approved_date_documents} "
        f"({(approved_date_documents / total_documents) * 100:.2f}%)"
    )
    print(
        "Documents without approved date: "
        f"{approved_date_missing_total} "
        f"({(approved_date_missing_total / total_documents) * 100:.2f}%)"
    )
    print(
        "Documents missing 'Godkänt den:' in text head: "
        f"{approved_prefix_missing} "
        f"({(approved_prefix_missing / total_documents) * 100:.2f}%)"
    )
    print(
        "Documents with 'Godkänt den:' but no extracted date: "
        f"{approved_date_missing_after_prefix} "
        f"({(approved_date_missing_after_prefix / total_documents) * 100:.2f}%)"
    )
    print(
        "Documents older than 2 years by approved date: "
        f"{stats['approved_dates_older_than_two_years']} "
        f"({stats['approved_dates_older_than_two_years_percent']:.2f}%)"
    )
    print(
        "Documents with approved date differing from publish_date: "
        f"{stats['approved_dates_differing_from_publish_date']} "
        f"({stats['approved_dates_differing_from_publish_date_percent']:.2f}%)"
    )
    average_approved_age_days = stats["average_approved_age_days"]
    median_approved_age_days = stats["median_approved_age_days"]
    if average_approved_age_days is not None:
        print(f"Average approved-date age (years): {average_approved_age_days / 365.25:.2f}")
    if median_approved_age_days is not None:
        print(f"Median approved-date age (years): {median_approved_age_days / 365.25:.2f}")
    print(f"Newest approved date: {stats['newest_approved_date'] or '(missing)'}")
    print(f"Oldest approved date: {stats['oldest_approved_date'] or '(missing)'}")


def write_metadata_value_files(stats: dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_value_counts = stats["metadata_value_counts"]
    for field_name, counts in metadata_value_counts.items():
        file_path = output_path / f"{field_name}.txt"
        with file_path.open("w", encoding="utf-8") as handle:
            handle.write(f"{field_name}\n")
            handle.write(f"unique_values={len(counts)}\n")
            if not counts:
                handle.write("(none)\n")
                continue
            for value in sorted(counts):
                handle.write(f"{value}: {counts[value]}\n")


def write_approved_date_counts(stats: dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    approved_date_counts = stats["approved_date_counts"]
    approved_date_documents = stats["approved_date_documents"]
    with output.open("w", encoding="utf-8") as handle:
        handle.write("approved_dates\n")
        handle.write(f"documents_with_approved_date: {stats['approved_date_documents']}\n")
        if stats["average_approved_age_days"] is not None:
            handle.write(f"average_approved_date_age_years: {stats['average_approved_age_days'] / 365.25:.2f}\n")
        if stats["median_approved_age_days"] is not None:
            handle.write(f"median_approved_date_age_years: {stats['median_approved_age_days'] / 365.25:.2f}\n")
        handle.write(
            "older_than_two_years_by_approved_date: "
            f"{stats['approved_dates_older_than_two_years']} "
            f"({stats['approved_dates_older_than_two_years_percent']:.2f}%)\n"
        )
        handle.write(
            "differing_from_publish_date: "
            f"{stats['approved_dates_differing_from_publish_date']} "
            f"({stats['approved_dates_differing_from_publish_date_percent']:.2f}%)\n"
        )
        for approved_date in sorted(approved_date_counts):
            count = approved_date_counts[approved_date]
            percent = (count / approved_date_documents) * 100 if approved_date_documents > 0 else 0.0
            handle.write(f"{approved_date}: {count} ({percent:.2f}%)\n")
        handle.write(f"missing_prefix_in_text_head: {stats['approved_prefix_missing']}\n")
        handle.write(
            "missing_date_after_prefix_in_text_head: "
            f"{stats['approved_date_missing_after_prefix']}\n"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    stats = collect_stats(args.parsed_dir, args.metadata_dir)
    print_stats(stats)
    write_metadata_value_files(stats, args.metadata_values_output_dir)
    write_approved_date_counts(stats, args.approved_dates_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
