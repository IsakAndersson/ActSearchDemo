"""Export relevant chunks from demo form submissions to CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUBMISSIONS_JSON_DIR = (
    Path(__file__).resolve().parents[1]
    / "flask"
    / "output"
    / "form_submissions"
    / "submissions_json_format"
)
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().with_name(
    "relevant_chunks_from_form_submissions.csv"
)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _metadata_dict(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _metadata_value(result: dict[str, Any], keys: tuple[str, ...]) -> str:
    metadata = _metadata_dict(result)
    for key in keys:
        value = _to_text(metadata.get(key))
        if value:
            return value
    return ""


def _result_title(result: dict[str, Any]) -> str:
    direct_title = _to_text(result.get("title"))
    if direct_title:
        return direct_title
    return _metadata_value(
        result,
        ("title", "document_title", "doc_title", "name", "filename", "file_name"),
    )


def _result_url(result: dict[str, Any]) -> str:
    for key in ("source_url", "url", "link", "href", "source"):
        direct_value = _to_text(result.get(key))
        if direct_value:
            return direct_value
    return _metadata_value(result, ("source_url", "url", "link", "href", "source"))


def _pooled_from_methods(result: dict[str, Any]) -> str:
    metadata = _metadata_dict(result)
    pooled_from = metadata.get("pooled_from")
    if isinstance(pooled_from, list):
        methods = [_to_text(value) for value in pooled_from]
        methods = [value for value in methods if value]
        if methods:
            return "|".join(methods)

    direct_method = _to_text(result.get("result_method")) or _to_text(result.get("demo_method"))
    if direct_method:
        return direct_method

    return _to_text(metadata.get("demo_method"))


def _section_heading(result: dict[str, Any]) -> str:
    direct_heading = _to_text(result.get("section_heading"))
    if direct_heading:
        return direct_heading
    return _metadata_value(result, ("section_heading",))


def _chunk_text(result: dict[str, Any]) -> str:
    for key in ("section_text", "chunk_text", "preview_text", "text"):
        value = _to_text(result.get(key))
        if value:
            return value

    return _metadata_value(result, ("section_text",))


def _is_relevant(result: dict[str, Any]) -> bool:
    assessment = result.get("assessment")
    if not isinstance(assessment, dict):
        return False
    return _to_text(assessment.get("rating")) == "relevant"


def _iter_submission_rows_from_json_dir(
    json_dir: Path,
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    for path in sorted(json_dir.glob("submission_*.json")):
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if not isinstance(raw_payload, dict):
            continue

        submission_id = raw_payload.get("submission_id")
        if isinstance(submission_id, int):
            resolved_submission_id = submission_id
        else:
            resolved_submission_id = -1

        created_at = _to_text(raw_payload.get("created_at"))
        payload = raw_payload.get("payload")
        if isinstance(payload, dict):
            yield resolved_submission_id, created_at, payload


def submission_to_relevant_chunk_rows(
    *,
    submission_id: int,
    created_at: str,
    submission_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    participant_name = _to_text(submission_payload.get("participant_name"))
    information_need = _to_text(submission_payload.get("information_need"))
    query = _to_text(submission_payload.get("query"))
    general_comment = _to_text(submission_payload.get("general_comment"))

    for result_index, result in enumerate(submission_payload.get("results", []), start=1):
        if not isinstance(result, dict) or not _is_relevant(result):
            continue

        assessment = result.get("assessment")
        assessment_dict = assessment if isinstance(assessment, dict) else {}
        rows.append(
            {
                "submission_id": submission_id,
                "created_at": created_at,
                "participant_name": participant_name,
                "information_need": information_need,
                "query": query,
                "general_comment": general_comment,
                "result_index": result_index,
                "chunk_id": _to_text(result.get("chunk_id")),
                "chunk_type": _to_text(result.get("chunk_type")),
                "source_path": _to_text(result.get("source_path")),
                "document_title": _result_title(result),
                "document_url": _result_url(result),
                "pooled_from_methods": _pooled_from_methods(result),
                "section_heading": _section_heading(result),
                "chunk_text": _chunk_text(result),
                "relevant_scope": _to_text(assessment_dict.get("relevant_scope")),
                "relevant_section": _to_text(assessment_dict.get("relevant_section")),
                "assessment_comment": _to_text(assessment_dict.get("comment")),
            }
        )
    return rows


def export_relevant_chunks_from_json_dir(
    json_dir: Path,
    output_csv: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for submission_id, created_at, payload in _iter_submission_rows_from_json_dir(json_dir):
        rows.extend(
            submission_to_relevant_chunk_rows(
                submission_id=submission_id,
                created_at=created_at,
                submission_payload=payload,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "submission_id",
        "created_at",
        "participant_name",
        "information_need",
        "query",
        "general_comment",
        "result_index",
        "chunk_id",
        "chunk_type",
        "source_path",
        "document_title",
        "document_url",
        "pooled_from_methods",
        "section_heading",
        "chunk_text",
        "relevant_scope",
        "relevant_section",
        "assessment_comment",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export all relevant chunks from form submissions to CSV."
    )
    parser.add_argument(
        "--json-dir",
        default=str(DEFAULT_SUBMISSIONS_JSON_DIR),
        help="Path to submissions_json_format directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    rows = export_relevant_chunks_from_json_dir(
        json_dir=Path(args.json_dir),
        output_csv=Path(args.output),
    )
    print(
        f"Exported {len(rows)} relevant chunks to {Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
