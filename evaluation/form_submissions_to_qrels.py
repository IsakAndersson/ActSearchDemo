"""Export demo form submissions to qrels CSV."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from .doc_id import normalize_doc_id
except ImportError:
    from doc_id import normalize_doc_id


DEFAULT_SUBMISSIONS_DB = (
    Path(__file__).resolve().parents[1]
    / "flask"
    / "output"
    / "form_submissions"
    / "form_submissions.sqlite3"
)
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().with_name("qrels_from_form_submissions.csv")
DEFAULT_AUDIT_CSV = Path(__file__).resolve().with_name("form_submissions_audit.csv")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_doc_id(result: dict[str, Any]) -> str:
    for key in ("document", "title"):
        text = _to_text(result.get(key))
        normalized = normalize_doc_id(text)
        if normalized:
            return normalized

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "title",
            "document_title",
            "doc_title",
            "name",
            "filename",
            "file_name",
            "source_path",
            "source_url",
            "url",
        ):
            text = _to_text(metadata.get(key))
            normalized = normalize_doc_id(text)
            if normalized:
                return normalized

    for key in ("source_path", "source_url", "url"):
        text = _to_text(result.get(key))
        normalized = normalize_doc_id(text)
        if normalized:
            return normalized

    return ""


def _relevance_from_assessment(assessment: dict[str, Any] | None) -> int:
    if not isinstance(assessment, dict):
        return 0
    return 1 if _to_text(assessment.get("rating")) == "relevant" else 0


def submission_to_qrels_rows(
    submission_payload: dict[str, Any],
    include_non_relevant: bool = False,
    metadata: dict[str, Any] | None = None,
    include_metadata: bool = False,
    query_type: str | None = None,
) -> list[dict[str, Any]]:
    query_id = _to_text(submission_payload.get("query"))
    if not query_id:
        return []

    rows: list[dict[str, Any]] = []
    for result in submission_payload.get("results", []):
        if not isinstance(result, dict):
            continue
        doc_id = _extract_doc_id(result)
        if not doc_id:
            continue
        relevance = _relevance_from_assessment(result.get("assessment"))
        if relevance == 0 and not include_non_relevant:
            continue
        row = {
            "query_id": query_id,
            "doc_id": doc_id,
            "relevance": relevance,
        }
        if query_type:
            row["query_type"] = query_type
        if include_metadata and metadata:
            row.update(metadata)
        rows.append(row)
    return rows


def _iter_submission_records_from_db(db_path: Path) -> Iterable[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            SELECT
                id,
                created_at,
                participant_name,
                information_need,
                query_text,
                general_comment,
                payload_json
            FROM submissions
            ORDER BY id ASC
            """
        )
        for (
            submission_id,
            created_at,
            participant_name,
            information_need,
            query_text,
            general_comment,
            payload_json,
        ) in cursor.fetchall():
            if not isinstance(payload_json, str):
                continue
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield {
                    "submission_id": submission_id,
                    "created_at": created_at,
                    "participant_name": participant_name,
                    "information_need": information_need,
                    "query_text": query_text,
                    "general_comment": general_comment,
                    "payload": payload,
                }


def _iter_submission_payloads_from_db(db_path: Path) -> Iterable[dict[str, Any]]:
    for record in _iter_submission_records_from_db(db_path):
        payload = record.get("payload")
        if isinstance(payload, dict):
            yield payload


def _parse_submission_ids(raw_ids: str | None) -> set[int]:
    if not raw_ids:
        return set()
    ids: set[int] = set()
    for item in raw_ids.split(","):
        item = item.strip()
        if not item:
            continue
        ids.add(int(item))
    return ids


def build_submission_audit_df(db_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in _iter_submission_records_from_db(db_path):
        payload = record["payload"]
        relevant_count = 0
        non_relevant_count = 0
        unrated_count = 0
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            assessment = result.get("assessment")
            rating = assessment.get("rating") if isinstance(assessment, dict) else None
            if rating == "relevant":
                relevant_count += 1
            elif rating == "not_relevant":
                non_relevant_count += 1
            else:
                unrated_count += 1

        rows.append(
            {
                "submission_id": record["submission_id"],
                "created_at": record["created_at"],
                "participant_name": record["participant_name"],
                "query_id": _to_text(payload.get("query")) or record["query_text"],
                "information_need": record["information_need"],
                "general_comment": record["general_comment"],
                "num_results": len(payload.get("results", [])),
                "num_relevant": relevant_count,
                "num_non_relevant": non_relevant_count,
                "num_unrated": unrated_count,
                "looks_like_test": (
                    _to_text(payload.get("query")).lower() == "test"
                    or _to_text(record["information_need"]).lower() == "test"
                ),
            }
        )
    return pd.DataFrame(rows)


def export_qrels_from_db(
    db_path: Path,
    output_csv: Path,
    include_non_relevant: bool = False,
    exclude_submission_ids: set[int] | None = None,
    include_metadata: bool = False,
    query_type: str | None = None,
) -> pd.DataFrame:
    exclude_submission_ids = exclude_submission_ids or set()
    rows: list[dict[str, Any]] = []
    for record in _iter_submission_records_from_db(db_path):
        if int(record["submission_id"]) in exclude_submission_ids:
            continue
        metadata = {
            "submission_id": record["submission_id"],
            "created_at": record["created_at"],
            "participant_name": record["participant_name"],
            "information_need": record["information_need"],
            "general_comment": record["general_comment"],
        }
        rows.extend(
            submission_to_qrels_rows(
                record["payload"],
                include_non_relevant=include_non_relevant,
                metadata=metadata,
                include_metadata=include_metadata,
                query_type=query_type,
            )
        )

    columns = ["query_id", "doc_id", "relevance"]
    if query_type:
        columns.append("query_type")
    if include_metadata:
        columns.extend(
            [
                "submission_id",
                "created_at",
                "participant_name",
                "information_need",
                "general_comment",
            ]
        )
    qrels_df = pd.DataFrame(rows, columns=columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    qrels_df.to_csv(output_csv, index=False)
    return qrels_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export demo form submissions from SQLite to qrels CSV."
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_SUBMISSIONS_DB),
        help="Path to form_submissions.sqlite3.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--include-non-relevant",
        action="store_true",
        help="Include rows with relevance=0 for explicitly non-relevant documents.",
    )
    parser.add_argument(
        "--exclude-submission-ids",
        help="Comma-separated submission ids to exclude, for example '6,7,15'.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include submission provenance columns in the qrels CSV.",
    )
    parser.add_argument(
        "--query-type",
        help="Optional query_type value written to every qrels row.",
    )
    parser.add_argument(
        "--audit-output",
        default=str(DEFAULT_AUDIT_CSV),
        help="Path to write a submission-level audit CSV before exporting qrels.",
    )
    args = parser.parse_args()

    audit_df = build_submission_audit_df(Path(args.db_path))
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(audit_path, index=False)

    qrels_df = export_qrels_from_db(
        db_path=Path(args.db_path),
        output_csv=Path(args.output),
        include_non_relevant=args.include_non_relevant,
        exclude_submission_ids=_parse_submission_ids(args.exclude_submission_ids),
        include_metadata=args.include_metadata,
        query_type=args.query_type,
    )
    print(
        f"Wrote audit for {len(audit_df)} submissions to {audit_path.resolve()}",
        flush=True,
    )
    print(
        f"Exported {len(qrels_df)} qrels rows to {Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
