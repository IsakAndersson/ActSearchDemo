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


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_doc_id(result: dict[str, Any]) -> str:
    for key in ("document", "title", "source_path", "source_url", "url"):
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

    return ""


def _relevance_from_assessment(assessment: dict[str, Any] | None) -> int:
    if not isinstance(assessment, dict):
        return 0
    return 1 if _to_text(assessment.get("rating")) == "relevant" else 0


def submission_to_qrels_rows(
    submission_payload: dict[str, Any],
    include_non_relevant: bool = False,
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
        rows.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": relevance,
            }
        )
    return rows


def _iter_submission_payloads_from_db(db_path: Path) -> Iterable[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            "SELECT payload_json FROM submissions ORDER BY id ASC"
        )
        for (payload_json,) in cursor.fetchall():
            if not isinstance(payload_json, str):
                continue
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def export_qrels_from_db(
    db_path: Path,
    output_csv: Path,
    include_non_relevant: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in _iter_submission_payloads_from_db(db_path):
        rows.extend(
            submission_to_qrels_rows(
                payload,
                include_non_relevant=include_non_relevant,
            )
        )

    qrels_df = pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"])
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
    args = parser.parse_args()

    qrels_df = export_qrels_from_db(
        db_path=Path(args.db_path),
        output_csv=Path(args.output),
        include_non_relevant=args.include_non_relevant,
    )
    print(
        f"Exported {len(qrels_df)} qrels rows to {Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
