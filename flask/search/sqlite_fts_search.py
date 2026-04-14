"""SQLite FTS5 search over parsed Docplus content."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import threading
from typing import Dict, Iterable, List, Tuple

from document_structure import get_document_sections

from search.bm25_search import chunk_text, extract_title, iter_parsed_documents, tokenize


DEFAULT_DB_PATH = os.path.join("output", "sqlite_fts", "docplus_fts.sqlite3")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
SCHEMA_VERSION = 1
LOG = logging.getLogger(__name__)
_BUILD_LOCK = threading.Lock()


def _parsed_dir_signature(parsed_dir: str) -> Tuple[int, int, int]:
    file_count = 0
    latest_mtime_ns = 0
    total_size = 0

    with os.scandir(parsed_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            file_count += 1
            stat = entry.stat()
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
            total_size += stat.st_size

    return file_count, latest_mtime_ns, total_size


def _index_signature(
    parsed_dir: str,
    max_chars: int,
    overlap: int,
    include_title_chunk: bool,
    use_chunking: bool,
) -> str:
    normalized_dir = os.path.abspath(parsed_dir)
    file_count, latest_mtime_ns, total_size = _parsed_dir_signature(normalized_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "parsed_dir": normalized_dir,
        "file_count": file_count,
        "latest_mtime_ns": latest_mtime_ns,
        "total_size": total_size,
        "max_chars": int(max_chars),
        "overlap": int(overlap),
        "include_title_chunk": bool(include_title_chunk),
        "use_chunking": bool(use_chunking),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _metadata_search_text(metadata: Dict[str, object]) -> str:
    values: List[str] = []
    for key in sorted(metadata.keys()):
        if key in {"section_path", "section_text", "source_url", "downloaded_at"}:
            continue
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, (int, float)):
            values.append(str(value))
    return " ".join(values)


def _iter_chunk_rows(
    parsed_dir: str,
    max_chars: int,
    overlap: int,
    include_title_chunk: bool,
    use_chunking: bool,
) -> Iterable[Dict[str, object]]:
    chunk_id = 0
    for payload in iter_parsed_documents(parsed_dir):
        metadata = payload.get("metadata", {}) or {}
        title = extract_title(payload)
        body_metadata = {**metadata, "title": title} if title else metadata
        sections = get_document_sections(payload, fallback_title=title)

        if include_title_chunk and title:
            title_metadata = {**body_metadata}
            yield {
                "chunk_id": chunk_id,
                "source_path": payload["path"],
                "chunk_type": "title",
                "preview_text": title,
                "text": title,
                "metadata": title_metadata,
                "section_heading": title,
                "section_index": 0,
                "section_level": 1,
                "section_path": [title],
                "section_path_text": title,
                "section_text": payload.get("text", ""),
                "title_text": title,
                "body_text": title,
                "metadata_text": _metadata_search_text(title_metadata),
            }
            chunk_id += 1

        for section in sections:
            section_text = section["text"]
            heading = str(section.get("heading") or title or "").strip()
            section_metadata = {
                **body_metadata,
                "section_heading": heading,
                "section_index": section.get("index", 0),
                "section_level": section.get("level", 1),
                "section_page": section.get("page"),
                "section_path": section.get("path") or [heading],
                "section_path_text": section.get("path_text") or heading,
                "section_title": heading,
            }

            if heading:
                title_chunk_text = f"{title}\n\n{heading}" if title and heading != title else heading
                yield {
                    "chunk_id": chunk_id,
                    "source_path": payload["path"],
                    "chunk_type": "section_title",
                    "preview_text": heading,
                    "text": title_chunk_text,
                    "metadata": section_metadata,
                    "section_heading": heading,
                    "section_index": section.get("index", 0),
                    "section_level": section.get("level", 1),
                    "section_path": section.get("path") or [heading],
                    "section_path_text": section.get("path_text") or heading,
                    "section_text": section_text,
                    "title_text": title_chunk_text,
                    "body_text": heading,
                    "metadata_text": _metadata_search_text(section_metadata),
                }
                chunk_id += 1

            if not isinstance(section_text, str) or not section_text.strip():
                continue

            if use_chunking:
                section_chunks = chunk_text(section_text, max_chars=max_chars, overlap=overlap)
            else:
                section_chunks = [section_text]

            for section_chunk_index, preview_text in enumerate(section_chunks):
                chunk_text_value = preview_text
                if heading and use_chunking and section_chunk_index == 0:
                    chunk_text_value = f"{heading}\n\n{preview_text}"
                elif heading and not use_chunking:
                    chunk_text_value = f"{heading}\n\n{section_text}"

                yield {
                    "chunk_id": chunk_id,
                    "source_path": payload["path"],
                    "chunk_type": "section",
                    "preview_text": preview_text,
                    "text": chunk_text_value,
                    "metadata": section_metadata,
                    "section_heading": heading,
                    "section_index": section.get("index", 0),
                    "section_level": section.get("level", 1),
                    "section_path": section.get("path") or [heading],
                    "section_path_text": section.get("path_text") or heading,
                    "section_text": section_text,
                    "title_text": heading or title,
                    "body_text": chunk_text_value,
                    "metadata_text": _metadata_search_text(section_metadata),
                }
                chunk_id += 1


def _initialize_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                title_text,
                body_text,
                metadata_text,
                tokenize = 'unicode61 remove_diacritics 0'
            )
            """
        )
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "SQLite FTS5 is unavailable in this Python/SQLite build."
        ) from exc

    connection.execute(
        """
        CREATE TABLE chunk_metadata (
            chunk_id INTEGER PRIMARY KEY,
            source_path TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            preview_text TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            section_heading TEXT,
            section_index INTEGER,
            section_level INTEGER,
            section_path_json TEXT,
            section_path_text TEXT,
            section_text TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def build_sqlite_fts_index(
    parsed_dir: str,
    db_path: str = DEFAULT_DB_PATH,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    include_title_chunk: bool = True,
    use_chunking: bool = True,
) -> str:
    normalized_parsed_dir = os.path.abspath(parsed_dir)
    normalized_db_path = os.path.abspath(db_path)
    signature = _index_signature(
        parsed_dir=normalized_parsed_dir,
        max_chars=max_chars,
        overlap=overlap,
        include_title_chunk=include_title_chunk,
        use_chunking=use_chunking,
    )

    directory = os.path.dirname(normalized_db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with _BUILD_LOCK:
        if os.path.exists(normalized_db_path):
            try:
                with sqlite3.connect(normalized_db_path) as connection:
                    row = connection.execute(
                        "SELECT value FROM index_meta WHERE key = ?",
                        ("signature",),
                    ).fetchone()
                if row and row[0] == signature:
                    return normalized_db_path
            except sqlite3.Error:
                LOG.warning("Rebuilding invalid SQLite FTS index at %s", normalized_db_path)

        with sqlite3.connect(normalized_db_path) as connection:
            connection.execute("DROP TABLE IF EXISTS chunk_fts")
            connection.execute("DROP TABLE IF EXISTS chunk_metadata")
            connection.execute("DROP TABLE IF EXISTS index_meta")
            _initialize_schema(connection)

            rows_inserted = 0
            for row in _iter_chunk_rows(
                parsed_dir=normalized_parsed_dir,
                max_chars=max_chars,
                overlap=overlap,
                include_title_chunk=include_title_chunk,
                use_chunking=use_chunking,
            ):
                connection.execute(
                    """
                    INSERT INTO chunk_metadata (
                        chunk_id,
                        source_path,
                        chunk_type,
                        preview_text,
                        text,
                        metadata_json,
                        section_heading,
                        section_index,
                        section_level,
                        section_path_json,
                        section_path_text,
                        section_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["chunk_id"],
                        row["source_path"],
                        row["chunk_type"],
                        row["preview_text"],
                        row["text"],
                        json.dumps(row["metadata"], ensure_ascii=False),
                        row["section_heading"],
                        row["section_index"],
                        row["section_level"],
                        json.dumps(row["section_path"], ensure_ascii=False),
                        row["section_path_text"],
                        row["section_text"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunk_fts (rowid, title_text, body_text, metadata_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        row["chunk_id"],
                        row["title_text"],
                        row["body_text"],
                        row["metadata_text"],
                    ),
                )
                rows_inserted += 1

            if rows_inserted == 0:
                raise ValueError("No parsed documents with text found to search.")

            connection.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?)",
                ("signature", signature),
            )
            connection.commit()

    return normalized_db_path


def _fts_query_from_user_query(query: str) -> str:
    tokens = tokenize(query)
    if not tokens:
        return ""
    return " OR ".join(dict.fromkeys(tokens))


def sqlite_fts_search(
    parsed_dir: str,
    query: str,
    top_k: int = 5,
    db_path: str = DEFAULT_DB_PATH,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    include_title_chunk: bool = True,
    use_chunking: bool = True,
    title_weight: float = 5.0,
    body_weight: float = 1.0,
    metadata_weight: float = 0.5,
) -> List[dict]:
    if top_k <= 0:
        return []

    match_query = _fts_query_from_user_query(query)
    if not match_query:
        return []

    built_db_path = build_sqlite_fts_index(
        parsed_dir=parsed_dir,
        db_path=db_path,
        max_chars=max_chars,
        overlap=overlap,
        include_title_chunk=include_title_chunk,
        use_chunking=use_chunking,
    )

    with sqlite3.connect(built_db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                m.chunk_id,
                m.source_path,
                m.chunk_type,
                m.preview_text,
                m.text,
                m.metadata_json,
                m.section_heading,
                m.section_index,
                m.section_level,
                m.section_path_json,
                m.section_path_text,
                m.section_text,
                bm25(chunk_fts, ?, ?, ?) AS rank_score
            FROM chunk_fts
            JOIN chunk_metadata AS m ON m.chunk_id = chunk_fts.rowid
            WHERE chunk_fts MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?
            """,
            (title_weight, body_weight, metadata_weight, match_query, top_k),
        ).fetchall()

    results: List[dict] = []
    for row in rows:
        rank_score = float(row[12]) if isinstance(row[12], (int, float)) else 0.0
        results.append(
            {
                "score": -rank_score,
                "chunk_id": int(row[0]),
                "source_path": row[1],
                "chunk_type": row[2],
                "preview_text": row[3],
                "text": row[4],
                "metadata": json.loads(row[5]),
                "section_heading": row[6],
                "section_index": row[7],
                "section_level": row[8],
                "section_path": json.loads(row[9]) if row[9] else None,
                "section_path_text": row[10],
                "section_text": row[11],
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SQLite FTS5 search over parsed Docplus JSON.")
    parser.add_argument("--parsed-dir", required=True, help="Directory with parsed JSON files.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite database path.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters.")
    parser.add_argument(
        "--use-chunking",
        dest="use_chunking",
        action="store_true",
        help="Chunk section text before indexing.",
    )
    parser.add_argument(
        "--no-use-chunking",
        dest="use_chunking",
        action="store_false",
        help="Index one chunk per section.",
    )
    parser.set_defaults(use_chunking=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = sqlite_fts_search(
        parsed_dir=args.parsed_dir,
        query=args.query,
        top_k=args.top_k,
        db_path=args.db_path,
        max_chars=args.max_chars,
        overlap=args.overlap,
        use_chunking=args.use_chunking,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
