import json
import os
import sqlite3
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from search.sqlite_fts_search import build_sqlite_fts_index, sqlite_fts_search


def test_sqlite_fts_search_returns_section_metadata(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    db_path = tmp_path / "index.sqlite3"
    (parsed_dir / "doc1.json").write_text(
        json.dumps(
            {
                "raw_text": "SECTION ONE\nadrenalin guidance for treatment",
                "text": "SECTION ONE adrenalin guidance for treatment",
                "sections": [
                    {
                        "heading": "SECTION ONE",
                        "level": 1,
                        "raw_text": "adrenalin guidance for treatment",
                        "text": "adrenalin guidance for treatment",
                    }
                ],
                "metadata": {"title": "Emergency care", "source_url": "https://example.com/doc"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = sqlite_fts_search(str(parsed_dir), "adrenalin", top_k=3, db_path=str(db_path))

    assert db_path.exists()
    assert results
    assert results[0]["chunk_type"] in {"section", "section_title"}
    assert results[0]["section_heading"] == "SECTION ONE"
    assert results[0]["section_text"] == "adrenalin guidance for treatment"
    assert results[0]["section_path"] == ["SECTION ONE"]


def test_sqlite_fts_search_indexes_title_chunks(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    db_path = tmp_path / "index.sqlite3"
    (parsed_dir / "doc1.json").write_text(
        json.dumps(
            {
                "raw_text": "ignored",
                "text": "ignored",
                "sections": [
                    {
                        "heading": "Body Section",
                        "level": 1,
                        "raw_text": "",
                        "text": "",
                    }
                ],
                "metadata": {"title": "Rare Title Phrase", "source_url": "https://example.com/doc"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = sqlite_fts_search(str(parsed_dir), "Rare Title Phrase", top_k=1, db_path=str(db_path))

    assert len(results) == 1
    assert results[0]["chunk_type"] == "title"
    assert results[0]["metadata"]["title"] == "Rare Title Phrase"


def test_build_sqlite_fts_index_stores_signature(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    db_path = tmp_path / "index.sqlite3"
    (parsed_dir / "doc1.json").write_text(
        json.dumps(
            {
                "raw_text": "alpha beta",
                "text": "alpha beta",
                "sections": [
                    {
                        "heading": "Section",
                        "level": 1,
                        "raw_text": "alpha beta",
                        "text": "alpha beta",
                    }
                ],
                "metadata": {"title": "Doc 1"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    built_path = build_sqlite_fts_index(str(parsed_dir), db_path=str(db_path))

    with sqlite3.connect(built_path) as connection:
        row = connection.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            ("signature",),
        ).fetchone()

    assert row is not None
    assert "\"parsed_dir\"" in row[0]


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 9, 0), reason="FTS5 requires modern SQLite")
def test_sqlite_runtime_supports_sqlite():
    assert sqlite3.sqlite_version_info >= (3, 9, 0)
