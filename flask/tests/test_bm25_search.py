import json
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from search.bm25_search import bm25_search


def test_bm25_search_returns_section_metadata(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
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

    results = bm25_search(str(parsed_dir), "adrenalin", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_type"] in {"section", "section_title"}
    assert results[0]["section_heading"] == "SECTION ONE"
    assert results[0]["section_text"] == "adrenalin guidance for treatment"
    assert results[0]["section_path"] == ["SECTION ONE"]


def test_bm25_search_indexes_section_titles_even_without_body_text(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "doc1.json").write_text(
        json.dumps(
            {
                "raw_text": "ignored",
                "text": "ignored",
                "sections": [
                    {
                        "heading": "Parent Chapter",
                        "level": 1,
                        "raw_text": "",
                        "text": "",
                    },
                    {
                        "heading": "Child Chapter",
                        "level": 2,
                        "raw_text": "body text",
                        "text": "body text",
                    },
                ],
                "metadata": {"title": "Emergency care", "source_url": "https://example.com/doc"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = bm25_search(str(parsed_dir), "Parent Chapter", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_type"] == "section_title"
    assert results[0]["section_heading"] == "Parent Chapter"


def test_bm25_search_prefixes_first_split_section_chunk_with_heading(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "doc1.json").write_text(
        json.dumps(
            {
                "raw_text": "ignored",
                "text": "ignored",
                "sections": [
                    {
                        "heading": "SECTION ONE",
                        "level": 1,
                        "raw_text": "alpha beta gamma delta",
                        "text": "alpha beta gamma delta",
                    }
                ],
                "metadata": {"title": "Emergency care", "source_url": "https://example.com/doc"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = bm25_search(str(parsed_dir), "alpha", top_k=5, max_chars=10, overlap=0)

    section_results = [item for item in results if item["chunk_type"] == "section"]
    assert section_results
    assert section_results[0]["text"].startswith("SECTION ONE\n\n")
