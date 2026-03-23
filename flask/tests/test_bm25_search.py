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
                "text": "SECTION ONE\nadrenalin guidance for treatment",
                "cleaned_text": "SECTION ONE adrenalin guidance for treatment",
                "sections": [
                    {
                        "heading": "SECTION ONE",
                        "level": 1,
                        "text": "adrenalin guidance for treatment",
                        "cleaned_text": "adrenalin guidance for treatment",
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
    assert results[0]["chunk_type"] == "section"
    assert results[0]["preview_text"] == "adrenalin guidance for treatment"
    assert results[0]["section_heading"] == "SECTION ONE"
    assert results[0]["section_text"] == "adrenalin guidance for treatment"
