import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    search_log = tmp_path / "search_events.csv"
    click_log = tmp_path / "click_events.csv"
    rating_log = tmp_path / "rating_events.csv"

    monkeypatch.setenv("DOCPLUS_SEARCH_LOG_PATH", str(search_log))
    monkeypatch.setenv("DOCPLUS_CLICK_LOG_PATH", str(click_log))
    monkeypatch.setenv("DOCPLUS_RATING_LOG_PATH", str(rating_log))
    monkeypatch.setattr(
        app_module,
        "bm25_search",
        lambda parsed_dir, query, top_k, **kwargs: [
            {
                "score": 1.5,
                "metadata": {
                    "title": "Result title",
                    "source_url": "https://example.com/doc",
                },
                "source_path": "parsed/doc.json",
                "chunk_type": "paragraph",
            }
        ],
    )

    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_search_log_includes_user_name(client, tmp_path):
    response = client.post(
        "/search",
        json={
            "query": "tax rules",
            "method": "bm25",
            "user_name": "Ada Lovelace",
        },
    )

    assert response.status_code == 200
    rows = _read_rows(tmp_path / "search_events.csv")
    assert rows[0]["user_name"] == "Ada Lovelace"


def test_click_and_rating_logs_include_participant_name_fallback(client, tmp_path):
    click_response = client.post(
        "/search/click",
        json={
            "search_id": "search-1",
            "query": "tax rules",
            "requested_method": "bm25",
            "result_method": "bm25",
            "participant_name": "Grace Hopper",
        },
    )
    rating_response = client.post(
        "/search/rating",
        json={
            "search_id": "search-1",
            "query": "tax rules",
            "requested_method": "bm25",
            "result_method": "bm25",
            "participant_name": "Grace Hopper",
            "user_score": 4,
        },
    )

    assert click_response.status_code == 200
    assert rating_response.status_code == 200

    click_rows = _read_rows(tmp_path / "click_events.csv")
    rating_rows = _read_rows(tmp_path / "rating_events.csv")
    assert click_rows[0]["user_name"] == "Grace Hopper"
    assert rating_rows[0]["user_name"] == "Grace Hopper"
