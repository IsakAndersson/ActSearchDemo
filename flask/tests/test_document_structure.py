import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_structure import derive_document_sections, get_document_sections


def test_derive_document_sections_splits_on_headings():
    text = """
    General Overview
    Intro text for the document.

    TREATMENT OPTIONS
    First treatment paragraph.
    Second treatment paragraph.

    FOLLOW UP
    Follow up instructions.
    """

    sections = derive_document_sections(text, fallback_title="Demo document")

    assert [section["heading"] for section in sections] == [
        "General Overview",
        "TREATMENT OPTIONS",
        "FOLLOW UP",
    ]
    assert sections[1]["cleaned_text"] == "First treatment paragraph. Second treatment paragraph."


def test_get_document_sections_prefers_stored_sections():
    payload = {
        "text": "Ignored raw text",
        "sections": [
            {
                "heading": "Stored heading",
                "level": 2,
                "text": "Stored raw text",
                "cleaned_text": "Stored raw text",
            }
        ],
    }

    sections = get_document_sections(payload, fallback_title="Fallback")

    assert len(sections) == 1
    assert sections[0]["heading"] == "Stored heading"
    assert sections[0]["level"] == 2
