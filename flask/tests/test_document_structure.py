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
    assert sections[1]["text"] == "First treatment paragraph. Second treatment paragraph."
    assert sections[1]["raw_text"] == "First treatment paragraph.\nSecond treatment paragraph."


def test_get_document_sections_prefers_stored_sections():
    payload = {
        "text": "Ignored raw text",
        "sections": [
            {
                "heading": "Stored heading",
                "level": 2,
                "raw_text": "Stored raw\ntext",
                "text": "Stored raw text",
            }
        ],
    }

    sections = get_document_sections(payload, fallback_title="Fallback")

    assert len(sections) == 1
    assert sections[0]["heading"] == "Stored heading"
    assert sections[0]["level"] == 2
    assert sections[0]["text"] == "Stored raw text"
    assert sections[0]["raw_text"] == "Stored raw\ntext"


def test_derive_document_sections_prefers_table_of_contents_and_keeps_levels():
    text = """
    Demo Document

    Innehåll
    Overview ................................ 1
      Details ................................ 2
        Deep Dive ............................ 3
    Follow Up ............................... 4

    Overview
    Intro body.

    Details
    Detail body.

    Deep Dive
    Deep body.

    Follow Up
    Final body.
    """

    sections = derive_document_sections(text, fallback_title="Demo Document")

    assert [section["heading"] for section in sections] == [
        "Overview",
        "Details",
        "Deep Dive",
        "Follow Up",
    ]
    assert [section["level"] for section in sections] == [1, 2, 3, 1]
    assert sections[2]["path"] == ["Overview", "Details", "Deep Dive"]
    assert sections[3]["path_text"] == "Follow Up"


def test_derive_document_sections_does_not_split_tabular_values_into_headings():
    text = """
    Innehåll
    Titrering ................................ 1
    Uppföljning .............................. 2

    Titrering
    Steg 1
    5 mg
    Dos
    1+0+0
    10 mg
    Dos
    1+0+1

    Uppföljning
    Kontrollera blodtryck efter tre veckor.
    """

    sections = derive_document_sections(text, fallback_title="Demo Document")

    assert [section["heading"] for section in sections] == ["Titrering", "Uppföljning"]
    assert "5 mg" in sections[0]["text"]
    assert "10 mg" in sections[0]["text"]
