"""Helpers for deriving section-aware document chunks from extracted text."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_WHITESPACE_RE = re.compile(r"\s+")
_PAGE_MARKER_RE = re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLCM]+)[\)\.\-:]?\s+\S", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize extracted text into a single searchable line."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _normalize_line(line: str) -> str:
    return _WHITESPACE_RE.sub(" ", line).strip()


def _is_heading_candidate(line: str) -> bool:
    candidate = _normalize_line(line)
    if not candidate:
        return False
    if len(candidate) < 3 or len(candidate) > 140:
        return False
    if _PAGE_MARKER_RE.match(candidate):
        return False
    if candidate.endswith((".", ";", "?", "!", ":")):
        return False

    letters = [char for char in candidate if char.isalpha()]
    if not letters:
        return False

    word_count = len(candidate.split())
    if word_count > 18:
        return False

    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    titleish_words = sum(
        1
        for word in candidate.split()
        if word[:1].isupper() and any(char.islower() for char in word[1:])
    )

    return (
        _NUMBERED_HEADING_RE.match(candidate) is not None
        or upper_ratio >= 0.72
        or (
            2 <= word_count <= 10
            and titleish_words >= max(2, word_count - 1)
        )
    )


def _detect_heading_level(heading: str) -> int:
    normalized = _normalize_line(heading)
    numbered_match = re.match(r"^(\d+(?:\.\d+)*)", normalized)
    if numbered_match:
        return numbered_match.group(1).count(".") + 1
    roman_match = re.match(r"^([IVXLCM]+)[\)\.\-:]?\s+", normalized, re.IGNORECASE)
    if roman_match:
        return 1
    return 1


def _build_section_payload(
    *,
    heading: str,
    body_lines: List[str],
    index: int,
) -> Dict[str, Any]:
    raw_text = "\n".join(body_lines).strip()
    cleaned = clean_text(raw_text)
    return {
        "index": index,
        "heading": heading,
        "level": _detect_heading_level(heading),
        "raw_text": raw_text,
        "text": cleaned,
        "cleaned_text": cleaned,
    }


def derive_document_sections(text: str, fallback_title: Optional[str] = None) -> List[Dict[str, Any]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    raw_lines = [_normalize_line(line) for line in normalized.split("\n")]
    lines = [line for line in raw_lines if line]
    if not lines:
        return []

    sections: List[Dict[str, Any]] = []
    current_heading = _normalize_line(fallback_title or "Document")
    current_body: List[str] = []

    for line in lines:
        if _is_heading_candidate(line):
            if current_body:
                sections.append(
                    _build_section_payload(
                        heading=current_heading or _normalize_line(fallback_title or "Document"),
                        body_lines=current_body,
                        index=len(sections),
                    )
                )
                current_body = []
            current_heading = line
            continue
        current_body.append(line)

    if current_body:
        sections.append(
            _build_section_payload(
                heading=current_heading or _normalize_line(fallback_title or "Document"),
                body_lines=current_body,
                index=len(sections),
            )
        )

    if sections:
        return sections

    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [
        {
            "index": 0,
            "heading": _normalize_line(fallback_title or "Document"),
            "level": 1,
            "raw_text": text.strip(),
            "text": cleaned,
            "cleaned_text": cleaned,
        }
    ]


def get_document_sections(payload: Dict[str, Any], fallback_title: Optional[str] = None) -> List[Dict[str, Any]]:
    stored_sections = payload.get("sections")
    if isinstance(stored_sections, list):
        valid_sections: List[Dict[str, Any]] = []
        for index, item in enumerate(stored_sections):
            if not isinstance(item, dict):
                continue
            heading = _normalize_line(str(item.get("heading") or fallback_title or "Document"))
            raw_text = item.get("text")
            cleaned = item.get("cleaned_text")
            stored_raw_text = item.get("raw_text")
            if not isinstance(raw_text, str):
                raw_text = ""
            if not isinstance(stored_raw_text, str):
                stored_raw_text = raw_text
            if not isinstance(cleaned, str):
                cleaned = clean_text(raw_text)
            cleaned = cleaned.strip()
            if not cleaned:
                continue
            level_raw = item.get("level")
            level = int(level_raw) if isinstance(level_raw, int) and level_raw > 0 else _detect_heading_level(heading)
            valid_sections.append(
                {
                    "index": index,
                    "heading": heading,
                    "level": level,
                    "raw_text": stored_raw_text.strip(),
                    "text": cleaned,
                    "cleaned_text": cleaned,
                }
            )
        if valid_sections:
            return valid_sections

    text = payload.get("text")
    if not isinstance(text, str):
        return []
    return derive_document_sections(text, fallback_title=fallback_title)
