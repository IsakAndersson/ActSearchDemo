"""Helpers for deriving section-aware document chunks from extracted text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


_WHITESPACE_RE = re.compile(r"\s+")
_PAGE_MARKER_RE = re.compile(r"^\s*(?:page|sidan)\s+\d+(?:\s+(?:of|av)\s+\d+)?\s*$", re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLCM]+)[\)\.\-:]?\s+\S", re.IGNORECASE)
_TOC_HEADING_RE = re.compile(r"^(?:innehåll|contents?|table of contents)$", re.IGNORECASE)
_TOC_TRAILING_PAGE_RE = re.compile(r"(?:\s*[._·\-…]{2,}\s*|\s{2,})(\d+)\s*$")
_BODY_NOISE_RE = re.compile(
    r"^(?:docplus-id|version|handlingstyp|godkänt den|ansvarig|gäller för)\s*:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextLine:
    raw: str
    normalized: str
    leading_spaces: int
    index: int


@dataclass(frozen=True)
class TocEntry:
    heading: str
    level: int
    page: Optional[int]
    source_index: int


def clean_text(text: str) -> str:
    """Normalize extracted text into a single searchable line."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _normalize_line(line: str) -> str:
    return _WHITESPACE_RE.sub(" ", line).strip()


def _parse_text_lines(text: str) -> List[TextLine]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines: List[TextLine] = []
    for index, raw_line in enumerate(normalized.split("\n")):
        stripped_left = raw_line.lstrip(" ")
        leading_spaces = len(raw_line) - len(stripped_left)
        lines.append(
            TextLine(
                raw=raw_line,
                normalized=_normalize_line(raw_line),
                leading_spaces=leading_spaces,
                index=index,
            )
        )
    return lines


def _is_heading_candidate(line: str) -> bool:
    candidate = _normalize_line(line)
    if not candidate:
        return False
    if len(candidate) < 3 or len(candidate) > 140:
        return False
    if _PAGE_MARKER_RE.match(candidate) or _BODY_NOISE_RE.match(candidate):
        return False
    if candidate.endswith((".", ";", "?", "!")):
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
        or (2 <= word_count <= 10 and titleish_words >= max(2, word_count - 1))
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


def _strip_toc_page_reference(line: str) -> tuple[str, Optional[int], bool]:
    normalized = _normalize_line(line)
    match = _TOC_TRAILING_PAGE_RE.search(normalized)
    if not match:
        return normalized, None, False
    title = _normalize_line(normalized[: match.start()])
    if title.endswith((".", "-", "·", "_", "…")):
        title = _normalize_line(title.rstrip(".-·_… "))
    page_value = int(match.group(1))
    return title, page_value, True


def _estimate_toc_level(heading: str, leading_spaces: int, indent_levels: Dict[int, int]) -> int:
    detected = _detect_heading_level(heading)
    if detected > 1:
        return detected
    indent_key = max(0, leading_spaces // 2)
    if indent_key not in indent_levels:
        indent_levels[indent_key] = len(indent_levels) + 1
    return indent_levels[indent_key]


def _extract_toc_entries(lines: List[TextLine]) -> tuple[List[TocEntry], int]:
    max_scan = min(len(lines), 160)
    toc_start: Optional[int] = None
    for index, line in enumerate(lines[:max_scan]):
        if _TOC_HEADING_RE.match(line.normalized):
            toc_start = index
            break
    if toc_start is None:
        return [], -1

    entries: List[TocEntry] = []
    indent_levels: Dict[int, int] = {}
    pending_parts: List[str] = []
    pending_indent = 0
    after_heading = False
    end_index = toc_start

    for index in range(toc_start + 1, len(lines)):
        line = lines[index]
        if not line.normalized:
            if entries and pending_parts:
                pending_parts = []
            if entries:
                after_heading = True
            continue

        if _TOC_HEADING_RE.match(line.normalized):
            continue
        if _PAGE_MARKER_RE.match(line.normalized):
            continue
        if after_heading and entries and line.normalized == entries[0].heading:
            end_index = index - 1
            break

        title_part, page_value, has_page_reference = _strip_toc_page_reference(line.raw)
        if not title_part:
            if entries:
                after_heading = True
            continue

        if has_page_reference:
            title_parts = [*pending_parts, title_part]
            heading = _normalize_line(" ".join(part for part in title_parts if part))
            pending_parts = []
            if heading and len(heading.split()) <= 24:
                entries.append(
                    TocEntry(
                        heading=heading,
                        level=_estimate_toc_level(heading, pending_indent or line.leading_spaces, indent_levels),
                        page=page_value,
                        source_index=index,
                    )
                )
                end_index = index
                after_heading = True
            continue

        if entries and after_heading and _is_heading_candidate(line.normalized):
            end_index = index - 1
            break

        pending_parts.append(title_part)
        pending_indent = line.leading_spaces

    return (entries if len(entries) >= 2 else []), end_index


def _find_heading_line_index(lines: List[TextLine], heading: str, start_at: int) -> Optional[int]:
    target = _normalize_line(heading)
    for index in range(max(start_at, 0), len(lines)):
        candidate = lines[index].normalized
        if not candidate:
            continue
        if candidate == target:
            return index
        if candidate.startswith(f"{target} "):
            remainder = candidate[len(target) :].strip()
            if remainder and remainder[0].islower():
                return index
    return None


def _derive_section_paths(sections: List[Dict[str, Any]]) -> None:
    path_stack: List[str] = []
    for section in sections:
        level = max(int(section.get("level", 1)), 1)
        heading = str(section.get("heading") or "").strip()
        while len(path_stack) >= level:
            path_stack.pop()
        path_stack.append(heading)
        section["path"] = path_stack.copy()
        section["path_text"] = " > ".join(path_stack)


def _build_section_payload(
    *,
    heading: str,
    body_lines: List[str],
    index: int,
    level: Optional[int] = None,
    page: Optional[int] = None,
) -> Dict[str, Any]:
    raw_text = "\n".join(line for line in body_lines if _normalize_line(line)).strip()
    cleaned = clean_text(raw_text)
    resolved_level = level if isinstance(level, int) and level > 0 else _detect_heading_level(heading)
    return {
        "index": index,
        "heading": heading,
        "title": heading,
        "level": resolved_level,
        "page": page,
        "raw_text": raw_text,
        "text": cleaned,
    }


def _derive_sections_from_toc(lines: List[TextLine], fallback_title: Optional[str]) -> List[Dict[str, Any]]:
    toc_entries, toc_end_index = _extract_toc_entries(lines)
    if not toc_entries:
        return []

    sections: List[Dict[str, Any]] = []
    search_index = toc_end_index + 1
    body_start = search_index
    matched_entries: List[tuple[TocEntry, int]] = []

    for entry in toc_entries:
        match_index = _find_heading_line_index(lines, entry.heading, search_index)
        if match_index is None:
            continue
        matched_entries.append((entry, match_index))
        search_index = match_index + 1

    if len(matched_entries) < 2:
        return []

    first_match_index = matched_entries[0][1]
    if body_start < first_match_index:
        preface_lines = [line.raw for line in lines[body_start:first_match_index] if line.normalized]
        if preface_lines:
            sections.append(
                _build_section_payload(
                    heading=_normalize_line(fallback_title or "Document"),
                    body_lines=preface_lines,
                    index=0,
                    level=1,
                )
            )

    for match_pos, (entry, heading_index) in enumerate(matched_entries):
        next_heading_index = (
            matched_entries[match_pos + 1][1]
            if match_pos + 1 < len(matched_entries)
            else len(lines)
        )
        body_lines = [line.raw for line in lines[heading_index + 1 : next_heading_index] if line.normalized]
        sections.append(
            _build_section_payload(
                heading=entry.heading,
                body_lines=body_lines,
                index=len(sections),
                level=entry.level,
                page=entry.page,
            )
        )

    sections = [section for section in sections if section["heading"] and (section["text"] or section["heading"])]
    for index, section in enumerate(sections):
        section["index"] = index
    _derive_section_paths(sections)
    return sections


def _derive_sections_from_headings(lines: List[TextLine], fallback_title: Optional[str]) -> List[Dict[str, Any]]:
    content_lines = [line.normalized for line in lines if line.normalized]
    if not content_lines:
        return []

    sections: List[Dict[str, Any]] = []
    current_heading = _normalize_line(fallback_title or "Document")
    current_body: List[str] = []

    for line in content_lines:
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

    sections = [section for section in sections if section["text"]]
    if sections:
        _derive_section_paths(sections)
    return sections


def derive_document_sections(text: str, fallback_title: Optional[str] = None) -> List[Dict[str, Any]]:
    lines = _parse_text_lines(text)
    if not any(line.normalized for line in lines):
        return []

    toc_sections = _derive_sections_from_toc(lines, fallback_title=fallback_title)
    if toc_sections:
        return toc_sections

    sections = _derive_sections_from_headings(lines, fallback_title=fallback_title)
    if sections:
        return sections

    cleaned = clean_text(text)
    if not cleaned:
        return []
    section = {
        "index": 0,
        "heading": _normalize_line(fallback_title or "Document"),
        "title": _normalize_line(fallback_title or "Document"),
        "level": 1,
        "page": None,
        "raw_text": text.strip(),
        "text": cleaned,
    }
    _derive_section_paths([section])
    return [section]


def get_document_sections(payload: Dict[str, Any], fallback_title: Optional[str] = None) -> List[Dict[str, Any]]:
    stored_sections = payload.get("sections")
    if isinstance(stored_sections, list):
        valid_sections: List[Dict[str, Any]] = []
        for index, item in enumerate(stored_sections):
            if not isinstance(item, dict):
                continue
            heading = _normalize_line(str(item.get("heading") or item.get("title") or fallback_title or "Document"))
            text_value = item.get("text")
            stored_raw_text = item.get("raw_text")
            if not isinstance(text_value, str):
                text_value = ""
            if not isinstance(stored_raw_text, str):
                stored_raw_text = text_value
            text_value = text_value.strip() or clean_text(stored_raw_text)
            if not text_value and not heading:
                continue
            level_raw = item.get("level")
            level = int(level_raw) if isinstance(level_raw, int) and level_raw > 0 else _detect_heading_level(heading)
            page_value = item.get("page")
            path_value = item.get("path")
            if not isinstance(path_value, list) or not all(isinstance(part, str) for part in path_value):
                path_value = []
            valid_sections.append(
                {
                    "index": index,
                    "heading": heading,
                    "title": heading,
                    "level": level,
                    "page": page_value if isinstance(page_value, int) and page_value > 0 else None,
                    "raw_text": stored_raw_text.strip(),
                    "text": text_value,
                    "path": [_normalize_line(part) for part in path_value if _normalize_line(part)],
                    "path_text": str(item.get("path_text") or "").strip(),
                }
            )
        if valid_sections:
            if not all(section.get("path") for section in valid_sections):
                _derive_section_paths(valid_sections)
            return valid_sections

    text = payload.get("text")
    if not isinstance(text, str):
        return []
    return derive_document_sections(text, fallback_title=fallback_title)
