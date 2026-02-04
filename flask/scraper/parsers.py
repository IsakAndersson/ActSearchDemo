"""Text extraction helpers for downloaded documents."""
from __future__ import annotations

import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError:  # pragma: no cover - optional dependency
    pdf_extract_text = None

try:
    from docx import Document
except ImportError:  # pragma: no cover - optional dependency
    Document = None


def extract_text(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    if ext == ".pdf":
        return _extract_pdf(filename)
    if ext in {".docx"}:
        return _extract_docx(filename)
    logger.warning("No parser available for %s; returning empty text.", ext)
    return ""


def _extract_pdf(filename: str) -> str:
    if not pdf_extract_text:
        logger.warning("pdfminer.six not installed; skipping PDF extraction.")
        return ""
    return pdf_extract_text(filename) or ""


def _extract_docx(filename: str) -> str:
    if not Document:
        logger.warning("python-docx not installed; skipping DOCX extraction.")
        return ""
    doc = Document(filename)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)
