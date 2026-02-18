"""Utilities for stable document-id normalization in evaluation."""
from __future__ import annotations

import os
import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)


def _strip_known_extension(text: str) -> str:
    stem, ext = os.path.splitext(text)
    if ext.lower() in {".pdf", ".doc", ".docx", ".rtf", ".txt"}:
        return stem
    return text


def normalize_doc_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASH_TRANSLATION)
    text = text.strip()
    text = _strip_known_extension(text)
    text = _WS_RE.sub(" ", text).strip()
    return text.casefold()
