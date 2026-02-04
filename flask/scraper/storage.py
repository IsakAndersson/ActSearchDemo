"""Storage utilities for scraped documents."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


@dataclass
class DownloadResult:
    url: str
    filename: str
    extracted_text_len: int


class DocumentStore:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.binary_dir = os.path.join(output_dir, "documents")
        self.text_dir = os.path.join(output_dir, "parsed")
        os.makedirs(self.binary_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)

    def write_binary(self, url: str, content: bytes) -> str:
        filename = self._filename_from_url(url)
        path = os.path.join(self.binary_dir, filename)
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def write_text(self, binary_path: str, text: str, metadata: dict) -> str:
        basename = os.path.splitext(os.path.basename(binary_path))[0]
        text_path = os.path.join(self.text_dir, f"{basename}.json")
        payload = {
            "binary_path": binary_path,
            "text": text,
            "metadata": metadata,
        }
        with open(text_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return text_path

    def _filename_from_url(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        ext = os.path.splitext(url)[1] or ".bin"
        return f"doc_{digest}{ext}"
