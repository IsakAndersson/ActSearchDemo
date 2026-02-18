"""BM25 search over parsed Docplus content."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse


TOKEN_RE = re.compile(r"[0-9A-Za-zÅÄÖåäö]+")
E5_CHUNK_SIZE = 250
E5_CHUNK_OVERLAP = 50


@dataclass
class ChunkRecord:
    chunk_id: int
    source_path: str
    text: str
    metadata: dict
    chunk_type: str = "body"


def iter_parsed_documents(parsed_dir: str) -> Iterable[dict]:
    for filename in sorted(os.listdir(parsed_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(parsed_dir, filename)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield {"path": path, **payload}


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        word_len = len(word) + 1
        if current and current_len + word_len > max_chars:
            chunks.append(" ".join(current).strip())
            if overlap > 0:
                overlap_words = []
                remaining = overlap
                for previous_word in reversed(current):
                    remaining -= len(previous_word) + 1
                    overlap_words.append(previous_word)
                    if remaining <= 0:
                        break
                current = list(reversed(overlap_words))
                current_len = sum(len(w) + 1 for w in current)
            else:
                current = []
                current_len = 0
        current.append(word)
        current_len += word_len
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _extract_title_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    query = parse_qs(parsed.query)
    for key in ("filename", "file", "name"):
        values = query.get(key)
        if values:
            title = unquote(values[0]).strip()
            if title:
                return title
    last = parsed.path.split("/")[-1].strip()
    if last and last.lower() != "getdocument":
        return unquote(last)
    return ""


def extract_title(payload: dict) -> str:
    metadata = payload.get("metadata") or {}
    for key in ("title", "document_title", "doc_title", "name", "filename", "file_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source_url = metadata.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        return _extract_title_from_url(source_url.strip())
    return ""


def build_bm25_corpus(
    parsed_dir: str,
    max_chars: int,
    overlap: int,
    include_title_chunk: bool,
) -> Tuple[List[ChunkRecord], List[Dict[str, int]], List[int], Dict[str, int], float]:
    chunks: List[ChunkRecord] = []
    term_freqs: List[Dict[str, int]] = []
    doc_lengths: List[int] = []
    document_frequencies: Dict[str, int] = {}
    total_length = 0
    chunk_id = 0

    for payload in iter_parsed_documents(parsed_dir):
        text = payload.get("text") or ""
        metadata = payload.get("metadata", {}) or {}
        title = extract_title(payload)

        candidate_chunks: List[ChunkRecord] = []
        if include_title_chunk and title:
            candidate_chunks.append(
                ChunkRecord(
                    chunk_id=-1,
                    source_path=payload["path"],
                    text=title,
                    metadata={**metadata, "title": title},
                    chunk_type="title",
                )
            )
        if text.strip():
            body_metadata = {**metadata, "title": title} if title else metadata
            for chunk_text_value in chunk_text(text, max_chars=max_chars, overlap=overlap):
                candidate_chunks.append(
                    ChunkRecord(
                        chunk_id=-1,
                        source_path=payload["path"],
                        text=chunk_text_value,
                        metadata=body_metadata,
                        chunk_type="body",
                    )
                )

        for candidate in candidate_chunks:
            chunk_text_value = candidate.text
            tokens = tokenize(chunk_text_value)
            if not tokens:
                continue
            freqs: Dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            for token in set(tokens):
                document_frequencies[token] = document_frequencies.get(token, 0) + 1

            doc_len = len(tokens)
            total_length += doc_len
            doc_lengths.append(doc_len)
            term_freqs.append(freqs)
            candidate.chunk_id = chunk_id
            chunks.append(
                candidate
            )
            chunk_id += 1

    if not chunks:
        raise ValueError("No parsed documents with text found to search.")

    avg_doc_len = total_length / len(chunks)
    return chunks, term_freqs, doc_lengths, document_frequencies, avg_doc_len


def bm25_search(
    parsed_dir: str,
    query: str,
    top_k: int = 5,
    max_chars: int = E5_CHUNK_SIZE,
    overlap: int = E5_CHUNK_OVERLAP,
    include_title_chunk: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[dict]:
    chunks, term_freqs, doc_lengths, doc_freqs, avg_doc_len = build_bm25_corpus(
        parsed_dir=parsed_dir,
        max_chars=max_chars,
        overlap=overlap,
        include_title_chunk=include_title_chunk,
    )
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    total_docs = len(chunks)
    idf: Dict[str, float] = {}
    for token in set(query_tokens):
        df = doc_freqs.get(token, 0)
        idf[token] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

    results = []
    for record, freqs, doc_len in zip(chunks, term_freqs, doc_lengths):
        score = 0.0
        for token in query_tokens:
            tf = freqs.get(token, 0)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf.get(token, 0.0) * (numerator / denominator)
        if score <= 0:
            continue
        results.append(
            {
                "score": score,
                "chunk_id": record.chunk_id,
                "text": record.text,
                "metadata": record.metadata,
                "source_path": record.source_path,
                "chunk_type": record.chunk_type,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BM25 search over parsed Docplus JSON.")
    parser.add_argument("--parsed-dir", required=True, help="Directory with parsed JSON files.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=E5_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {E5_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=E5_CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {E5_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--include-title-chunk",
        dest="include_title_chunk",
        action="store_true",
        help="Include one title-only chunk per document when title can be extracted.",
    )
    parser.add_argument(
        "--no-include-title-chunk",
        dest="include_title_chunk",
        action="store_false",
        help="Disable title-only chunks.",
    )
    parser.set_defaults(include_title_chunk=True)
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = bm25_search(
        parsed_dir=args.parsed_dir,
        query=args.query,
        top_k=args.top_k,
        max_chars=args.max_chars,
        overlap=args.overlap,
        include_title_chunk=args.include_title_chunk,
        k1=args.k1,
        b=args.b,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
