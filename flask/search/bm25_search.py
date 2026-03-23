"""BM25 search over parsed Docplus content."""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from document_structure import get_document_sections


TOKEN_RE = re.compile(r"[0-9A-Za-zÅÄÖåäö]+")
E5_CHUNK_SIZE = 500
E5_CHUNK_OVERLAP = 50


@dataclass
class ChunkRecord:
    chunk_id: int
    source_path: str
    text: str
    metadata: dict
    chunk_type: str = "body"
    preview_text: str = ""


@dataclass
class BM25Index:
    chunks: List[ChunkRecord]
    doc_lengths: List[int]
    doc_freqs: Dict[str, int]
    avg_doc_len: float
    postings: Dict[str, List[Tuple[int, int]]]


@dataclass
class BM25CacheEntry:
    signature: Tuple[int, int, int]
    index: BM25Index


_BM25_INDEX_CACHE: Dict[Tuple[str, int, int, bool, bool, bool], BM25CacheEntry] = {}
_BM25_CACHE_LOCK = threading.Lock()
LOG = logging.getLogger(__name__)


def iter_parsed_documents(parsed_dir: str) -> Iterable[dict]:
    for filename in sorted(os.listdir(parsed_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(parsed_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            LOG.warning("Skipping invalid parsed JSON %s: %s", path, exc)
            continue
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
    use_cleaned_text: bool,
    use_chunking: bool,
) -> Tuple[List[ChunkRecord], List[Dict[str, int]], List[int], Dict[str, int], float]:
    chunks: List[ChunkRecord] = []
    term_freqs: List[Dict[str, int]] = []
    doc_lengths: List[int] = []
    document_frequencies: Dict[str, int] = {}
    total_length = 0
    chunk_id = 0

    for payload in iter_parsed_documents(parsed_dir):
        text_source_key = "cleaned_text" if use_cleaned_text else "text"
        fallback_key = "text" if use_cleaned_text else "cleaned_text"
        primary_text = payload.get(text_source_key)
        fallback_text = payload.get(fallback_key)
        text = primary_text if isinstance(primary_text, str) else ""
        if not text and isinstance(fallback_text, str):
            text = fallback_text
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
                    preview_text=title,
                )
            )
        if text.strip():
            body_metadata = {**metadata, "title": title} if title else metadata
            if use_chunking:
                sections = get_document_sections(payload, fallback_title=title)
                if sections:
                    for section in sections:
                        section_text = section["cleaned_text"] if use_cleaned_text else section["text"]
                        if not isinstance(section_text, str) or not section_text.strip():
                            continue
                        heading = str(section.get("heading") or title or "").strip()
                        section_metadata = {
                            **body_metadata,
                            "section_heading": heading,
                            "section_index": section.get("index", 0),
                            "section_level": section.get("level", 1),
                            "section_text": section_text,
                        }
                        section_chunks = chunk_text(section_text, max_chars=max_chars, overlap=overlap)
                        for preview_text in section_chunks:
                            chunk_text_value = (
                                f"{heading}\n\n{preview_text}"
                                if heading and preview_text != heading
                                else preview_text
                            )
                            candidate_chunks.append(
                                ChunkRecord(
                                    chunk_id=-1,
                                    source_path=payload["path"],
                                    text=chunk_text_value,
                                    metadata=section_metadata,
                                    chunk_type="section",
                                    preview_text=preview_text,
                                )
                            )
                else:
                    for preview_text in chunk_text(text, max_chars=max_chars, overlap=overlap):
                        candidate_chunks.append(
                            ChunkRecord(
                                chunk_id=-1,
                                source_path=payload["path"],
                                text=preview_text,
                                metadata=body_metadata,
                                chunk_type="body",
                                preview_text=preview_text,
                            )
                        )
            else:
                candidate_chunks.append(
                    ChunkRecord(
                        chunk_id=-1,
                        source_path=payload["path"],
                        text=text,
                        metadata=body_metadata,
                        chunk_type="body",
                        preview_text=text,
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


def _parsed_dir_signature(parsed_dir: str) -> Tuple[int, int, int]:
    file_count = 0
    latest_mtime_ns = 0
    total_size = 0

    with os.scandir(parsed_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            file_count += 1
            stat = entry.stat()
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
            total_size += stat.st_size

    return file_count, latest_mtime_ns, total_size


def _build_postings(term_freqs: List[Dict[str, int]]) -> Dict[str, List[Tuple[int, int]]]:
    postings: Dict[str, List[Tuple[int, int]]] = {}
    for doc_id, freqs in enumerate(term_freqs):
        for token, tf in freqs.items():
            token_postings = postings.get(token)
            if token_postings is None:
                postings[token] = [(doc_id, tf)]
            else:
                token_postings.append((doc_id, tf))
    return postings


def _build_bm25_index(
    parsed_dir: str,
    max_chars: int,
    overlap: int,
    include_title_chunk: bool,
    use_cleaned_text: bool,
    use_chunking: bool,
) -> BM25Index:
    chunks, term_freqs, doc_lengths, doc_freqs, avg_doc_len = build_bm25_corpus(
        parsed_dir=parsed_dir,
        max_chars=max_chars,
        overlap=overlap,
        include_title_chunk=include_title_chunk,
        use_cleaned_text=use_cleaned_text,
        use_chunking=use_chunking,
    )
    postings = _build_postings(term_freqs)
    return BM25Index(
        chunks=chunks,
        doc_lengths=doc_lengths,
        doc_freqs=doc_freqs,
        avg_doc_len=avg_doc_len,
        postings=postings,
    )


def _get_or_build_bm25_index(
    parsed_dir: str,
    max_chars: int,
    overlap: int,
    include_title_chunk: bool,
    use_cleaned_text: bool,
    use_chunking: bool,
) -> BM25Index:
    normalized_dir = os.path.abspath(parsed_dir)
    key = (normalized_dir, max_chars, overlap, include_title_chunk, use_cleaned_text, use_chunking)
    signature = _parsed_dir_signature(normalized_dir)

    with _BM25_CACHE_LOCK:
        cached = _BM25_INDEX_CACHE.get(key)
        if cached is not None and cached.signature == signature:
            return cached.index

    index = _build_bm25_index(
        parsed_dir=normalized_dir,
        max_chars=max_chars,
        overlap=overlap,
        include_title_chunk=include_title_chunk,
        use_cleaned_text=use_cleaned_text,
        use_chunking=use_chunking,
    )

    with _BM25_CACHE_LOCK:
        _BM25_INDEX_CACHE[key] = BM25CacheEntry(signature=signature, index=index)
    return index


def bm25_search(
    parsed_dir: str,
    query: str,
    top_k: int = 5,
    max_chars: int = E5_CHUNK_SIZE,
    overlap: int = E5_CHUNK_OVERLAP,
    include_title_chunk: bool = True,
    use_cleaned_text: bool = True,
    use_chunking: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[dict]:
    if top_k <= 0:
        return []

    index = _get_or_build_bm25_index(
        parsed_dir=parsed_dir,
        max_chars=max_chars,
        overlap=overlap,
        include_title_chunk=include_title_chunk,
        use_cleaned_text=use_cleaned_text,
        use_chunking=use_chunking,
    )
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    token_counts: Dict[str, int] = {}
    for token in query_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    total_docs = len(index.chunks)
    scores: Dict[int, float] = {}
    for token, query_tf in token_counts.items():
        postings = index.postings.get(token)
        if not postings:
            continue

        df = index.doc_freqs.get(token, 0)
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        for doc_id, tf in postings:
            doc_len = index.doc_lengths[doc_id]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / index.avg_doc_len)
            partial_score = query_tf * idf * (numerator / denominator)
            scores[doc_id] = scores.get(doc_id, 0.0) + partial_score

    if not scores:
        return []

    top_scores = heapq.nlargest(top_k, scores.items(), key=lambda pair: pair[1])
    results = []
    for doc_id, score in top_scores:
        record = index.chunks[doc_id]
        results.append(
            {
                "score": score,
                "chunk_id": record.chunk_id,
                "text": record.text,
                "metadata": record.metadata,
                "source_path": record.source_path,
                "chunk_type": record.chunk_type,
                "preview_text": record.preview_text,
                "section_heading": record.metadata.get("section_heading"),
                "section_index": record.metadata.get("section_index"),
                "section_level": record.metadata.get("section_level"),
                "section_text": record.metadata.get("section_text"),
            }
        )
    return results


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
    parser.add_argument(
        "--use-cleaned-text",
        dest="use_cleaned_text",
        action="store_true",
        help="Use cleaned_text field from parsed JSON as BM25 source text.",
    )
    parser.add_argument(
        "--no-use-cleaned-text",
        dest="use_cleaned_text",
        action="store_false",
        help="Use raw text field from parsed JSON as BM25 source text.",
    )
    parser.set_defaults(use_cleaned_text=True)
    parser.add_argument(
        "--use-chunking",
        dest="use_chunking",
        action="store_true",
        help="Chunk document text before BM25 indexing.",
    )
    parser.add_argument(
        "--no-use-chunking",
        dest="use_chunking",
        action="store_false",
        help="Index whole document text as one BM25 unit (no chunking).",
    )
    parser.set_defaults(use_chunking=True)
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
        use_cleaned_text=args.use_cleaned_text,
        use_chunking=args.use_chunking,
        k1=args.k1,
        b=args.b,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
