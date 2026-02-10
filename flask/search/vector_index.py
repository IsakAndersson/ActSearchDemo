"""Build and query a FAISS vector index for scraped Docplus content."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "KBLab/bert-base-swedish-cased"


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


def _extract_title_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
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
    return None


def extract_title(payload: dict) -> Optional[str]:
    metadata = payload.get("metadata") or {}
    for key in ("title", "document_title", "doc_title", "name", "filename", "file_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source_url = metadata.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        title = _extract_title_from_url(source_url.strip())
        if title:
            return title
    return None


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings).astype("float32")


def resolve_device(device_preference: str) -> torch.device:
    if device_preference == "cpu":
        return torch.device("cpu")
    if device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_index(
    parsed_dir: str,
    output_dir: str,
    model_name: str,
    max_chars: int,
    overlap: int,
    batch_size: int,
    device_preference: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = resolve_device(device_preference)
    model.to(device)
    model.eval()

    chunks: List[ChunkRecord] = []
    texts: List[str] = []
    chunk_id = 0
    for payload in iter_parsed_documents(parsed_dir):
        text = payload.get("text") or ""
        if not text.strip():
            continue
        for chunk_text_value in chunk_text(text, max_chars=max_chars, overlap=overlap):
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source_path=payload["path"],
                    text=chunk_text_value,
                    metadata=payload.get("metadata", {}),
                )
            )
            texts.append(chunk_text_value)
            chunk_id += 1

    if not texts:
        raise ValueError("No parsed documents with text found to index.")

    embeddings = embed_texts(texts, tokenizer, model, device, batch_size)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss_path = os.path.join(output_dir, "docplus.faiss")
    faiss.write_index(index, faiss_path)

    metadata_path = os.path.join(output_dir, "docplus_metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for record in chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": record.chunk_id,
                        "source_path": record.source_path,
                        "text": record.text,
                        "metadata": record.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_index_with_titles(
    parsed_dir: str,
    output_dir: str,
    model_name: str,
    max_chars: int,
    overlap: int,
    batch_size: int,
    device_preference: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = resolve_device(device_preference)
    model.to(device)
    model.eval()

    chunks: List[ChunkRecord] = []
    texts: List[str] = []
    chunk_id = 0
    for payload in iter_parsed_documents(parsed_dir):
        text = payload.get("text") or ""
        metadata = payload.get("metadata", {}) or {}
        title = extract_title(payload)
        if title:
            title_metadata = {**metadata, "title": title}
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source_path=payload["path"],
                    text=title,
                    metadata=title_metadata,
                    chunk_type="title",
                )
            )
            texts.append(title)
            chunk_id += 1

        if not text.strip():
            continue
        body_metadata = {**metadata, "title": title} if title else metadata
        for chunk_text_value in chunk_text(text, max_chars=max_chars, overlap=overlap):
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source_path=payload["path"],
                    text=chunk_text_value,
                    metadata=body_metadata,
                    chunk_type="body",
                )
            )
            texts.append(chunk_text_value)
            chunk_id += 1

    if not texts:
        raise ValueError("No parsed documents with text found to index.")

    embeddings = embed_texts(texts, tokenizer, model, device, batch_size)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss_path = os.path.join(output_dir, "docplus_titles.faiss")
    faiss.write_index(index, faiss_path)

    metadata_path = os.path.join(output_dir, "docplus_titles_metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for record in chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": record.chunk_id,
                        "source_path": record.source_path,
                        "text": record.text,
                        "metadata": record.metadata,
                        "chunk_type": record.chunk_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def query_index(
    index_path: str,
    metadata_path: str,
    query: str,
    model_name: str,
    top_k: int,
    device_preference: str,
) -> List[dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = resolve_device(device_preference)
    model.to(device)
    model.eval()

    query_embedding = embed_texts([query], tokenizer, model, device, batch_size=1)
    faiss.normalize_L2(query_embedding)
    index = faiss.read_index(index_path)
    scores, indices = index.search(query_embedding, top_k)

    metadata = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            metadata.append(json.loads(line))

    results = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        result = {
            "score": float(score),
            "chunk_id": entry["chunk_id"],
            "text": entry["text"],
            "metadata": entry.get("metadata", {}),
            "source_path": entry["source_path"],
        }
        if "chunk_type" in entry:
            result["chunk_type"] = entry["chunk_type"]
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or query a Docplus vector index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a FAISS index from parsed JSON.")
    build.add_argument("--parsed-dir", required=True, help="Directory with parsed JSON files.")
    build.add_argument("--output-dir", required=True, help="Directory to store FAISS index.")
    build.add_argument("--model-name", default=DEFAULT_MODEL, help="Hugging Face model name.")
    build.add_argument("--max-chars", type=int, default=1200, help="Chunk size in characters.")
    build.add_argument("--overlap", type=int, default=200, help="Overlap between chunks.")
    build.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    build.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for embedding model (auto selects CUDA when available).",
    )

    query = subparsers.add_parser("query", help="Query a FAISS index.")
    query.add_argument("--index-path", required=True, help="Path to FAISS index file.")
    query.add_argument("--metadata-path", required=True, help="Path to metadata JSONL file.")
    query.add_argument("--query", required=True, help="Search query text.")
    query.add_argument("--model-name", default=DEFAULT_MODEL, help="Hugging Face model name.")
    query.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    query.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for embedding model (auto selects CUDA when available).",
    )

    build_titles = subparsers.add_parser(
        "build-titles",
        help="Build a FAISS index with title-only chunks plus body chunks.",
    )
    build_titles.add_argument("--parsed-dir", required=True, help="Directory with parsed JSON files.")
    build_titles.add_argument("--output-dir", required=True, help="Directory to store FAISS index.")
    build_titles.add_argument("--model-name", default=DEFAULT_MODEL, help="Hugging Face model name.")
    build_titles.add_argument("--max-chars", type=int, default=1200, help="Chunk size in characters.")
    build_titles.add_argument("--overlap", type=int, default=200, help="Overlap between chunks.")
    build_titles.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    build_titles.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for embedding model (auto selects CUDA when available).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        build_index(
            parsed_dir=args.parsed_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_chars=args.max_chars,
            overlap=args.overlap,
            batch_size=args.batch_size,
            device_preference=args.device,
        )
        return

    if args.command == "query":
        results = query_index(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            query=args.query,
            model_name=args.model_name,
            top_k=args.top_k,
            device_preference=args.device,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.command == "build-titles":
        build_index_with_titles(
            parsed_dir=args.parsed_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_chars=args.max_chars,
            overlap=args.overlap,
            batch_size=args.batch_size,
            device_preference=args.device,
        )
        return

    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
