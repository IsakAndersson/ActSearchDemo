"""Build and query a FAISS vector index for scraped Docplus content."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from document_structure import get_document_sections

try:
    import faiss
except ImportError as exc:
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

try:
    import torch
except ImportError as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    AutoModel = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


@dataclass(frozen=True)
class VectorModelProfile:
    key: str
    model_name: str
    chunk_size: int
    chunk_overlap: int
    include_title_chunk: bool


VECTOR_MODEL_PROFILES: Dict[str, VectorModelProfile] = {
    "swedish_bert": VectorModelProfile(
        key="swedish_bert",
        model_name="KBLab/bert-base-swedish-cased",
        chunk_size=250,
        chunk_overlap=50,
        include_title_chunk=True,
    ),
    "e5_large_instruct": VectorModelProfile(
        key="e5_large_instruct",
        model_name="intfloat/multilingual-e5-large-instruct",
        chunk_size=250,
        chunk_overlap=50,
        include_title_chunk=True,
    ),
}
DEFAULT_MODEL_PROFILE_KEY = "swedish_bert"
DEFAULT_MODEL = VECTOR_MODEL_PROFILES[DEFAULT_MODEL_PROFILE_KEY].model_name
DEFAULT_CHUNK_SIZE = VECTOR_MODEL_PROFILES[DEFAULT_MODEL_PROFILE_KEY].chunk_size
DEFAULT_CHUNK_OVERLAP = VECTOR_MODEL_PROFILES[DEFAULT_MODEL_PROFILE_KEY].chunk_overlap
LOG = logging.getLogger(__name__)
_MODEL_CACHE: Dict[tuple[str, str], tuple[AutoTokenizer, AutoModel]] = {}
_INDEX_CACHE: Dict[str, faiss.Index] = {}
_METADATA_CACHE: Dict[str, List[dict]] = {}
_CACHE_LOCK = threading.Lock()
TEXT_SOURCE_KEYS = ("text",)


@dataclass
class ChunkRecord:
    chunk_id: int
    source_path: str
    text: str
    metadata: dict
    chunk_type: str = "body"
    preview_text: str = ""


@dataclass
class IndexBuildState:
    records: List[ChunkRecord]
    texts: List[str]
    next_chunk_id: int = 0


def resolve_text_source(text_source: Optional[str]) -> str:
    candidate = (text_source or "text").strip()
    if candidate in TEXT_SOURCE_KEYS:
        return candidate
    choices = ", ".join(TEXT_SOURCE_KEYS)
    raise ValueError(f"Unknown text source '{candidate}'. Valid options: {choices}")


def get_document_text(payload: dict, text_source: str) -> str:
    resolved_text_source = resolve_text_source(text_source)
    value = payload.get(resolved_text_source)
    return value if isinstance(value, str) else ""


def vector_dependencies_available() -> bool:
    return all(
        error is None
        for error in (_FAISS_IMPORT_ERROR, _TORCH_IMPORT_ERROR, _TRANSFORMERS_IMPORT_ERROR)
    )


def _installed_faiss_distributions() -> List[str]:
    installed: List[str] = []
    for package_name in ("faiss-cpu", "faiss-gpu"):
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        installed.append(package_name)
    return installed


def _validate_faiss_module() -> None:
    required_symbols = ("IndexFlatIP", "read_index", "write_index")
    missing_symbols = [name for name in required_symbols if not hasattr(faiss, name)]
    if missing_symbols:
        module_path = getattr(faiss, "__file__", None) or "<namespace package>"
        raise RuntimeError(
            "The imported FAISS module is incomplete and cannot be used. "
            f"Missing symbols: {', '.join(missing_symbols)}. "
            f"Imported module path: {module_path}. "
            "Reinstall FAISS so the Python bindings are installed correctly."
        )


def ensure_vector_dependencies() -> None:
    if vector_dependencies_available():
        installed_faiss = _installed_faiss_distributions()
        if len(installed_faiss) > 1:
            raise RuntimeError(
                "Conflicting FAISS installations detected: "
                f"{', '.join(installed_faiss)}. "
                "Uninstall one of them so only a single FAISS build remains in the environment."
            )
        _validate_faiss_module()
        return

    missing: List[str] = []
    if _FAISS_IMPORT_ERROR is not None:
        missing.append("faiss")
    if _TORCH_IMPORT_ERROR is not None:
        missing.append("torch")
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        missing.append("transformers")

    raise RuntimeError(
        "Vector search dependencies are unavailable. "
        f"Missing: {', '.join(missing)}. "
        "Install the optional vector dependencies to enable FAISS-based search."
    )


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


def _resolve_profile(profile_key: Optional[str]) -> Optional[VectorModelProfile]:
    if not profile_key:
        return None
    profile = VECTOR_MODEL_PROFILES.get(profile_key)
    if profile is None:
        choices = ", ".join(sorted(VECTOR_MODEL_PROFILES.keys()))
        raise ValueError(f"Unknown profile '{profile_key}'. Valid options: {choices}")
    return profile


def resolve_model_name(
    profile_key: Optional[str],
    model_name: Optional[str],
) -> str:
    profile = _resolve_profile(profile_key)
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    if profile:
        return profile.model_name
    return DEFAULT_MODEL


def resolve_chunk_settings(
    profile_key: Optional[str],
    max_chars: Optional[int],
    overlap: Optional[int],
) -> tuple[int, int]:
    profile = _resolve_profile(profile_key)
    resolved_max_chars = (
        max_chars if max_chars is not None else profile.chunk_size if profile else DEFAULT_CHUNK_SIZE
    )
    resolved_overlap = (
        overlap if overlap is not None else profile.chunk_overlap if profile else DEFAULT_CHUNK_OVERLAP
    )
    return resolved_max_chars, resolved_overlap


def resolve_include_title_chunk(profile_key: Optional[str], include_title_chunk: bool) -> bool:
    if include_title_chunk:
        return True
    selected_profile_key = profile_key or DEFAULT_MODEL_PROFILE_KEY
    profile = _resolve_profile(selected_profile_key)
    if profile is None:
        return False
    return profile.include_title_chunk


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


def _uses_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _with_query_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.lower().startswith("query:"):
        return stripped
    return f"query: {stripped}"


def _with_passage_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.lower().startswith("passage:"):
        return stripped
    return f"passage: {stripped}"


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def l2_normalize_rows(embeddings: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return row-wise L2-normalized embeddings while keeping zero rows stable."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.clip(norms, epsilon, None)
    normalized = embeddings / safe_norms
    normalized[norms.squeeze(axis=1) == 0.0] = 0.0
    return normalized.astype("float32", copy=False)


def embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    normalize: bool = True,
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
    stacked = np.vstack(embeddings).astype("float32")
    if normalize:
        return l2_normalize_rows(stacked)
    return stacked


def _flush_index_batch(
    *,
    state: IndexBuildState,
    index: faiss.Index | None,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
) -> faiss.Index:
    if not state.texts:
        if index is None:
            raise ValueError("Cannot flush empty index batch without an initialized FAISS index.")
        return index

    embeddings = embed_texts(
        state.texts,
        tokenizer,
        model,
        device,
        batch_size,
        normalize=True,
    )
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    state.texts.clear()
    return index


def resolve_device(device_preference: str) -> torch.device:
    ensure_vector_dependencies()
    if device_preference == "cpu":
        return torch.device("cpu")
    if device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_local_encoder(model_name: str) -> tuple[AutoTokenizer, AutoModel]:
    ensure_vector_dependencies()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        return tokenizer, model
    except OSError as exc:
        raise RuntimeError(
            "Model files were not found in the local Hugging Face cache. "
            f"Pre-download '{model_name}' before running with offline loading."
        ) from exc


def _get_cached_encoder(model_name: str, device: torch.device) -> tuple[AutoTokenizer, AutoModel]:
    cache_key = (model_name, str(device))

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    LOG.info("Loading encoder model '%s' on %s...", model_name, device)
    tokenizer, model = load_local_encoder(model_name)
    model.to(device)
    model.eval()

    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _get_cached_index(index_path: str) -> faiss.Index:
    normalized_path = os.path.abspath(index_path)

    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(normalized_path)
    if cached is not None:
        return cached

    LOG.info("Reading FAISS index: %s", normalized_path)
    index = faiss.read_index(normalized_path)
    with _CACHE_LOCK:
        _INDEX_CACHE[normalized_path] = index
    return index


def _get_cached_metadata(metadata_path: str) -> List[dict]:
    normalized_path = os.path.abspath(metadata_path)

    with _CACHE_LOCK:
        cached = _METADATA_CACHE.get(normalized_path)
    if cached is not None:
        return cached

    LOG.info("Loading metadata JSONL: %s", normalized_path)
    loaded_metadata: List[dict] = []
    with open(normalized_path, "r", encoding="utf-8") as handle:
        for line in handle:
            loaded_metadata.append(json.loads(line))

    with _CACHE_LOCK:
        _METADATA_CACHE[normalized_path] = loaded_metadata
    return loaded_metadata


def clear_runtime_caches() -> None:
    """Release cached models, indexes, and metadata between long-running batches."""
    with _CACHE_LOCK:
        cached_models = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()
        _INDEX_CACHE.clear()
        _METADATA_CACHE.clear()

    for _, model in cached_models:
        try:
            model.to("cpu")
        except Exception:
            continue

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _append_chunk(
    *,
    state: IndexBuildState,
    source_path: str,
    text: str,
    metadata: dict,
    chunk_type: str,
    use_e5_prefixes: bool,
    preview_text: str = "",
) -> None:
    state.records.append(
        ChunkRecord(
            chunk_id=state.next_chunk_id,
            source_path=source_path,
            text=text,
            metadata=metadata,
            chunk_type=chunk_type,
            preview_text=preview_text or text,
        )
    )
    state.texts.append(_with_passage_prefix(text) if use_e5_prefixes else text)
    state.next_chunk_id += 1


def build_index(
    parsed_dir: str,
    output_dir: str,
    model_name: str,
    max_chars: int,
    overlap: int,
    batch_size: int,
    device_preference: str,
    include_title_chunk: bool = False,
    text_source: str = "text",
) -> None:
    resolved_text_source = resolve_text_source(text_source)
    os.makedirs(output_dir, exist_ok=True)
    device = resolve_device(device_preference)
    tokenizer, model = _get_cached_encoder(model_name=model_name, device=device)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    state = IndexBuildState(records=[], texts=[])
    use_e5_prefixes = _uses_e5_prefixes(model_name)
    index: faiss.Index | None = None
    flush_threshold = max(batch_size * 32, batch_size)
    for payload in iter_parsed_documents(parsed_dir):
        metadata = payload.get("metadata", {}) or {}
        title = extract_title(payload)

        if include_title_chunk and title:
            title_metadata = {**metadata, "title": title}
            _append_chunk(
                state=state,
                source_path=payload["path"],
                text=title,
                metadata=title_metadata,
                chunk_type="title",
                use_e5_prefixes=use_e5_prefixes,
                preview_text=title,
            )

        body_metadata = {**metadata, "title": title} if title else metadata
        sections = get_document_sections(payload, fallback_title=title)
        if sections:
            for section in sections:
                section_text = section.get("text")
                heading = str(section.get("heading") or title or "").strip()
                section_metadata = {
                    **body_metadata,
                    "section_heading": heading,
                    "section_index": section.get("index", 0),
                    "section_level": section.get("level", 1),
                    "section_page": section.get("page"),
                    "section_path": section.get("path") or [heading],
                    "section_path_text": section.get("path_text") or heading,
                    "section_title": heading,
                    "section_text": section_text,
                }
                if heading:
                    title_chunk_text = f"{title}\n\n{heading}" if title and heading != title else heading
                    _append_chunk(
                        state=state,
                        source_path=payload["path"],
                        text=title_chunk_text,
                        metadata=section_metadata,
                        chunk_type="section_title",
                        use_e5_prefixes=use_e5_prefixes,
                        preview_text=heading,
                    )
                    if len(state.texts) >= flush_threshold:
                        index = _flush_index_batch(
                            state=state,
                            index=index,
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            batch_size=batch_size,
                        )
                if not isinstance(section_text, str) or not section_text.strip():
                    continue
                for chunk_index, preview_text in enumerate(chunk_text(section_text, max_chars=max_chars, overlap=overlap)):
                    chunk_text_value = preview_text
                    if heading and chunk_index == 0:
                        chunk_text_value = f"{heading}\n\n{preview_text}"
                    _append_chunk(
                        state=state,
                        source_path=payload["path"],
                        text=chunk_text_value,
                        metadata=section_metadata,
                        chunk_type="section",
                        use_e5_prefixes=use_e5_prefixes,
                        preview_text=preview_text,
                    )
                    if len(state.texts) >= flush_threshold:
                        index = _flush_index_batch(
                            state=state,
                            index=index,
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            batch_size=batch_size,
                        )
    if not state.records:
        raise ValueError("No parsed documents with text found to index.")

    index = _flush_index_batch(
        state=state,
        index=index,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    faiss_path = os.path.join(output_dir, "docplus.faiss")
    faiss.write_index(index, faiss_path)

    metadata_path = os.path.join(output_dir, "docplus_metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for record in state.records:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": record.chunk_id,
                        "source_path": record.source_path,
                        "text": record.text,
                        "metadata": record.metadata,
                        "chunk_type": record.chunk_type,
                        "preview_text": record.preview_text,
                        "section_text": record.metadata.get("section_text"),
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
    device = resolve_device(device_preference)
    tokenizer, model = _get_cached_encoder(model_name=model_name, device=device)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    state = IndexBuildState(records=[], texts=[])
    use_e5_prefixes = _uses_e5_prefixes(model_name)
    index: faiss.Index | None = None
    flush_threshold = max(batch_size * 32, batch_size)
    for payload in iter_parsed_documents(parsed_dir):
        metadata = payload.get("metadata", {}) or {}
        title = extract_title(payload)
        if title:
            title_metadata = {**metadata, "title": title}
            _append_chunk(
                state=state,
                source_path=payload["path"],
                text=title,
                metadata=title_metadata,
                chunk_type="title",
                use_e5_prefixes=use_e5_prefixes,
                preview_text=title,
            )
            if len(state.texts) >= flush_threshold:
                index = _flush_index_batch(
                    state=state,
                    index=index,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                )

    if not state.records:
        raise ValueError("No parsed documents with title metadata found to index.")

    index = _flush_index_batch(
        state=state,
        index=index,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    faiss_path = os.path.join(output_dir, "docplus_titles.faiss")
    faiss.write_index(index, faiss_path)

    metadata_path = os.path.join(output_dir, "docplus_titles_metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        for record in state.records:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": record.chunk_id,
                        "source_path": record.source_path,
                        "text": record.text,
                        "metadata": record.metadata,
                        "chunk_type": record.chunk_type,
                        "preview_text": record.preview_text,
                        "section_text": record.metadata.get("section_text"),
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
    device = resolve_device(device_preference)
    tokenizer, model = _get_cached_encoder(model_name=model_name, device=device)

    LOG.info("Embedding query...")
    query_text = _with_query_prefix(query) if _uses_e5_prefixes(model_name) else query
    # Cosine similarity in FAISS: L2-normalize query vectors and use inner product index.
    query_embedding = embed_texts([query_text], tokenizer, model, device, batch_size=1, normalize=True)
    index = _get_cached_index(index_path)
    scores, indices = index.search(query_embedding, top_k)
    metadata = _get_cached_metadata(metadata_path)

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
            "preview_text": entry.get("preview_text", entry["text"]),
            "section_heading": entry.get("metadata", {}).get("section_heading"),
            "section_index": entry.get("metadata", {}).get("section_index"),
            "section_level": entry.get("metadata", {}).get("section_level"),
            "section_path": entry.get("metadata", {}).get("section_path"),
            "section_path_text": entry.get("metadata", {}).get("section_path_text"),
            "section_text": entry.get("section_text") or entry.get("metadata", {}).get("section_text"),
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
    build.add_argument(
        "--profile",
        choices=tuple(sorted(VECTOR_MODEL_PROFILES.keys())),
        help="Optional named model profile (model + chunking defaults).",
    )
    build.add_argument("--model-name", help="Hugging Face model name.")
    build.add_argument("--max-chars", type=int, help="Chunk size in characters.")
    build.add_argument("--overlap", type=int, help="Overlap between chunks.")
    build.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    build.add_argument(
        "--text-source",
        choices=TEXT_SOURCE_KEYS,
        default="text",
        help="Parsed JSON field used for section text chunks.",
    )
    build.add_argument(
        "--include-title-chunk",
        action="store_true",
        help="Include one extra title chunk per document in addition to section chunks.",
    )
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
    query.add_argument(
        "--profile",
        choices=tuple(sorted(VECTOR_MODEL_PROFILES.keys())),
        help="Optional named model profile (model default).",
    )
    query.add_argument("--model-name", help="Hugging Face model name.")
    query.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    query.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for embedding model (auto selects CUDA when available).",
    )

    build_titles = subparsers.add_parser(
        "build-titles",
        help="Build a FAISS index with title-only chunks from parsed metadata.",
    )
    build_titles.add_argument("--parsed-dir", required=True, help="Directory with parsed JSON files.")
    build_titles.add_argument("--output-dir", required=True, help="Directory to store FAISS index.")
    build_titles.add_argument(
        "--profile",
        choices=tuple(sorted(VECTOR_MODEL_PROFILES.keys())),
        help="Optional named model profile (model + chunking defaults).",
    )
    build_titles.add_argument("--model-name", help="Hugging Face model name.")
    build_titles.add_argument("--max-chars", type=int, help="Chunk size in characters.")
    build_titles.add_argument("--overlap", type=int, help="Overlap between chunks.")
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
        model_name = resolve_model_name(profile_key=args.profile, model_name=args.model_name)
        max_chars, overlap = resolve_chunk_settings(
            profile_key=args.profile,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
        include_title_chunk = resolve_include_title_chunk(
            profile_key=args.profile,
            include_title_chunk=args.include_title_chunk,
        )
        build_index(
            parsed_dir=args.parsed_dir,
            output_dir=args.output_dir,
            model_name=model_name,
            max_chars=max_chars,
            overlap=overlap,
            batch_size=args.batch_size,
            device_preference=args.device,
            include_title_chunk=include_title_chunk,
            text_source=args.text_source,
        )
        return

    if args.command == "query":
        model_name = resolve_model_name(profile_key=args.profile, model_name=args.model_name)
        results = query_index(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            query=args.query,
            model_name=model_name,
            top_k=args.top_k,
            device_preference=args.device,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.command == "build-titles":
        model_name = resolve_model_name(profile_key=args.profile, model_name=args.model_name)
        max_chars, overlap = resolve_chunk_settings(
            profile_key=args.profile,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
        build_index_with_titles(
            parsed_dir=args.parsed_dir,
            output_dir=args.output_dir,
            model_name=model_name,
            max_chars=max_chars,
            overlap=overlap,
            batch_size=args.batch_size,
            device_preference=args.device,
        )
        return

    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
