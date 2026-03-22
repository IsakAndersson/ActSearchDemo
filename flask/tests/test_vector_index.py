import os
import sys
import json

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from search import vector_index
from search.vector_index import build_index, get_document_text, l2_normalize_rows, resolve_text_source


def test_l2_normalize_rows_returns_unit_vectors():
    embeddings = np.array([[3.0, 4.0], [5.0, 12.0]], dtype="float32")

    normalized = l2_normalize_rows(embeddings)

    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, np.array([1.0, 1.0], dtype="float32"))


def test_l2_normalize_rows_keeps_zero_rows_zero():
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0]], dtype="float32")

    normalized = l2_normalize_rows(embeddings)

    assert np.array_equal(normalized[0], np.array([0.0, 0.0], dtype="float32"))
    assert np.allclose(normalized[1], np.array([1.0, 0.0], dtype="float32"))


def test_resolve_text_source_accepts_supported_fields():
    assert resolve_text_source("text") == "text"
    assert resolve_text_source("cleaned_text") == "cleaned_text"


def test_ensure_vector_dependencies_rejects_conflicting_faiss_installs(monkeypatch):
    monkeypatch.setattr(vector_index, "_FAISS_IMPORT_ERROR", None)
    monkeypatch.setattr(vector_index, "_TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(vector_index, "_TRANSFORMERS_IMPORT_ERROR", None)
    monkeypatch.setattr(vector_index, "_installed_faiss_distributions", lambda: ["faiss-cpu", "faiss-gpu"])

    with pytest.raises(RuntimeError, match="Conflicting FAISS installations detected"):
        vector_index.ensure_vector_dependencies()


def test_ensure_vector_dependencies_rejects_incomplete_faiss_module(monkeypatch):
    monkeypatch.setattr(vector_index, "_FAISS_IMPORT_ERROR", None)
    monkeypatch.setattr(vector_index, "_TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(vector_index, "_TRANSFORMERS_IMPORT_ERROR", None)
    monkeypatch.setattr(vector_index, "_installed_faiss_distributions", lambda: ["faiss-gpu"])

    class IncompleteFaiss:
        __file__ = None

    monkeypatch.setattr(vector_index, "faiss", IncompleteFaiss())

    with pytest.raises(RuntimeError, match="The imported FAISS module is incomplete"):
        vector_index.ensure_vector_dependencies()


def test_get_document_text_uses_requested_field_only():
    payload = {
        "text": "raw text",
        "cleaned_text": "clean text",
    }

    assert get_document_text(payload, "text") == "raw text"
    assert get_document_text(payload, "cleaned_text") == "clean text"


def test_get_document_text_returns_empty_string_when_selected_field_missing():
    payload = {"text": "raw text"}

    assert get_document_text(payload, "cleaned_text") == ""


def test_build_index_rejects_non_positive_batch_size(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_index, "resolve_device", lambda _: "cpu")
    monkeypatch.setattr(vector_index, "_get_cached_encoder", lambda **_: ("tokenizer", "model"))

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        build_index(
            parsed_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            model_name="demo-model",
            max_chars=100,
            overlap=0,
            batch_size=0,
            device_preference="cpu",
        )


def test_build_index_streams_embedding_batches(tmp_path, monkeypatch):
    parsed_dir = tmp_path / "parsed"
    output_dir = tmp_path / "out"
    parsed_dir.mkdir()

    documents = [
        {
            "path": str(parsed_dir / "doc1.json"),
            "text": "ignored",
            "metadata": {"title": "Doc 1"},
        }
    ]
    chunk_values = [f"chunk {i}" for i in range(40)]
    embed_batch_sizes = []

    monkeypatch.setattr(vector_index, "iter_parsed_documents", lambda _: documents)
    monkeypatch.setattr(vector_index, "chunk_text", lambda *args, **kwargs: chunk_values)
    monkeypatch.setattr(vector_index, "resolve_device", lambda _: "cpu")
    monkeypatch.setattr(vector_index, "_get_cached_encoder", lambda **_: ("tokenizer", "model"))
    monkeypatch.setattr(
        vector_index,
        "faiss",
        type(
            "FakeFaiss",
            (),
            {
                "IndexFlatIP": staticmethod(
                    lambda dim: type("FakeIndex", (), {"add": lambda self, embeddings: None})()
                ),
                "write_index": staticmethod(lambda index, path: None),
            },
        )(),
    )

    def fake_embed_texts(texts, tokenizer, model, device, batch_size, normalize=True):
        embed_batch_sizes.append(len(texts))
        return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setattr(vector_index, "embed_texts", fake_embed_texts)

    build_index(
        parsed_dir=str(parsed_dir),
        output_dir=str(output_dir),
        model_name="demo-model",
        max_chars=100,
        overlap=0,
        batch_size=1,
        device_preference="cpu",
    )

    metadata_path = output_dir / "docplus_metadata.jsonl"
    assert metadata_path.exists()
    assert embed_batch_sizes == [32, 8]
    assert sum(1 for _ in metadata_path.open("r", encoding="utf-8")) == 40
    first_record = json.loads(metadata_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_record["chunk_type"] == "section"
    assert first_record["preview_text"] == "chunk 0"
    assert first_record["metadata"]["section_heading"] == "Doc 1"
