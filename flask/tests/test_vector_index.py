import os
import sys

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from search.vector_index import l2_normalize_rows


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
