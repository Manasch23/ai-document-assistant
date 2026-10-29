import numpy as np
import pytest

from app.errors import AppError
from app.store import Store


def test_chunk_capacity_and_model_mismatch(tmp_path):
    store = Store(tmp_path, "model-1", max_chunks=1)
    vector = np.asarray([[1, 0, 0]], dtype="float32")
    store.add("one.txt", b"one", [{"text": "one", "page": None}], vector)
    with pytest.raises(AppError, match="capacity"):
        store.add("two.txt", b"two", [{"text": "two", "page": None}], vector)
    store.close()
    with pytest.raises(RuntimeError, match="EMBEDDING_MODEL changed"):
        Store(tmp_path, "model-2")


def test_nan_vectors_rejected_atomically(tmp_path):
    store = Store(tmp_path, "model-1")
    with pytest.raises(AppError, match="Invalid embedding"):
        store.add(
            "one.txt",
            b"one",
            [{"text": "one", "page": None}],
            np.asarray([[float("nan"), 0, 0]], dtype="float32"),
        )
    assert store.list_documents() == []
    store.close()
