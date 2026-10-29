import threading

import numpy as np
from fastembed import TextEmbedding

from app.errors import AppError


class LocalEmbeddings:
    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        try:
            with self._lock:
                if self._model is None:
                    self._model = TextEmbedding(
                        model_name=self.model_name, cache_dir=self.cache_dir, threads=2
                    )
                stream = (
                    self._model.query_embed(texts) if query else self._model.passage_embed(texts)
                )
                vectors = np.asarray(list(stream), dtype="float32")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return np.ascontiguousarray(vectors / np.maximum(norms, 1e-12))
        except Exception as exc:
            raise AppError(
                "Embedding model unavailable. First use needs internet to download the model. "
                "Check your connection and EMBEDDING_MODEL setting, then retry.",
                503,
            ) from exc
