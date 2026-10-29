import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeEmbeddings:
    """Deterministic topic vectors; no downloads or external calls in the test suite."""

    def encode(self, texts, query=False):
        vectors = np.asarray(
            [
                [text.lower().count("refund") + 0.01, text.lower().count("shipping") + 0.01, 0.01]
                for text in texts
            ],
            dtype="float32",
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeGrok:
    def answer(self, question, sources):
        return f"The retrieved passage says: {sources[0]['text']} [S1]"


@pytest.fixture
def client(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", xai_api_key="")
    app = create_app(settings, FakeEmbeddings(), FakeGrok())
    with TestClient(app) as client:
        yield client
