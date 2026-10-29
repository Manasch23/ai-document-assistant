import httpx
import pytest

from app.config import Settings
from app.errors import AppError
from app.ollama import Ollama


def test_ollama_payload_and_extraction(monkeypatch):
    def post(self, url, **kwargs):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert kwargs["json"]["model"] == "llama3.2:3b"
        assert kwargs["json"]["stream"] is False
        return httpx.Response(200, json={"message": {"content": "30 days [S1]"}})

    monkeypatch.setattr(httpx.Client, "post", post)
    result = Ollama(Settings(_env_file=None)).answer(
        "Refund?", [{"label": "S1", "filename": "policy.txt", "page": None, "text": "30 days"}]
    )
    assert result == "30 days [S1]"


def test_ollama_not_running(monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx.Client, "post", fail)
    with pytest.raises(AppError, match="Could not connect to Ollama") as error:
        Ollama(Settings(_env_file=None)).answer("?", [])
    assert error.value.status == 503
