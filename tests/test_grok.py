import json

import httpx
import pytest

from app.config import Settings
from app.errors import AppError
from app.grok import Grok


@pytest.mark.parametrize("status,expected", [(401, 502), (403, 502), (429, 429), (500, 502)])
def test_provider_errors(monkeypatch, status, expected):
    def post(self, url, **kwargs):
        return httpx.Response(status)

    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(AppError) as error:
        Grok(Settings(_env_file=None, xai_api_key="test-only")).answer("?", [])
    assert error.value.status == expected


def test_responses_api_payload_and_extraction(monkeypatch):
    def post(self, url, **kwargs):
        assert url == "https://api.x.ai/v1/responses"
        assert kwargs["json"]["store"] is False
        assert kwargs["headers"]["Authorization"] == "Bearer test-only"
        prompt = json.loads(kwargs["json"]["input"][1]["content"])
        assert prompt["document_excerpts"][0]["text"] == "Refund within 30 days."
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {"type": "reasoning", "content": []},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "30 days [S1]"}],
                    },
                ],
            },
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    result = Grok(Settings(_env_file=None, xai_api_key="test-only")).answer(
        "Refund?",
        [{"label": "S1", "filename": "policy.txt", "page": None, "text": "Refund within 30 days."}],
    )
    assert result == "30 days [S1]"


def test_timeout(monkeypatch):
    def post(self, url, **kwargs):
        raise httpx.ReadTimeout("timeout")

    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(AppError) as error:
        Grok(Settings(_env_file=None, xai_api_key="test-only")).answer("?", [])
    assert error.value.status == 504


@pytest.mark.parametrize("data", [{}, {"status": "incomplete"}, {"output": []}])
def test_empty_or_incomplete_responses(monkeypatch, data):
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: httpx.Response(200, json=data))
    with pytest.raises(AppError):
        Grok(Settings(_env_file=None, xai_api_key="test-only")).answer("?", [])
