import json

import httpx

from app.errors import AppError
from app.grok import SYSTEM


class Ollama:
    """Local Ollama chat client. It never sends document excerpts to a cloud provider."""

    def __init__(self, settings):
        self.settings = settings

    def answer(self, question: str, sources: list[dict]) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "document_excerpts": [
                                {key: source[key] for key in ("label", "filename", "page", "text")}
                                for source in sources
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        try:
            with httpx.Client(timeout=httpx.Timeout(180, connect=5)) as client:
                response = client.post(url, json=payload)
            if response.status_code == 404:
                raise AppError(
                    f"Ollama cannot find {self.settings.ollama_model}. Run: ollama pull "
                    f"{self.settings.ollama_model}",
                    503,
                )
            if response.status_code >= 400:
                raise AppError(
                    "Ollama could not generate an answer. Check the server terminal.", 502
                )
            text = response.json().get("message", {}).get("content", "").strip()
            if not text:
                raise AppError("Ollama returned an empty answer. Please retry.", 502)
            return text
        except AppError:
            raise
        except httpx.TimeoutException as exc:
            raise AppError("Ollama timed out. Try a shorter question.", 504) from exc
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            raise AppError(
                "Could not connect to Ollama. Install Ollama, run `ollama pull llama3.2:3b`, "
                "then restart this app.",
                503,
            ) from exc
