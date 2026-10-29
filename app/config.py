from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama is the no-cost default: it runs the answer model on this computer.
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    xai_api_key: SecretStr = SecretStr("")
    grok_model: str = "grok-4.7"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = Path("data")
    max_upload_bytes: int = 20 * 1024 * 1024
    max_characters: int = 1_000_000
    max_documents: int = 100
    max_chunks: int = 25_000
