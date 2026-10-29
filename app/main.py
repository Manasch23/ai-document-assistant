import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.embeddings import LocalEmbeddings
from app.errors import AppError
from app.grok import Grok
from app.ingest import extract_chunks
from app.ollama import Ollama
from app.store import Store

STATIC = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)
    top_k: int = Field(default=5, ge=1, le=8)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Enter a question.")
        return value.strip()


class BodyLimit:
    """Bound request bodies even without Content-Length (including multipart overhead)."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = self.limit + 1
        if declared > self.limit:
            return await JSONResponse({"detail": "Upload exceeds the 20 MB limit."}, 413)(
                scope, receive, send
            )
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                # FastAPI handles HTTPException during multipart parsing correctly.
                from starlette.exceptions import HTTPException

                raise HTTPException(413, "Upload exceeds the 20 MB limit.")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings=None, embedder=None, grok=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(
            settings.data_dir, settings.embedding_model, settings.max_documents, settings.max_chunks
        )
        app.state.embedder = embedder or LocalEmbeddings(
            settings.embedding_model, str(settings.data_dir / "models")
        )
        if grok is not None:
            app.state.answerer = grok
        elif settings.llm_provider.lower() == "ollama":
            app.state.answerer = Ollama(settings)
        elif settings.llm_provider.lower() == "grok":
            app.state.answerer = Grok(settings)
        else:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'grok'.")
        yield
        app.state.store.close()

    app = FastAPI(title="AI Document Assistant", version="1.0.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=settings.max_upload_bytes + 64 * 1024)

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
            if not request.url.path.startswith(("/docs", "/redoc"))
            else "frame-ancestors 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.exception("Unexpected application error")
        return JSONResponse(
            {"detail": "An unexpected error occurred. Check the server terminal."}, 500
        )

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        provider = settings.llm_provider.lower()
        grok_configured = bool(settings.xai_api_key.get_secret_value().strip())
        return {
            "status": "ok",
            "provider": provider,
            "provider_configured": provider == "ollama" or grok_configured,
            "grok_configured": grok_configured,
            "model": settings.ollama_model if provider == "ollama" else settings.grok_model,
            "embedding_model": settings.embedding_model,
        }

    @app.get("/api/documents")
    def documents():
        return {"documents": app.state.store.list_documents()}

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile = File(...)):
        try:
            raw = await file.read(settings.max_upload_bytes + 1)
        finally:
            await file.close()
        if len(raw) > settings.max_upload_bytes:
            raise AppError("Upload exceeds the 20 MB limit.", 413)
        name = (file.filename or "document").replace("\\", "/").rsplit("/", 1)[-1]
        name = "".join(c for c in name if c.isprintable())[:200]

        def process():
            existing = app.state.store.duplicate(raw)
            if existing:
                return {"document": existing, "duplicate": True}
            chunks = extract_chunks(name, raw, settings.max_characters)
            vectors = app.state.embedder.encode([c["text"] for c in chunks])
            document, duplicate = app.state.store.add(name, raw, chunks, vectors)
            return {"document": document, "duplicate": duplicate}

        return await run_in_threadpool(process)

    @app.delete("/api/documents/{document_id}", status_code=204)
    def delete(document_id: UUID):
        app.state.store.delete(str(document_id))

    @app.post("/api/ask")
    def ask(body: Question):
        if not app.state.store.list_documents():
            raise AppError("Upload at least one document before asking a question.", 409)
        needs_grok_key = (
            settings.llm_provider.lower() == "grok"
            and not settings.xai_api_key.get_secret_value().strip()
            and grok is None
        )
        if needs_grok_key:
            raise AppError(
                "Add XAI_API_KEY to .env and restart the server to enable Grok answers.", 503
            )
        vector = app.state.embedder.encode([body.question], query=True)
        sources = app.state.store.search(vector, [str(i) for i in body.document_ids], body.top_k)
        answer = app.state.answerer.answer(body.question, sources)
        labels = set(re.findall(r"\[(S\d+)\]", answer))
        valid = {s["label"] for s in sources}
        invalid = labels - valid
        if invalid:
            raise AppError(
                "The answer model returned invalid source references. Please retry.", 502
            )
        for source in sources:
            source["cited"] = source["label"] in labels
        return {
            "answer": answer,
            "sources": sources,
            "model": (
                settings.ollama_model
                if settings.llm_provider.lower() == "ollama"
                else settings.grok_model
            ),
            "citation_warning": not bool(labels),
        }

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
