import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np

from app.errors import AppError


class Store:
    """SQLite is the source of truth; FAISS is rebuilt from safe float32 vectors.

    One process only. A lock covers transactions and snapshots across FastAPI worker threads.
    No pickle files or user-supplied index files are deserialized.
    """

    def __init__(self, directory: Path, model: str, max_documents=100, max_chunks=25_000):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "documents.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.max_documents = max_documents
        self.max_chunks = max_chunks
        self.db.executescript("""
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, digest TEXT UNIQUE NOT NULL,
                size INTEGER NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_id TEXT NOT NULL
                REFERENCES documents(id) ON DELETE CASCADE,
                page INTEGER, text TEXT NOT NULL, vector BLOB NOT NULL
            );
        """)
        old = self.db.execute("SELECT value FROM meta WHERE key='model'").fetchone()
        if old and old[0] != model:
            self.db.close()
            raise RuntimeError(
                "EMBEDDING_MODEL changed. Use a new DATA_DIR and re-upload documents."
            )
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('model', ?)", (model,))

    def close(self):
        self.db.close()

    def list_documents(self) -> list[dict]:
        with self.lock:
            return [
                dict(row)
                for row in self.db.execute("""
                SELECT d.id, d.name, d.size, d.created_at, COUNT(c.id) AS chunks
                FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                GROUP BY d.id ORDER BY d.created_at DESC
            """)
            ]

    def duplicate(self, raw: bytes):
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            row = self.db.execute("SELECT id FROM documents WHERE digest=?", (digest,)).fetchone()
            return (
                next((d for d in self.list_documents() if d["id"] == row[0]), None) if row else None
            )

    def add(self, name: str, raw: bytes, chunks: list[dict], vectors: np.ndarray):
        with self.lock:
            existing = self.duplicate(raw)
            if existing:
                return existing, True
            if len(self.list_documents()) >= self.max_documents:
                raise AppError("Document limit reached. Delete a document first.", 409)
            count = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count + len(chunks) > self.max_chunks:
                raise AppError(
                    "Index capacity reached. Delete documents or upload a smaller file.", 409
                )
            if len(vectors) != len(chunks) or vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise AppError("Invalid embedding output.", 503)
            dimension = self.db.execute("SELECT vector FROM chunks LIMIT 1").fetchone()
            if dimension and len(dimension[0]) != vectors.shape[1] * 4:
                raise AppError("Embedding dimensions changed. Use a new DATA_DIR.", 503)
            identifier = str(uuid4())
            with self.db:
                self.db.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (
                        identifier,
                        name,
                        hashlib.sha256(raw).hexdigest(),
                        len(raw),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self.db.executemany(
                    "INSERT INTO chunks(document_id,page,text,vector) VALUES (?,?,?,?)",
                    [
                        (identifier, c["page"], c["text"], np.asarray(v, dtype="float32").tobytes())
                        for c, v in zip(chunks, vectors)
                    ],
                )
            return next(d for d in self.list_documents() if d["id"] == identifier), False

    def delete(self, identifier: str):
        with self.lock, self.db:
            result = self.db.execute("DELETE FROM documents WHERE id=?", (identifier,))
            if not result.rowcount:
                raise AppError("Document not found.", 404)

    def search(self, query: np.ndarray, identifiers: list[str], top_k: int):
        with self.lock:
            known = {d["id"] for d in self.list_documents()}
            if any(identifier not in known for identifier in identifiers):
                raise AppError(
                    "A selected document no longer exists. Refresh your document list.", 404
                )
            where = "WHERE c.document_id IN (" + ",".join("?" for _ in identifiers) + ")"
            rows = self.db.execute(
                "SELECT c.*, d.name FROM chunks c JOIN documents d ON d.id=c.document_id "
                + (where if identifiers else "")
                + " ORDER BY c.id",
                identifiers,
            ).fetchall()
        if not rows:
            raise AppError("Upload at least one document before asking a question.", 409)
        matrix = np.stack([np.frombuffer(r["vector"], dtype="float32") for r in rows])
        if matrix.shape[1] != query.shape[1]:
            raise AppError("Query and document embedding dimensions differ.", 503)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(np.ascontiguousarray(matrix))
        scores, positions = index.search(query, min(top_k, len(rows)))
        return [
            {
                "label": f"S{i + 1}",
                "document_id": rows[p]["document_id"],
                "filename": rows[p]["name"],
                "page": rows[p]["page"],
                "text": rows[p]["text"],
                "score": round(float(score), 4),
            }
            for i, (score, p) in enumerate(zip(scores[0], positions[0]))
        ]
