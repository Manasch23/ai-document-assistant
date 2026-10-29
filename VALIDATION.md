# Delivery verification

Verified on Windows with Python 3.12.6 on 22 October 2026 (Asia/Kolkata).

## Completed

- **35 automated tests passed.** These use the actual FastAPI, LangChain splitter,
  SQLite, PDF/DOCX parsers, and FAISS libraries. Embeddings and paid Grok calls are
  substituted in this suite for deterministic offline testing.
- Ruff code checks passed.
- Downloaded the real `BAAI/bge-small-en-v1.5` embedding model. Encoded three passages
  into 384-dimensional vectors, persisted them, and queried FAISS. A question about
  annual leave retrieved the vacation passage with a cosine similarity of 0.6947.
- Started the local HTTP server and opened the interface in the browser.
- Uploaded the example handbook through the HTTP API with the real embedding model:
  received HTTP 201 and two indexed passages. The browser displayed the document.
- Verified the suggested-question button fills the question field.
- Verified asking without an API key returns HTTP 503 and a setup message.
- Saved the browser preview in `docs/preview.png`.

This execution environment applies restrictive Windows permissions to Python-created
temporary directories. Dependency installation and validation used a workspace-only
runner to accommodate those permissions. That runner is not part of the delivered app.

## Not verified here

- **Live Grok answers:** no user API key or credits were supplied. The request format,
  response extraction, incomplete responses, auth failures, credit/rate errors, and
  timeouts were tested with mocked provider responses against the documented API.
- Docker build and Linux execution were not run. The included GitHub Actions workflow
  is configured to test on Windows and Linux after you push the repository.
- The project has not been published or pushed to a GitHub account.

Use the handbook questions in the README after adding your key for the final live
provider check. The project is intended for a local, single-user demonstration; the
README explains its deployment and document parsing limits.
