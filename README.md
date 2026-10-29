# AI Document Assistant

A simple web app where you can upload a document and ask questions about it.

The app finds useful parts of your document and shows the source text with each answer.

## What it can do

- Upload PDF, DOCX, TXT, and Markdown files
- Ask questions about uploaded documents
- Show the source passages used for an answer
- Keep documents on your computer
- Delete documents when they are no longer needed

## Tools used

- Python
- FastAPI
- SQLite
- FAISS
- Ollama
- Grok API (optional)

## Option 1: Run for free with Ollama

1. Install Python 3.12 and Ollama.
2. Open PowerShell in the project folder.
3. Run:

```powershell
.\start-free.ps1
```

4. Open this address in your browser:

```text
http://127.0.0.1:8000
```

This option runs the model on your own computer. You do not need an API key.

## Option 2: Run with Grok

You can also use the Grok API for answers instead of Ollama.

1. Create an API key from [xAI Console](https://console.x.ai/).
2. In the project folder, copy the example settings file:

```powershell
Copy-Item .env.example .env
```

3. Open `.env` and change it to this:

```text
LLM_PROVIDER=grok
XAI_API_KEY=your_grok_api_key_here
GROK_MODEL=grok-4.7
```

4. Start the project:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

5. Open `http://127.0.0.1:8000` in your browser.

Grok may require paid API credits. Never upload your `.env` file or API key to GitHub.

## How to use it

1. Upload a document from the left side of the page.
2. Wait for the document to finish processing.
3. Type a question about the document.
4. Read the answer and open the source passages to check it.

## Note

The first Ollama run may take a little longer because it downloads the local model needed for document search and answers.

## Author

Manas
