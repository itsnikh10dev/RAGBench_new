# RAGBench

Intelligent document Q&A over your PDFs, powered by Gemini + FAISS.

RAGBench was migrated from a single-file Streamlit app to a FastAPI backend
with a vanilla HTML/CSS/JS frontend. **The RAG pipeline itself is unchanged**
— PyPDF2 → `RecursiveCharacterTextSplitter` → Gemini embeddings
(`models/gemini-embedding-001`) → FAISS → similarity search →
Gemini 3.6 Flash (`temperature=0.2`).

## Project structure

```
RAGBench_project/
├── backend/
│   ├── __init__.py
│   ├── main.py          FastAPI app: /api/* routes + serves the frontend
│   └── rag_engine.py     The RAG pipeline (ported from app.py, UI-free)
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── uploads/               Uploaded PDFs land here
├── faiss_index/           Persisted FAISS index
├── .env                    GOOGLE_API_KEY=... (create this yourself, gitignored)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup (Windows / PowerShell)

From `D:\RAGBench_project`, with your existing virtual environment:

```powershell
(.venv) PS D:\RAGBench_project> pip install -r requirements.txt
```

Create `.env` in the project root (this file is already in `.gitignore`,
never commit it):

```
GOOGLE_API_KEY=your_google_api_key
```

## Run

```powershell
(.venv) PS D:\RAGBench_project> uvicorn backend.main:app --reload
```

Then open:

```
http://127.0.0.1:8000
```

FastAPI serves both the API and the frontend, so this is the only server you
need to run — the old `python -m streamlit run app.py` command is no longer
used.

## Using the app

1. **Documents** sidebar → drop or browse one or more PDF files.
2. Click **Process Documents**. This uploads the files, extracts text with
   PyPDF2, splits it with `RecursiveCharacterTextSplitter`, embeds it with
   `models/gemini-embedding-001`, and rebuilds the FAISS index at
   `faiss_index/`.
3. Once the index shows **Ready**, ask a question in the main panel and
   press Enter or click **Ask RAGBench**.
4. Expand **Retrieved sources** under any answer to see the exact chunks
   that were retrieved and passed to Gemini as context.

If a question can't be answered from the retrieved context, the model
responds with exactly: `Answer is not available in the context.`

## API reference

| Method | Route          | Purpose                                            |
|--------|----------------|-----------------------------------------------------|
| GET    | `/api/status`  | API-key / FAISS-index readiness, model names        |
| POST   | `/api/upload`  | Multipart upload of one or more PDFs                |
| POST   | `/api/process` | Runs the extraction → chunking → embedding → FAISS pipeline on everything in `uploads/` |
| POST   | `/api/ask`     | `{ "question": "..." }` → similarity search + Gemini answer + sources |

The `GOOGLE_API_KEY` never leaves the backend — it is read from `.env` via
`python-dotenv` and is never sent to, stored in, or displayed by the
frontend.

## Notes

- Re-processing documents rebuilds the FAISS index from scratch (the same
  behavior as the original Streamlit app).
- Only `.pdf` files are accepted by `/api/upload`.
- Uploaded files and the FAISS index are excluded from Git via
  `.gitignore`.
