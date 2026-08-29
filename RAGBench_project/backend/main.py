"""
backend/main.py

FastAPI application for RAGBench.

Serves:
  - the JSON API under /api/*
  - the static HTML/CSS/JS frontend under /

Run with (from D:\\RAGBench_project, inside the venv):

    uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000
"""

import os
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import rag_engine as rag

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragbench.main")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSION = ".pdf"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file, sane guardrail

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(title="RAGBench API", version="1.0.0")


class AskRequest(BaseModel):
    question: str


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a raw Python traceback to the client."""
    logger.exception("Unhandled server error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Unable to contact the RAG backend."},
    )


# ============================================================
# STATUS
# ============================================================
@app.get("/api/status")
def get_status():
    return {
        "api_configured": rag.is_api_configured(),
        "faiss_ready": rag.is_index_ready(),
        "embedding_model": rag.EMBEDDING_MODEL,
        "chat_model": rag.CHAT_MODEL,
    }


# ============================================================
# UPLOAD
# ============================================================
@app.post("/api/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files selected.")

    uploaded = []

    for f in files:
        filename = f.filename or ""
        if not filename.lower().endswith(ALLOWED_EXTENSION):
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' is not a PDF file. Only .pdf files are accepted.",
            )

        # Never trust the client-supplied filename directly as a path.
        safe_name = os.path.basename(filename)
        dest_path = UPLOAD_DIR / safe_name

        try:
            content = await f.read()
        except Exception:
            logger.exception("Failed reading upload '%s'", safe_name)
            raise HTTPException(status_code=400, detail=f"Could not read '{safe_name}'.")

        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"'{safe_name}' is empty.")

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"'{safe_name}' exceeds the 50 MB upload limit.",
            )

        try:
            with open(dest_path, "wb") as out:
                out.write(content)
        except Exception:
            logger.exception("Failed saving upload '%s'", safe_name)
            raise HTTPException(status_code=500, detail=f"Could not save '{safe_name}'.")

        uploaded.append({"name": safe_name, "size": dest_path.stat().st_size})

    return {"success": True, "files": uploaded}


# ============================================================
# PROCESS
# ============================================================
@app.post("/api/process")
def process_documents():
    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(
            status_code=400,
            detail="No PDF files found. Please upload documents first.",
        )

    text, errors = rag.get_pdf_text([str(p) for p in pdf_files])

    if not text.strip():
        detail = "Could not extract text from the uploaded PDF."
        if errors:
            detail += " " + " ".join(errors)
        raise HTTPException(status_code=422, detail=detail)

    chunks = rag.get_text_chunks(text)
    if not chunks:
        raise HTTPException(
            status_code=422, detail="No text chunks were created from the PDF(s)."
        )

    try:
        rag.get_vector_store(chunks)
    except rag.RAGEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "documents": len(pdf_files),
        "chunks": len(chunks),
        "warnings": errors,
    }


# ============================================================
# ASK
# ============================================================
@app.post("/api/ask")
def ask(payload: AskRequest):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if not rag.is_api_configured():
        raise HTTPException(
            status_code=500, detail="GOOGLE_API_KEY is not configured on the server."
        )

    if not rag.is_index_ready():
        raise HTTPException(
            status_code=400,
            detail="Please process your documents before asking a question.",
        )

    try:
        result = rag.answer_question(question)
    except rag.RAGEngineError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "answer": result["answer"], "sources": result["sources"]}


# ============================================================
# STATIC FRONTEND
# ============================================================
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
