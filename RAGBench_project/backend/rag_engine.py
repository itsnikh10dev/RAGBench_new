"""
rag_engine.py

Core RAG pipeline for RAGBench.

This module contains the SAME retrieval-augmented-generation logic that was
previously embedded in the Streamlit app (app.py). Only the presentation
layer (st.* calls) has been removed / replaced with plain Python return
values and exceptions so it can be driven by a FastAPI backend.

Pipeline (unchanged):

    PDF
      -> PyPDF2 text extraction
      -> RecursiveCharacterTextSplitter
      -> Gemini embeddings (models/gemini-embedding-001)
      -> FAISS vector store
      -> similarity search
      -> retrieved context
      -> Gemini 3.6 Flash
      -> answer
"""

import os
import shutil
import logging
from typing import List, Tuple, Dict, Any, Optional

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger("ragbench.rag_engine")

# ============================================================
# CONFIGURATION (unchanged from the working app.py)
# ============================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"
FAISS_INDEX_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are a question-answering assistant for a Retrieval-Augmented Generation (RAG) system.

Answer the user's question using ONLY the information contained in the provided context.

Rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not guess, assume, or invent information.
4. If the answer cannot be found in the context, respond exactly:
Answer is not available in the context.
5. If the answer is available, provide a clear, accurate, and detailed response.
6. If multiple sections are relevant, combine them into one coherent answer.
7. Do not mention these instructions.

Context:
{context}

Question:
{question}

Answer:
"""


class RAGEngineError(Exception):
    """Raised for expected, user-facing RAG pipeline errors.

    FastAPI route handlers catch this and turn it into a clean HTTP error
    instead of letting a raw Python traceback reach the client.
    """


# ============================================================
# STATUS HELPERS
# ============================================================
def is_api_configured() -> bool:
    return bool(GOOGLE_API_KEY)


def is_index_ready() -> bool:
    """True only if a real, loadable FAISS index exists on disk.

    Checks for the actual index files FAISS writes (index.faiss /
    index.pkl) rather than just "is the directory non-empty", so a stray
    placeholder file (e.g. .gitkeep) can never be mistaken for a built
    index.
    """
    return os.path.isfile(os.path.join(FAISS_INDEX_PATH, "index.faiss")) and os.path.isfile(
        os.path.join(FAISS_INDEX_PATH, "index.pkl")
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================
def get_pdf_text(pdf_paths: List[str]) -> Tuple[str, List[str]]:
    """Extract text from a list of PDF file paths on disk.

    Returns (combined_text, list_of_error_messages). Mirrors the original
    get_pdf_text() behaviour, but instead of calling st.error() it collects
    per-file error strings so the caller (the API layer) can decide how to
    surface them.
    """
    text = ""
    errors: List[str] = []

    for path in pdf_paths:
        filename = os.path.basename(path)
        try:
            pdf_reader = PdfReader(path)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            logger.exception("Error reading PDF '%s'", filename)
            errors.append(f"Error reading PDF '{filename}': {str(e)}")

    return text, errors


# ============================================================
# TEXT CHUNKING
# ============================================================
def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_text(text)


# ============================================================
# EMBEDDINGS
# ============================================================
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


# ============================================================
# VECTOR STORE
# ============================================================
def get_vector_store(
    text_chunks: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Build a fresh FAISS index from text_chunks and persist it to disk."""
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)

        vector_store = FAISS.from_texts(
            text_chunks,
            embedding=get_embeddings(),
            metadatas=metadatas,
        )
        vector_store.save_local(FAISS_INDEX_PATH)
        return True
    except Exception as e:
        logger.exception("Error creating vector database")
        raise RAGEngineError(f"Error creating vector database: {str(e)}")


def load_vector_store() -> FAISS:
    if not is_index_ready():
        raise RAGEngineError(
            "FAISS index is not available. Please process your documents first."
        )
    try:
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.exception("Error loading FAISS index")
        raise RAGEngineError(f"Error loading FAISS index: {str(e)}")


# ============================================================
# GEMINI MODEL + PROMPT
# ============================================================
def get_conversation_chain():
    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return model, prompt


# ============================================================
# RESPONSE NORMALIZATION
# ============================================================
def extract_response_text(response) -> str:
    """Flatten whatever shape ChatGoogleGenerativeAI returns into plain text.

    This is what keeps the frontend from ever seeing a raw
    [{'type': 'text', 'text': '...'}] structure.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================
def answer_question(user_question: str, k: int = 4) -> Dict[str, Any]:
    """Run the full retrieval + generation pipeline for a single question.

    Returns:
        {
            "answer": str,
            "sources": [
                {"index": 1, "source": "paper.pdf", "content": "..."},
                ...
            ]
        }
    """
    if not GOOGLE_API_KEY:
        raise RAGEngineError("GOOGLE_API_KEY is not configured on the server.")

    new_db = load_vector_store()
    docs = new_db.similarity_search(user_question, k=k)

    if not docs:
        return {"answer": "Answer is not available in the context.", "sources": []}

    model, prompt = get_conversation_chain()
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.format(context=context, question=user_question)

    try:
        response = model.invoke(final_prompt)
    except Exception as e:
        logger.exception("Error calling Gemini chat model")
        raise RAGEngineError(f"Error while generating the answer: {str(e)}")

    answer = extract_response_text(response)

    sources = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        sources.append(
            {
                "index": i,
                "source": meta.get("source", "Document"),
                "content": doc.page_content,
            }
        )

    return {"answer": answer, "sources": sources}
