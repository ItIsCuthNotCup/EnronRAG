import logging
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    API_HOST,
    API_PORT,
    TOP_K_RESULTS,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    OLLAMA_BASE_URL,
    MODEL_NAME,
    VECTOR_DB_DIR,
    CHROMA_DIR
)
from backend.rag import RAGEngine

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Enron Email RAG API")

# CORS middleware — restrict to the local Streamlit origin only.
# allow_origins=["*"] combined with allow_credentials=True is invalid per
# the CORS spec and would be rejected by browsers anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter: max 30 requests per minute per client IP.
# ---------------------------------------------------------------------------
_rate_limit_store: dict = defaultdict(list)
_RATE_LIMIT_MAX = 30
_RATE_LIMIT_WINDOW = 60  # seconds


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]
    if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    _rate_limit_store[client_ip].append(now)


# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=TOP_K_RESULTS, ge=1, le=50)

class QueryResponse(BaseModel):
    answer: str
    sources: list
    retrieval_time: float = 0.0
    total_time: float = 0.0

# Global variables for the embedder, vector DB and RAG engine
embedder = None
vector_db = None
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global embedder, vector_db, rag_engine

    # Import here to avoid circular imports
    from embedding.embedder import EmailEmbedder
    from vector_db.database import VectorDatabase

    try:
        logger.info("Initializing embedder...")
        embedder = EmailEmbedder(
            model_name=EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
            batch_size=EMBEDDING_BATCH_SIZE,
            instruction=None  # No instruction needed
        )

        logger.info("Initializing vector database (ChromaDB)...")
        vector_db = VectorDatabase(
            db_path=CHROMA_DIR,
            embedding_dim=embedder.embedding_dim
        )

        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(
            embedder=embedder,
            vector_db=vector_db,
            ollama_base_url=OLLAMA_BASE_URL,
            model_name=MODEL_NAME
        )

        logger.info("API initialization complete")

    except Exception as e:
        logger.error(f"Error during API initialization: {e}")
        # Continue running even with initialization error
        # The error will be reported when endpoints are called

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Enron Email RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request):
    """Query the Enron email dataset."""
    _check_rate_limit(http_request.client.host)

    if not rag_engine:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Check server logs."
        )

    try:
        logger.info(f"Processing query: {request.query}")
        result = rag_engine.retrieve_and_generate(request.query, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the query."
        )

def start_api():
    """Start the FastAPI server."""
    uvicorn.run(
        "backend.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False
    )

if __name__ == "__main__":
    start_api()
