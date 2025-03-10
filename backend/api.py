import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K_RESULTS

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
async def query(request: QueryRequest):
    """Query the Enron email dataset."""
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
            detail=f"Error processing query: {str(e)}"
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