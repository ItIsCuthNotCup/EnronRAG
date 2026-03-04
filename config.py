# Enron Email RAG Configuration
# NOTE: The Windows launcher (ENRON_TERMINAL.bat) regenerates this file on every run.
# Manual edits here will be lost the next time the .bat is executed.

# Path to the raw Enron email CSV (e.g. the Kaggle dataset "emails.csv").
# Used by process_enron_data.py as a fallback search location.
CSV_FILE = "emails.csv"

# Number of documents (email chunks) returned per query in RAG mode.
# Higher values give more context but increase LLM prompt size and latency.
TOP_K_RESULTS = 5

# Root directory for all vector database files.
VECTOR_DB_DIR = "vector_db"

# Sub-directory where ChromaDB persists its SQLite and segment files.
# Must match the persist_directory used when the DB was built.
CHROMA_DIR = "vector_db/chroma"

# Base URL for the locally-running Ollama inference server.
# Change port if Ollama was started on a non-default port.
OLLAMA_BASE_URL = "http://localhost:11434"

# Ollama model tag used for answer generation.
# Run `ollama pull <MODEL_NAME>` before starting the backend.
MODEL_NAME = "llama3:8b"

# When True, the backend falls back to direct CSV search instead of ChromaDB.
# Useful for debugging without a built vector database.
USE_CSV_SEARCH = False
