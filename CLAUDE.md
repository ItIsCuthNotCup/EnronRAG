# CLAUDE.md — Enron Email Terminal (EnronRAG)

AI assistant guide for understanding and working with this codebase.

---

## Project Overview

**Enron Email Terminal** is a retro terminal-style web application that lets users query the historical Enron email dataset using natural language. It combines a Streamlit-based UI with RAG (Retrieval-Augmented Generation) to surface relevant emails and generate answers.

The application has two operating modes:
1. **Simple keyword mode** (default, no external dependencies) — `app.py` uses hardcoded sample data and keyword matching.
2. **Full RAG mode** — `backend/api.py` + `backend/rag.py` wire a ChromaDB vector store to an Ollama-hosted LLM for semantic search and AI-generated answers.

---

## Repository Structure

```
EnronRAG/
├── app.py                    # Streamlit entry point (simple keyword search, retro UI)
├── config.py                 # Central configuration (paths, model names, settings)
├── process_enron_data.py     # One-time script: copies CSV → chunks → builds ChromaDB
├── ENRON_TERMINAL.bat        # Windows launcher: installs deps, configures, starts app
│
├── backend/
│   ├── api.py                # FastAPI server (full RAG mode, /query endpoint)
│   └── rag.py                # RAGEngine: embeds query → retrieves docs → calls Ollama
│
├── data_processing/
│   ├── ingestion.py          # Parses raw .eml files into JSON batches
│   ├── preprocessing.py      # Cleans email bodies and header fields
│   └── chunking.py           # Splits emails into overlapping text chunks
│
├── embedding/
│   └── embedder.py           # EmailEmbedder: wraps SentenceTransformer for batch encoding
│
├── vector_db/
│   ├── database.py           # VectorDatabase: wraps ChromaDB (add_documents, search)
│   └── DATABASE_LOCATION.txt # Auto-generated marker showing DB path and stats
│
├── utils/
│   ├── data_detection.py     # Detects Enron dataset format (CSV vs raw maildir)
│   └── logger.py             # Logging setup helpers
│
├── data/
│   └── sample_email.txt      # One sample email for reference/testing
│
└── logs/                     # Runtime logs (gitignored)
```

---

## Key Configuration (`config.py`)

All tuneable parameters live here. The `.bat` launcher regenerates this file on each run.

| Variable | Default | Purpose |
|---|---|---|
| `CSV_FILE` | `"emails.csv"` | Path to raw Enron CSV |
| `TOP_K_RESULTS` | `5` | Number of docs retrieved per query |
| `VECTOR_DB_DIR` | `"vector_db"` | Root dir for vector DB files |
| `CHROMA_DIR` | `"vector_db/chroma"` | ChromaDB persistence directory |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama API endpoint |
| `MODEL_NAME` | `"llama3:8b"` | Ollama model for generation |
| `USE_CSV_SEARCH` | `False` | Fallback to CSV search if True |

Additional constants used by the full RAG pipeline (referenced in `api.py` but not in the committed `config.py`): `API_HOST`, `API_PORT`, `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`.

---

## Data Flow

### Simple mode (`app.py`)
```
User types query in browser
  → URL parameter ?command=<query>
  → search_emails() keyword matching against SAMPLE_DATA (5 hardcoded emails)
  → formatted text response rendered in terminal UI
```

### Full RAG mode (`backend/api.py`)
```
POST /query  {query, top_k}
  → RAGEngine.retrieve_and_generate()
      1. EmailEmbedder.embed_query()      # SentenceTransformer encode
      2. VectorDatabase.search()          # ChromaDB cosine similarity
      3. RAGEngine.generate_prompt()      # Format top-k emails as context
      4. RAGEngine.query_ollama()         # POST to Ollama /generate
  → {answer, sources, retrieval_time, total_time}
```

### Data ingestion pipeline (`process_enron_data.py`)
```
emails.csv  (or raw maildir)
  → find_source_data() / create_sample_data()
  → copy_dataset()
  → LangChain DataFrameLoader + RecursiveCharacterTextSplitter
  → HuggingFaceEmbeddings
  → Chroma vectorstore (batch insert, persisted to vector_db/chroma/)
```

The modular pipeline in `data_processing/` (ingestion → preprocessing → chunking) is an alternative path for raw `.eml` maildir files, but `process_enron_data.py` is what actually runs and uses LangChain directly.

---

## Running the Application

### Quickstart (Windows)
```bat
.\ENRON_TERMINAL.bat
```
This installs pip dependencies, writes `config.py`, and launches `streamlit run app.py`.

### Manual start
```bash
pip install streamlit pandas requests
streamlit run app.py
# Opens http://localhost:8501
```

### Full RAG mode (requires Ollama)
```bash
# 1. Start Ollama and pull a model
ollama pull llama3:8b

# 2. Build the vector database (one-time)
python process_enron_data.py              # uses real emails.csv if present
python process_enron_data.py --sample    # use built-in sample data
python process_enron_data.py --force     # rebuild from scratch
python process_enron_data.py --export    # also zip the DB as pre_built_db.zip

# 3. Start the FastAPI backend
python backend/api.py
# Serves at http://localhost:<API_PORT>

# 4. Run Streamlit (app.py must be updated to call the API instead of search_emails)
streamlit run app.py
```

---

## Architecture Notes

### `app.py` — The current user-facing app
- Uses `st.components.v1.html()` to render a full custom HTML/CSS/JS terminal rather than native Streamlit widgets.
- Session state tracks terminal history as a list of `(type, content)` tuples where `type ∈ {"command", "response", "message"}`.
- Commands are submitted via URL query parameters (`?command=<value>`), causing a full Streamlit rerender on each query — no websocket needed.
- HTML output is sanitized with `html.escape()` before injection.
- Logging goes to `logs/enron_app.log`.

### `backend/rag.py` — RAGEngine
- Holds an in-memory `query_cache` dict to avoid redundant Ollama calls for repeated queries.
- Ollama is called via plain `requests.post` (not an Ollama SDK).
- LLM temperature is `0.3`, max tokens `1024`.
- Documents are sorted by relevance score (descending) before prompt formatting.

### `vector_db/database.py` — VectorDatabase
- Uses `chromadb.PersistentClient` with `anonymized_telemetry=False`.
- Collection name is hardcoded as `"enron_emails"`.
- Similarity score = `1.0 - cosine_distance` (ChromaDB returns distances).
- Batch insert size: 1000 documents.

### `embedding/embedder.py` — EmailEmbedder
- Falls back to CPU automatically if CUDA is unavailable.
- Subject line is prepended to chunk text during embedding: `"Subject: <subject>\n<body>"`.
- Optional `instruction` prefix is supported (for instruction-tuned embedding models).

### `data_processing/`
- `ingestion.py`: walks a directory tree, parses RFC 2822 email files with Python's `email` stdlib, saves 1000-email JSON batches.
- `preprocessing.py`: strips forwarding headers, email signatures, collapses whitespace.
- `chunking.py`: sentence-boundary-aware splitting using `. ` as separator; overlap in terms of sentence count.

---

## Important Conventions

### Configuration
- Always modify `config.py` when changing model names, paths, or thresholds — do not hardcode values in other modules.
- The `.bat` file **overwrites** `config.py` on launch; changes to `config.py` will be lost unless you also update the `.bat`.

### Imports
- `backend/api.py` adds the project root to `sys.path` at startup; all other modules use relative imports or rely on being run from the project root.
- Run scripts from the project root (`/home/user/EnronRAG/`) to ensure path resolution works correctly.

### Logging
- All modules use `logging.getLogger(__name__)`.
- The central helper is `utils/logger.py:setup_logger()`, but `app.py` configures its own root logger directly.
- Log files go in the `logs/` directory (gitignored).

### Data files
- `emails.csv` and `*.sqlite3` are gitignored (large files).
- `vector_db/chroma/` contents are gitignored; only Python source files in `vector_db/` are tracked.
- `pre_built_db.zip` is gitignored.
- `data/sample_email.txt` is the only committed data sample.

### HTML safety
- User input rendered in terminal HTML must always be escaped with `html.escape()` before injection into `innerHTML`. This is done in `app.py:escape_html_except_br()` and in the history loop.

---

## External Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `fastapi` + `uvicorn` | REST API for full RAG mode |
| `chromadb` | Vector database |
| `sentence-transformers` | Text embeddings |
| `langchain` | Document loading / text splitting (in `process_enron_data.py`) |
| `torch` | Inference backend for SentenceTransformer |
| `pandas` | CSV loading and DataFrame operations |
| `requests` | HTTP calls to Ollama API |
| `tqdm` | Progress bars during batch processing |
| `numpy` | Embedding array operations |
| Ollama (external) | LLM inference server (not a pip package) |

---

## Known Limitations / Technical Debt

1. **`config.py` mismatch**: The committed `config.py` has fewer keys than `backend/api.py` expects (`API_HOST`, `API_PORT`, `EMBEDDING_MODEL`, etc. are missing). The full RAG backend will fail on import without these.
2. **Two separate search paths**: `app.py` uses its own hardcoded `search_emails()` function; `backend/rag.py` is the real RAG engine but is not called by `app.py`.
3. **`.bat` overwrites config**: The Windows launcher regenerates `config.py` from scratch, which will drop any manual additions.
4. **Hardcoded Windows path in `process_enron_data.py`**: Line 48 has a user-specific Windows path as a fallback source; remove or update before deploying elsewhere.
5. **ChromaDB version sensitivity**: LangChain's `langchain.vectorstores.Chroma` vs `chromadb.PersistentClient` — both are used in different files; ensure package versions are compatible.
6. **No requirements.txt**: Dependencies must be inferred from source; consider adding one.
