# Enron Email Terminal

A retro terminal-style web application for searching and exploring the Enron email dataset using natural language queries and RAG (Retrieval-Augmented Generation).

## Overview

The application renders a green-on-black CRT terminal in your browser. You type questions in plain English and get answers drawn from real Enron emails. It has two operating modes:

| Mode | When to use | What runs |
|------|-------------|-----------|
| **Simple (default)** | No extra setup needed | `app.py` does keyword matching against 5 hardcoded sample emails |
| **Full RAG** | When you have Ollama + the full dataset | FastAPI backend (`backend/api.py`) queries a ChromaDB vector store and an LLM |

For a detailed architecture breakdown see [CLAUDE.md](CLAUDE.md).

## Requirements

- Python 3.8+
- pip packages: `streamlit pandas requests`
- **Full RAG mode only:** `langchain chromadb sentence-transformers torch fastapi uvicorn tqdm numpy`
- **Full RAG mode only:** [Ollama](https://ollama.com) running locally with a model pulled (e.g. `llama3:8b`)

## Quickstart

### Simple mode (no Ollama required)

```bash
git clone https://github.com/ItIsCuthNotCup/EnronRAG.git
cd EnronRAG
pip install streamlit pandas requests
streamlit run app.py
# Opens http://localhost:8501 in your browser automatically
```

Type any question at the `$` prompt and press Enter.

### Windows launcher

```bat
.\ENRON_TERMINAL.bat
```

Installs dependencies, writes `config.py`, and starts the Streamlit app automatically.

### Full RAG mode

```bash
# 1. Install all dependencies
pip install streamlit pandas requests langchain chromadb sentence-transformers torch fastapi uvicorn tqdm numpy

# 2. Start Ollama and pull a model
ollama pull llama3:8b

# 3. Build the vector database (one-time, ~5–30 min depending on dataset size)
python process_enron_data.py              # use real emails.csv if present
python process_enron_data.py --sample    # use built-in 115-email sample
python process_enron_data.py --force     # rebuild from scratch
python process_enron_data.py --export    # also export DB as pre_built_db.zip

# 4. Start the FastAPI backend
python backend/api.py

# 5. Run the UI
streamlit run app.py
```

The full RAG backend serves `POST /query` on the port configured in `config.py`.

## Example Queries

- `Who was the CEO of Enron?`
- `Tell me about the accounting scandal`
- `What were the Raptor vehicles?`
- `What financial strategies did Enron use?`
- `What did Sherron Watkins warn about?`

## Configuration

All settings live in `config.py`. Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_NAME` | `llama3:8b` | Ollama model used for generation |
| `TOP_K_RESULTS` | `5` | Number of emails retrieved per query |
| `CHROMA_DIR` | `vector_db/chroma` | Where the vector DB is stored |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |

> **Note:** The Windows `.bat` launcher regenerates `config.py` on each run; manual edits will be overwritten.

## License

GPL-3.0 — see LICENSE for details.
