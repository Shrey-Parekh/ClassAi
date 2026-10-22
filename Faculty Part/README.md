# Faculty Part — ClassAI

AI-powered faculty resource assistant for NMIMS University. Uses a RAG pipeline with intent-based retrieval to answer questions about faculty profiles, HR policies, leave rules, procedures, and legal documents.

Runs fully locally — no cloud API required.

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama — `gemma3:12b` |
| Embeddings | `BAAI/bge-m3` (1024-dim, via sentence-transformers) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Vector DB | Qdrant (Docker) |
| BM25 | rank-bm25 with persistent index |
| API | FastAPI + uvicorn |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/download) with `gemma3:12b` pulled
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Tesseract OCR (optional, for scanned PDFs)
- poppler (optional, required by pdf2image for OCR fallback)

## Setup

```powershell
# 1. Pull the LLM (if not already done)
ollama pull gemma3:12b

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment
copy .env.example .env
# Edit .env if needed — defaults work out of the box for local Ollama

# 5. Start Qdrant
cd "Faculty Part"
docker-compose up -d

# 6. Ingest documents
python scripts/ingest_new.py --input data/raw --metadata data/metadata.json

# 7. Start the API server
python -m uvicorn src.api.main:app --reload --port 8000
```

Open `http://localhost:8000/chat` in your browser.

## Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| Student | student@nmims.edu | demo123 |
| Faculty | faculty@nmims.edu | demo123 |
| Admin | admin@nmims.edu | demo123 |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ingest_new.py` | Ingest documents into Qdrant |
| `scripts/reset_database.py` | Wipe Qdrant collection + BM25 index |
| `scripts/test_query.py` | Run a test query from the terminal |
| `scripts/check_chunks.py` | Inspect stored chunks by document name |
| `scripts/analyze_context_usage.py` | Analyse token usage logs |

## Resetting and Reingesting

```powershell
cd "Faculty Part"
python scripts/reset_database.py
python scripts/ingest_new.py --input data/raw --metadata data/metadata.json
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `LLM_MODEL` | `gemma3:12b` | Model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GEMINI_API_KEY` | — | Required only when `LLM_PROVIDER=gemini` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformers model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | BGE reranker model |

## Project Structure

```
Faculty Part/
├── config/
│   └── chunking_config.py       # Chunk size limits and intent budgets
├── data/
│   ├── raw/                     # Source PDFs and JSON faculty data
│   └── metadata.json            # Per-document metadata
├── frontend/                    # Chat UI (HTML/CSS/JS)
├── scripts/
│   ├── ingest_new.py            # Ingestion entry point
│   ├── reset_database.py        # Full reset
│   ├── check_chunks.py          # Chunk inspector
│   └── test_query.py            # CLI query tester
├── src/
│   ├── api/main.py              # FastAPI app
│   ├── chunking/
│   │   └── document_chunker.py  # Type-aware chunking + header splitting
│   ├── generation/
│   │   ├── answer_generator.py  # LLM answer generation
│   │   ├── prompt_templates.py  # Intent-specific JSON prompts
│   │   └── response_schema.py   # Pydantic response schema
│   ├── ingestion/
│   │   ├── document_processor.py # PDF/JSON/CSV extraction
│   │   └── new_pipeline.py      # Embedding + Qdrant storage
│   ├── retrieval/
│   │   ├── pipeline.py          # Main retrieval orchestrator
│   │   ├── hybrid_search.py     # Dense + BM25 hybrid search
│   │   ├── bge_reranker.py      # BGE cross-encoder reranker
│   │   └── query_understanding.py # Intent + entity detection
│   └── utils/
│       ├── llm.py               # Ollama/Gemini client with retry
│       ├── vector_db.py         # Qdrant wrapper
│       ├── query_embedder.py    # BGE-M3 query embedding
│       ├── bm25_persistence.py  # BM25 index save/load
│       ├── cache_manager.py     # Disk cache (diskcache)
│       ├── conversation_manager.py # Session history
│       └── rate_limiter.py      # Per-IP rate limiting
└── tests/
    └── chunking/
        └── test_document_chunker.py
```

## Supported Document Types

| File | Chunking Strategy |
|------|------------------|
| `facult_data.json` | One chunk per faculty profile section |
| `NMIMS_Employee_Resource_Book_2024-25.pdf` | HR policy — section-based |
| `NMIMS_Faculty_Academic_Guidelines.pdf` | Guidelines — section-based |
| `NMIMS_Faculty_Applications_Compendium.pdf` | Form document — one chunk per form |
| `NMIMS_Faculty_Employment_Agreement_Legal.pdf` | Legal — clause-based |
