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
| Frontend | Vanilla HTML/CSS/JS with XSS-safe renderer |

## Features

- **Intent-based retrieval**: Detects query type (lookup, procedure, eligibility, topic search) and adjusts search strategy
- **Hybrid search**: Combines dense vector search (BGE-M3) with BM25 sparse retrieval
- **BGE reranking**: Cross-encoder reranking for improved relevance
- **HyDE for topic search**: Generates hypothetical documents for better faculty profile matching
- **Low-confidence detection**: Triggers second-pass retrieval with signal terms
- **Conversation history**: Session-based chat with context preservation
- **Disk caching**: Caches responses for identical queries
- **Rate limiting**: Per-IP rate limiting to prevent abuse
- **Secure frontend**: XSS-safe rendering with CSP headers and token validation

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
python -m uvicorn src.api.main:app --reload --port 8001
```

Open `http://localhost:8001` for the landing page or `http://localhost:8001/chat` to start chatting.

## Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| Student | student@nmims.edu | demo123 |
| Faculty | faculty@nmims.edu | demo123 |
| Admin | admin@nmims.edu | demo123 |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ingest_new.py` | Ingest documents into Qdrant + build BM25 index |
| `scripts/reset_database.py` | Wipe Qdrant collection + BM25 index |
| `scripts/test_query.py` | Run a test query from the terminal |
| `scripts/check_chunks.py` | Inspect stored chunks by document name |
| `scripts/analyze_context_usage.py` | Analyze token usage logs |

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
│   ├── metadata.json            # Per-document metadata
│   └── logs/                    # Query and extraction logs
├── frontend/                    # Chat UI (HTML/CSS/JS)
│   ├── chat.html                # Main chat interface
│   ├── chat.js                  # Chat logic with SSE streaming
│   ├── chat.css                 # Chat styles
│   ├── renderer.js              # XSS-safe response renderer
│   ├── signin.html              # Authentication page
│   ├── signin.js                # Sign-in logic
│   ├── index.html               # Landing page
│   └── style.css                # Landing page styles
├── scripts/
│   ├── ingest_new.py            # Ingestion entry point
│   ├── reset_database.py        # Full reset
│   ├── check_chunks.py          # Chunk inspector
│   └── test_query.py            # CLI query tester
├── src/
│   ├── api/main.py              # FastAPI app with SSE streaming
│   ├── chunking/
│   │   └── document_chunker.py  # Type-aware chunking + header splitting
│   ├── generation/
│   │   ├── answer_generator.py  # LLM answer generation with structured output
│   │   ├── prompt_templates.py  # Intent-specific JSON prompts
│   │   └── response_schema.py   # Pydantic response schema
│   ├── ingestion/
│   │   ├── document_processor.py # PDF/JSON/CSV extraction
│   │   └── new_pipeline.py      # Embedding + Qdrant storage
│   ├── retrieval/
│   │   ├── pipeline.py          # Main retrieval orchestrator with HyDE
│   │   ├── hybrid_search.py     # Dense + BM25 hybrid search
│   │   ├── bge_reranker.py      # BGE cross-encoder reranker
│   │   └── query_understanding.py # Intent + entity detection
│   └── utils/
│       ├── llm.py               # Ollama/Gemini client with retry
│       ├── vector_db.py         # Qdrant wrapper
│       ├── dual_encoder_embeddings.py # BGE-M3 query/doc embedding
│       ├── bm25_persistence.py  # BM25 index save/load
│       ├── cache_manager.py     # Disk cache (diskcache)
│       ├── conversation_manager.py # Session history
│       ├── rate_limiter.py      # Per-IP rate limiting
│       └── chunk_preprocessor.py # Chunk text preprocessing
├── bm25_index/                  # Persistent BM25 index
├── cache/                       # Response cache database
├── conversations/               # Session history JSON files
├── qdrant_storage/              # Qdrant vector database
└── tests/
    └── chunking/
        └── test_document_chunker.py
```

## Supported Document Types

| File | Chunking Strategy |
|------|------------------|
| `faculty_data.json` | One chunk per faculty profile section |
| `NMIMS_Employee_Resource_Book_2024-25.pdf` | HR policy — section-based |
| `NMIMS_Faculty_Academic_Guidelines.pdf` | Guidelines — section-based |
| `NMIMS_Faculty_Applications_Compendium.pdf` | Form document — one chunk per form |
| `NMIMS_Faculty_Employment_Agreement_Legal.pdf` | Legal — clause-based |

## Query Limits

- Maximum query length: **5000 characters**
- Rate limit: Configurable per-IP (default: 10 requests/minute)

## Security Features

- **XSS Protection**: All user content rendered via `textContent` (no `innerHTML`)
- **CSP Headers**: Content Security Policy prevents inline scripts
- **Token Validation**: Backend validates JWT tokens on page load
- **Input Sanitization**: Query length limits and validation
- **Rate Limiting**: Prevents abuse and DoS attacks

## Architecture Highlights

### Retrieval Pipeline

1. **Query Understanding**: Detects intent (lookup, procedure, eligibility, topic_search) and extracts entities
2. **Metadata Filtering**: Applies domain and currency filters based on intent
3. **Query Embedding**: Uses BGE-M3 for dense vectors; HyDE for topic search
4. **Hybrid Search**: Combines dense (0.7) + BM25 (0.3) scores → top 40 candidates
5. **BGE Reranking**: Cross-encoder reranks to top 15 chunks
6. **Confidence Check**: If top score < 0.4, triggers second pass with signal terms
7. **Fallback**: Returns no-confidence response with reformulation suggestions

### Answer Generation

- Intent-specific prompts with structured JSON output
- Includes source attribution in answer text
- Low-confidence warnings when appropriate
- Fallback messages for out-of-scope queries

## Troubleshooting

**Ollama connection errors:**
```powershell
# Check if Ollama is running
ollama list

# Restart Ollama service
# Windows: Restart from system tray
# Linux/Mac: systemctl restart ollama
```

**Qdrant connection errors:**
```powershell
# Check Docker container status
docker ps

# Restart Qdrant
docker-compose restart
```

**Empty search results:**
```powershell
# Verify chunks are ingested
python scripts/check_chunks.py

# If empty, reingest
python scripts/reset_database.py
python scripts/ingest_new.py --input data/raw --metadata data/metadata.json
```

## License

Internal NMIMS tool — not for public distribution.
