# Faculty Part - ClassAI

AI-powered faculty resource assistant using semantic RAG with intent-based retrieval.

**Uses Ollama with Llama 2 - runs completely locally, no API keys needed!**

## Prerequisites

1. **Install Ollama**: https://ollama.ai/download
2. **Pull Llama 2**: `ollama pull llama2`
3. **Docker Desktop** for Qdrant

See [OLLAMA_SETUP.md](./OLLAMA_SETUP.md) for detailed Ollama installation.

## Quick Setup

```bash
cd "Faculty Part"

# 1. Start Ollama (if not running)
ollama serve

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate    # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env  # Windows
# cp .env.example .env  # Mac/Linux
# (No API keys needed - uses local Ollama)

# 5. Test setup
python test_setup.py

# 6. Start Qdrant
docker-compose up -d

# 7. Initialize
python setup.py
```

## Usage

```bash
# Ingest documents
python scripts/ingest_documents.py --input data/raw --metadata data/metadata.json

# Test query
python scripts/test_query.py "How do I apply for casual leave?"

# Start API
python -m src.api.main
# Chat UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Core Features

- **Semantic Chunking**: Procedures stay intact, rules keep conditions
- **Intent-Based Retrieval**: Routes queries to optimal chunk levels  
- **Hybrid Search**: Vector + BM25 keyword matching
- **Grounded Answers**: LLM answers only from retrieved chunks

Details: [docs/CHUNKING_STRATEGY.md](./docs/CHUNKING_STRATEGY.md)
