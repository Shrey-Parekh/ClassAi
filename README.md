# ClassAI - NMIMS Knowledge Assistant

> AI-powered institutional knowledge assistant with role-scoped RAG for faculty and students

ClassAI is a production-ready RAG (Retrieval-Augmented Generation) system that provides instant access to NMIMS institutional knowledge including policies, syllabi, question papers, and academic guidelines. Built with hybrid search, cross-encoder reranking, and role-based access control.

## 🎯 Features

### Core Capabilities
- **Hybrid RAG System**: Dense (semantic) + BM25 (keyword) + BGE cross-encoder reranking
- **Role-Based Access Control**: Student, Faculty, and Admin roles with scoped data access
- **Multi-Collection Support**: Separate collections for faculty resources and student materials
- **Structured JSON Responses**: Formatted answers with citations and source links
- **Conversation History**: Session-based chat with context preservation
- **Query Intelligence**: Automatic abbreviation expansion, intent detection, and query understanding

### Advanced Features
- **Metadata Filtering**: Comprehensive listing queries (e.g., "list all units in ML")
- **Abbreviation Expansion**: CS → Cyber Security, ML → Machine Learning, etc.
- **Scope Selection**: Query student/faculty/both collections simultaneously
- **Source Attribution**: Every answer includes document references
- **Caching**: Query result caching for improved performance
- **Rate Limiting**: Per-IP request throttling

## 🏗️ Architecture

```
ClassAI/
├── Faculty Part/          # Main application (runtime + faculty data)
│   ├── src/
│   │   ├── api/          # FastAPI endpoints
│   │   ├── retrieval/    # Hybrid search + reranking
│   │   ├── generation/   # LLM answer generation
│   │   ├── ingestion/    # Document processing pipeline
│   │   └── utils/        # Shared utilities
│   ├── frontend/         # Web UI (HTML/CSS/JS)
│   ├── data/            # Faculty documents (PDFs)
│   └── docker-compose.yml
│
└── Student Part/         # Student data ingestion only
    ├── ingest/          # LangChain-based indexing
    ├── data/
    │   ├── syllabus/    # Course syllabi (Markdown)
    │   └── question_papers/  # Exam papers (Markdown)
    └── .env
```

### Technology Stack

**Backend:**
- FastAPI (API server)
- Qdrant (vector database)
- BAAI/bge-m3 (embeddings, 1024-dim)
- BAAI/bge-reranker-v2-m3 (cross-encoder)
- Ollama Gemma3:12b (LLM)
- BM25Okapi (sparse retrieval)

**Frontend:**
- Vanilla JavaScript (no framework)
- Server-Sent Events (streaming)
- Responsive design with accessibility

**Data Processing:**
- PyMuPDF (PDF extraction)
- LangChain (student data chunking)
- Sentence Transformers (embeddings)

## 🚀 Quick Start

### Prerequisites

```bash
# Required
- Python 3.10+
- Docker & Docker Compose
- Ollama (for LLM)
- 8GB+ RAM
- 10GB+ disk space

# Optional
- CUDA-capable GPU (for faster embeddings)
```

### 1. Clone Repository

```bash
git clone <repository-url>
cd ClassAI
```

### 2. Start Infrastructure

```bash
cd "Faculty Part"
docker-compose up -d
```

This starts:
- Qdrant vector database (port 6333)
- Qdrant dashboard (port 6334)

### 3. Install Ollama & Pull Model

```bash
# Install Ollama from https://ollama.ai
ollama pull gemma3:12b
ollama pull bge-m3
```

### 4. Install Python Dependencies

```bash
# Faculty Part
cd "Faculty Part"
pip install -r requirements.txt

# Student Part (for data ingestion)
cd "../Student Part"
pip install -r requirements.txt
```

### 5. Configure Environment

```bash
# Faculty Part/.env
cp .env.example .env
# Edit .env with your settings (defaults work for local development)

# Student Part/.env
cp .env.example .env
# Edit .env (should point to same Qdrant instance)
```

### 6. Index Documents

**Index Faculty Documents:**
```bash
cd "Faculty Part"
python scripts/ingest_new.py
```

**Index Student Documents:**
```bash
cd "Student Part"
python ingest/index.py
```

### 7. Start Server

```bash
cd "Faculty Part"
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 8. Access Application

- **Homepage**: http://localhost:8000
- **Chat Interface**: http://localhost:8000/chat
- **Sign In**: http://localhost:8000/signin
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6334

## 🔐 Demo Credentials

| Role | Email | Password | Access |
|------|-------|----------|--------|
| **Student** | student@nmims.edu | demo123 | Student collection only |
| **Faculty** | faculty@nmims.edu | demo123 | Both collections, scope selector |
| **Admin** | admin@nmims.edu | demo123 | Full access, all scopes |

## 📚 Usage Examples

### Student Queries
```
"List all units in Machine Learning"
"Course Outcomes of CS"
"What topics are covered in Unit 2 of ML?"
"Show me question papers for Cyber Security"
```

### Faculty Queries
```
"What is the leave policy?"
"How do I apply for research grants?"
"Tell me about the Academic Guidelines"
"What forms do I need for sabbatical leave?"
```

### Scope Selection
- **Student Scope**: Queries only student collection (syllabi, question papers)
- **Faculty Scope**: Queries only faculty collection (policies, forms, guidelines)
- **Both Scope**: Queries both collections with RRF merging

## 🔧 Configuration

### Environment Variables

**Faculty Part/.env:**
```bash
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:12b
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3

# Vector Database
QDRANT_URL=http://localhost:6333
FACULTY_COLLECTION_NAME=faculty_chunks
STUDENT_COLLECTION_NAME=academic_rag

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Demo Authentication
DEMO_USER_STUDENT=student@nmims.edu:<bcrypt_hash>:student:Demo Student
DEMO_USER_FACULTY=faculty@nmims.edu:<bcrypt_hash>:faculty:Demo Faculty
DEMO_USER_ADMIN=admin@nmims.edu:<bcrypt_hash>:admin:Demo Admin
```

**Student Part/.env:**
```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=academic_rag
```

### Retrieval Parameters

Edit `Faculty Part/config/chunking_config.py`:

```python
# Intent-based chunk limits
INTENT_CHUNK_LIMITS = {
    "lookup": 15,
    "topic_search": 25,
    "procedure": 20,
    "general": 20,
}

# Retrieval pipeline
TOP_K_INITIAL = 40      # Hybrid search candidates
TOP_K_RERANKED = 15     # After cross-encoder reranking
```

## 📖 API Reference

### Authentication

**POST /api/auth/signin**
```json
{
  "email": "admin@nmims.edu",
  "password": "demo123"
}
```

Response:
```json
{
  "token": "...",
  "role": "admin",
  "user": {
    "name": "Demo Admin",
    "email": "admin@nmims.edu"
  }
}
```

### Query Endpoint

**POST /query**
```json
{
  "query": "List all units in Machine Learning",
  "scope": "student",
  "session_id": "uuid",
  "stream": true,
  "top_k": 20
}
```

Headers:
```
Authorization: Bearer <token>
Content-Type: application/json
```

Response (Server-Sent Events):
```
data: {"status": "retrieval", "message": "Searching..."}
data: {"status": "generation", "message": "Generating answer..."}
data: {"answer": {...}, "sources": [...]}
```

## 🧪 Testing

### Run Tests
```bash
cd "Faculty Part"
pytest tests/
```

### Manual Testing
```bash
# Test retrieval
python scripts/test_retrieval.py

# Test embeddings
python scripts/test_embeddings.py

# Check database
python check_db.py
```

## 🐛 Troubleshooting

### Common Issues

**1. Server won't start - "Token in active_tokens: False"**
- **Cause**: Server restarted, tokens cleared from memory
- **Fix**: Logout and login again

**2. "No chunks found" for valid queries**
- **Cause**: BM25 index not built or collection empty
- **Fix**: Re-run ingestion scripts

**3. HuggingFace connection errors**
- **Cause**: Model trying to check for updates
- **Fix**: Already handled with offline mode flags

**4. Slow embeddings**
- **Cause**: Running on CPU
- **Fix**: Use GPU or reduce batch size

**5. "Course Outcomes of CS" returns 0 chunks**
- **Cause**: Abbreviation not expanded
- **Fix**: Already handled with abbreviation expansion

### Debug Mode

Enable detailed logging:
```bash
export DEBUG=True
export LOG_LEVEL=DEBUG
python -m uvicorn src.api.main:app --reload
```

Check logs:
```bash
tail -f Faculty\ Part/logs/context_usage.jsonl
tail -f Faculty\ Part/embedding_log.jsonl
```

## 📊 Performance

### Benchmarks (Local Machine)

| Operation | Time | Notes |
|-----------|------|-------|
| Query (cold) | ~2-3s | First query after restart |
| Query (warm) | ~1-2s | With cache |
| Embedding (CPU) | ~100ms/chunk | BAAI/bge-m3 |
| Embedding (GPU) | ~20ms/chunk | CUDA acceleration |
| Reranking | ~50ms | 15 candidates |
| LLM Generation | ~1-2s | Streaming response |

### Optimization Tips

1. **Use GPU**: 5x faster embeddings
2. **Enable caching**: Reduces repeated queries
3. **Adjust top_k**: Lower = faster, higher = more comprehensive
4. **Use streaming**: Better perceived performance
5. **Persistent BM25**: Cached on disk, fast startup

## 🔒 Security Notes

⚠️ **Current Implementation is for Development/Demo Only**

**Before Production:**
- [ ] Replace in-memory tokens with Redis/database
- [ ] Implement JWT with expiration and refresh
- [ ] Add HTTPS/TLS
- [ ] Use proper user database (not .env)
- [ ] Add input sanitization and validation
- [ ] Implement proper CORS policies
- [ ] Add audit logging
- [ ] Use secrets management (Vault, AWS Secrets Manager)
- [ ] Add rate limiting per user (not just IP)
- [ ] Implement session timeout

## 🤝 Contributing

### Adding New Documents

**Faculty Documents:**
1. Place PDFs in `Faculty Part/data/raw/`
2. Update `Faculty Part/data/metadata.json`
3. Run: `python scripts/ingest_new.py`

**Student Documents:**
1. Place Markdown files in `Student Part/data/syllabus/` or `Student Part/data/question_papers/`
2. Run: `python ingest/index.py`

### Adding New Course Abbreviations

Edit `Faculty Part/src/retrieval/scope_router.py`:
```python
abbreviations = {
    r'\bCS\b': 'Cyber Security',
    r'\bML\b': 'Machine Learning',
    r'\bYOUR_ABBR\b': 'Full Name',  # Add here
}
```

## 📝 License

[Your License Here]

## 👥 Authors

[Your Team/Organization]

## 🙏 Acknowledgments

- BAAI for BGE models
- Qdrant for vector database
- Ollama for local LLM inference
- NMIMS for institutional support

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Status**: Production-Ready (with security hardening needed)
