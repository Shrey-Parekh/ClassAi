# RAG System Architecture Documentation

## System Overview

Faculty Part RAG system for NMIMS institutional knowledge retrieval using semantic search, hybrid retrieval, and structured generation.

**Core Stack:**
- Embedding Model: BAAI/bge-m3 (1024 dimensions, 8192 token context)
- Vector DB: Qdrant (local storage)
- Reranker: BAAI/bge-reranker-v2-m3
- LLM: Gemma 3:12b (via Ollama, 32K context)
- Sparse Retrieval: BM25 (rank-bm25) with disk persistence
- Framework: FastAPI + async operations

---

## 1. CHUNKING STRATEGY

### Current Implementation

**Two Competing Strategies** (problematic):

#### Strategy A: Semantic Chunker (`semantic_chunker.py`)
- **3-Level Hierarchy:**
  - Level 1 (Overview): Document summary, ~200-400 tokens
  - Level 2 (Procedure): Complete sections, max 980 tokens
  - Level 3 (Atomic): Facts/rules/deadlines, ~50-200 tokens

- **Chunking Logic:**
  - Splits on headers using regex patterns
  - Keeps procedures complete (never splits mid-step)
  - Extracts atomic facts (definitions, deadlines, rules)
  - Uses LLM for overview generation (optional)

- **Token Limits:**
  - MAX_LEVEL2_TOKENS = 980
  - OVERLAP_TOKENS = 50
  - No hard limit on Level 1/3

#### Strategy B: Document Chunker (`document_chunker.py`)
- **Document-Type-Specific:**
  - Faculty Profiles: 1 chunk per profile (max 8000 tokens)
  - HR Policies: 2000 tokens, 200 overlap
  - Legal Documents: 1500 tokens, 150 overlap
  - Guidelines: 2500 tokens, 250 overlap
  - Procedures: 3000 tokens, 300 overlap
  - Forms: 1000 tokens, no overlap

- **Chunking Logic:**
  - Detects document type from filepath
  - Splits by headers first
  - Falls back to paragraph boundaries
  - Size-based splitting with overlap as last resort

- **Special Handling:**
  - Faculty profiles: Keep bio + publications together
  - Forms: One form = one chunk
  - Procedures: Keep steps together

### Preprocessing Pipeline

**ChunkPreprocessor** (`chunk_preprocessor.py`):
- Normalizes whitespace and special characters
- UTF-8 safety checks
- Validates chunk size (MIN: 20 chars, MAX: 1960 chars)
- Splits oversized chunks at sentence boundaries with 2-sentence overlap
- Discards empty/too-short chunks

**Quality Filters:**
- Minimum 50 tokens
- Must contain alphabetic content
- Duplicate detection via MD5 hash
- Encoding safety (UTF-8 validation)

---

## 2. INGESTION PIPELINE

### Flow: Document → Vector DB

```
Raw Document (PDF/JSON/CSV)
    ↓
[DocumentProcessor] - Extract text, OCR, parse structured data
    ↓
[SemanticChunker OR DocumentChunker] - Create semantic chunks
    ↓
[ChunkPreprocessor] - Normalize, validate, split oversized
    ↓
[DualEncoderEmbeddings] - Generate embeddings (BAAI/bge-m3)
    ↓
[VectorDB.upsert] - Store in Qdrant with metadata
    ↓
[BM25 Index] - Build sparse index (persisted to disk)
```

### Embedding Strategy

**Primary: BAAI/bge-m3**
- 1024 dimensions
- 8192 token context window
- Used for both documents and queries (consistency guaranteed)

**Dual Encoder Support:**
- Primary: BAAI/bge-m3
- Fallback: all-mpnet-base-v2 (768 dimensions)
- Automatic fallback on encoding errors
- Batch embedding for efficiency (5+ chunks)

### Metadata Schema

**Stored with each chunk:**
```python
{
    "content": str,              # Cleaned chunk text
    "chunk_level": str,          # "level_1_overview" | "level_2_procedure" | "level_3_atomic"
    "content_type": str,         # "procedure" | "rule" | "policy" | "form" | etc.
    "token_count": int,
    "parent_doc_id": str,
    "original_chunk_id": str,
    "domain": str,               # "faculty_info" | "policies" | "procedures" | "general"
    "is_current": bool,          # For temporal filtering
    "doc_id": str,
    "title": str,
    "date": str,
    "applies_to": str,
    "source_type": str,          # "faculty_profile" | "hr_policy" | "legal_document" | etc.
    "chunk_type": str,           # "full_profile" | "policy_section" | "procedure" | etc.
    "person_name": str,          # For faculty profiles (cleaned, lowercase)
    "name_variants": List[str],  # Name parts for exact matching
    "department": str,
    "research_tags": List[str],
    "topic_tags": List[str],
    "has_steps": bool,
    "has_forms": bool,
    "embedding_model": str,
    "embedding_fallback": bool,
    "chunk_length_chars": int,
    "was_split": bool,
    "split_index": int
}
```

---

## 3. RETRIEVAL PIPELINE

### Query Flow

```
User Query
    ↓
[QueryAnalyzer] - Intent, domain, entities, expansion
    ↓
[Cache Check] - Return cached result if exists
    ↓
[Title Stripping] - Remove titles for clean embedding
    ↓
[Query Embedding] - BAAI/bge-m3 (or HyDE for topic search)
    ↓
[Optional: Name Embedding] - Dual embedding for faculty lookup
    ↓
[Metadata Pre-filtering] - Domain + is_current filters
    ↓
[Hybrid Search] - Dense (semantic) + Sparse (BM25) with intent-based weighting
    ↓ (20 candidates)
[BGE Reranking] - Cross-encoder reranking
    ↓ (15 final chunks)
[Dynamic Chunk Selection] - Token budget + relevance threshold
    ↓ (5-25 chunks based on intent)
[Answer Generation] - Structured JSON response
    ↓
[Cache Result] - Store for 1 hour
    ↓
Response to User
```

### Query Understanding

**QueryAnalyzer** (`query_understanding.py`):

**Intent Detection** (pattern-based scoring):
- `lookup`: Who/what questions, faculty info, research, publications
- `procedure`: How-to questions, steps, application processes
- `eligibility`: Can I, requirements, criteria, qualifications
- `general`: Fallback for unclear queries

**Domain Detection** (weighted scoring):
- `faculty_info`: Faculty, professor, research, publications, awards
- `policies`: Policy, rules, regulations, leave, salary, benefits
- `procedures`: Procedure, process, forms, applications, approval
- `general`: Fallback

**Entity Extraction** (regex-based):
- Person names (with title handling)
- Departments (CS, IT, mechanical, etc.)
- Research topics (AI, ML, blockchain, etc.)
- Form names (Form A-123)
- Document types

**Query Expansion:**
- Strips titles before expansion (clean keyword matching)
- Intent-based synonym addition
- Name variation generation for faculty queries
- Aggressive expansion for vague queries (20 terms)
- Moderate expansion for specific queries (8 terms)

**Title Stripping:**
- Applied before: query embedding, name embedding, sparse search
- NOT applied to: display query, user-facing text
- Titles removed: professor, prof, dr, doctor, mr, mrs, ms, sir, ma'am

**Metadata Filters:**
- Domain filtering (faculty_info, policies, procedures)
- Temporal filtering (is_current=True for "latest", "current", "2024+")
- Entity-based filtering (future: department, document type)

### Hybrid Search

**HybridSearchEngine** (`hybrid_search.py`):

**Dense Search (Semantic):**
- Uses BAAI/bge-m3 query embedding
- Cosine similarity in Qdrant
- Retrieves top 20 candidates

**Sparse Search (Keyword):**
- BM25Okapi algorithm (rank-bm25)
- Tokenized corpus (lowercase, split on whitespace)
- Persisted to disk for fast startup
- Retrieves top 20 candidates

**Intent-Based Weighting:**
```python
INTENT_WEIGHTS = {
    "person_lookup": {"dense": 0.40, "sparse": 0.60},  # Favor keywords for names
    "lookup": {"dense": 0.40, "sparse": 0.60},
    "topic_search": {"dense": 0.80, "sparse": 0.20},  # Favor semantics for topics
    "procedure": {"dense": 0.60, "sparse": 0.40},
    "eligibility": {"dense": 0.60, "sparse": 0.40},
    "general": {"dense": 0.70, "sparse": 0.30}
}
```

**Score Fusion:**
1. Normalize dense scores to 0-1
2. Normalize sparse scores to 0-1
3. Apply intent-based weights
4. Optional name boost (30% for faculty queries)
5. Combine: `final_score = (dense * w_d) + (sparse * w_s) + (name * w_n)`
6. Normalize final scores to 0-1
7. Sort by final score, return top 20

**BM25 Persistence:**
- Saves index, corpus, IDs to disk (pickle)
- Checksum validation (SHA256)
- Loads on startup (~1s vs ~10s rebuild)
- Auto-rebuild on checksum mismatch

### Reranking

**BGEReranker** (`bge_reranker.py`):
- Model: BAAI/bge-reranker-v2-m3
- Cross-encoder scoring (query + chunk → relevance score)
- Processes 20 candidates → returns top 15
- Scores normalized to 0-1 range

### Special Retrieval Paths

**Direct Metadata Match** (faculty name queries):
- Bypasses vector search entirely
- Filters by `name_variants` field (exact match)
- Returns immediately if found
- Fallback to hybrid search if no match

**HyDE (Hypothetical Document Embedding)** (topic search):
- Generates hypothetical faculty description matching topic
- Embeds the hypothetical description instead of raw query
- Improves topic-based faculty search
- Example: "machine learning" → "Dr. X specializes in machine learning and neural networks..."

**Dual Embedding** (faculty lookup):
- Primary: Standard query embedding
- Secondary: Name-focused embedding ("Faculty: [Name]")
- 30% boost for name-based matches
- Improves precision for name queries

---

## 4. ANSWER GENERATION

### Dynamic Chunk Selection

**Token Budget Management:**
- Total context: 16K tokens (Gemma 3:12b)
- System prompt: ~500 tokens
- Output reserve: ~2000 tokens
- Available for chunks: ~13,800 tokens

**Selection Strategy:**
1. Sort chunks by relevance (already done by reranker)
2. Add chunks until token budget exhausted
3. Stop at intent-specific max chunks
4. Stop if relevance drops below 0.3 (after first 5 chunks)

**Intent-Based Chunk Limits:**
```python
INTENT_CHUNK_LIMITS = {
    "lookup": 8,           # Person lookup
    "person_lookup": 8,
    "department_list": 30, # Entire department
    "topic_search": 25,    # Broad topic
    "procedure": 20,       # Policy + forms + examples
    "eligibility": 15,     # Policy sections
    "salary_benefits": 15,
    "general": 20
}
```

### Structured Generation

**AnswerGenerator** (`answer_generator.py`):

**Prompt Strategy:**
- Intent-specific prompts (lookup, procedure, eligibility, etc.)
- Forces strict JSON output (Ollama `format="json"`)
- Temperature: 0.1 (consistency)
- Max tokens: 4096 (detailed responses)

**Output Schema:**
```json
{
    "intent": "lookup",
    "title": "Dr. Pragati Khare",
    "subtitle": "Associate Professor, Computer Science",
    "sections": [
        {
            "type": "paragraph" | "bullets" | "steps" | "alert",
            "heading": "Research Interests",
            "content": "...",           // for paragraph/alert
            "items": ["...", "..."],    // for bullets/steps
            "severity": "info"          // for alert
        }
    ],
    "footer": "Last updated: 2024",
    "confidence": "high" | "medium" | "low" | "none",
    "fallback": "Error message if generation failed"
}
```

**Schema Normalization:**
- Handles LLM creative naming (list→bullets, text→paragraph)
- Fixes field name mismatches (points→items, body→content)
- Infers missing fields (severity for alerts)
- Validates against Pydantic schema

**Fallback Handling:**
- JSON parse errors → clean fallback response
- Validation errors → clean fallback response
- Empty chunks → "No results found" response

---

## 5. CACHING & OPTIMIZATION

### Multi-Tier Caching

**CacheManager** (`cache_manager.py`):

**Tier 1: Memory (LRU)**
- Python `@lru_cache` decorator
- Fastest access
- Limited size

**Tier 2: Disk (diskcache)**
- Persistent across restarts
- 1GB size limit
- Medium speed

**Tier 3: Redis (optional)**
- Distributed caching
- Fast access
- Requires Redis server

**Cache Keys:**
- Query results: `query:{hash(query:top_k)}` (1h TTL)
- Embeddings: `emb:{model}:{hash(text)}` (24h TTL)

**Cache Hit Rate:**
- Identical queries return instantly
- Reduces LLM/embedding calls
- Improves response time by ~80% for cached queries

### BM25 Persistence

**BM25PersistenceManager** (`bm25_persistence.py`):
- Saves: BM25 index, tokenized corpus, document IDs
- Format: Pickle (binary)
- Checksum: SHA256 of corpus metadata
- Location: `./bm25_index/`
- Startup: Load from disk (~1s) vs rebuild (~10s)

### Async Operations

**FastAPI Async:**
- `asyncio.to_thread()` for blocking operations
- Non-blocking retrieval and generation
- Concurrent request handling
- Streaming responses via Server-Sent Events (SSE)

---

## 6. CONVERSATION MANAGEMENT

**ConversationManager** (`conversation_manager.py`):

**Features:**
- Session-based tracking (UUID)
- Persistent JSON storage (`./conversations/`)
- Max 10 messages in context (auto-trim)
- Message metadata (sources, intent, timestamp)

**Endpoints:**
- `POST /conversation/new` - Create session
- `GET /conversation/{id}` - Get history
- `DELETE /conversation/{id}` - Clear session
- `GET /conversations` - List all sessions

**Integration:**
- Query endpoint accepts `session_id` parameter
- Automatically adds user/assistant messages
- Context available for future multi-turn support

---

## 7. API ENDPOINTS

### Core Endpoints

**POST /query**
- Request: `{query, top_k?, session_id?, stream?}`
- Response: `{answer, sources, intent, chunks_used, metadata}`
- Streaming: SSE events if `stream=true`
- Caching: Automatic cache check/store

**POST /api/auth/signin**
- Demo authentication (replace in production)
- Role-based access (student, faculty, admin)
- Returns token + user info

**GET /health**
- System status
- Component health checks
- Cache statistics

### Conversation Endpoints

**POST /conversation/new** - Create session
**GET /conversation/{id}** - Get history
**DELETE /conversation/{id}** - Clear session
**GET /conversations** - List sessions

### Static Endpoints

**GET /** - Landing page
**GET /chat** - Chat interface
**GET /signin** - Sign-in page
**GET /static/** - Frontend assets

---

## 8. FRONTEND ARCHITECTURE

### Streaming UI

**chat.js:**
- Server-Sent Events (SSE) consumer
- Real-time progress updates
- Thinking block with step progression
- Structured response rendering

**Streaming Flow:**
1. Send query with `stream: true`
2. Show thinking block
3. Receive SSE events:
   - `status`: Update thinking step
   - `result`: Final response
   - `error`: Error message
   - `done`: Close stream
4. Render structured JSON response
5. Show sources and processing steps

**Session Management:**
- Auto-initialize session on load
- Persist session ID in memory
- New session on "New Chat" click

---

## 9. PERFORMANCE CHARACTERISTICS

### Latency Breakdown

**Cold Query (no cache):**
- Query understanding: ~50ms
- Embedding generation: ~100-200ms
- Hybrid search: ~200-300ms
- Reranking: ~300-500ms
- Answer generation: ~2-5s (LLM)
- **Total: ~3-6s**

**Cached Query:**
- Cache lookup: ~10-50ms
- **Total: ~50ms** (60x faster)

**Startup Time:**
- Without BM25 persistence: ~10-15s
- With BM25 persistence: ~1-2s (10x faster)

### Throughput

**Concurrent Requests:**
- Async FastAPI handles 100+ concurrent
- Bottleneck: LLM generation (sequential)
- Caching reduces LLM load by ~40-60%

**Embedding Throughput:**
- Batch embedding: ~50 chunks/second
- Individual embedding: ~10 chunks/second
- Dual encoder fallback: ~5 chunks/second

---

## 10. KNOWN ISSUES & LIMITATIONS

### Chunking Issues

1. **Two competing strategies** - Semantic vs Document chunker
2. **Arbitrary token limits** - Force splits that break meaning
3. **Weak section detection** - Regex misses nested structures
4. **No chunk relationships** - Parent-child links unused
5. **Naive overlap** - Fixed tokens, ignores semantic boundaries
6. **Faculty profile splits** - Bio separated from publications

### Retrieval Issues

1. **Title handling inconsistency** - Stripped in some places, not others
2. **Name matching fragility** - Depends on exact name_variants field
3. **No query reformulation** - Single-shot retrieval
4. **Limited metadata filtering** - Only domain + is_current
5. **No re-ranking diversity** - May return redundant chunks

### Generation Issues

1. **Schema normalization complexity** - LLM creative naming requires fixes
2. **No citation tracking** - Sources shown but not linked to specific claims
3. **No confidence scoring** - Confidence field unused
4. **Limited error recovery** - Fallback is generic message

### System Issues

1. **No authentication** - Demo auth only
2. **No rate limiting** - Open to abuse
3. **No monitoring** - No metrics/logging infrastructure
4. **No A/B testing** - Can't compare chunking strategies
5. **No evaluation** - No automated quality metrics

---

## 11. CONFIGURATION

### Environment Variables (.env)

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:12b

# Embedding
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Cache
REDIS_HOST=localhost
REDIS_PORT=6379
ENABLE_REDIS=false

# API
API_HOST=0.0.0.0
API_PORT=8001
```

### Chunking Config (chunking_config.py)

```python
MAX_LEVEL2_TOKENS = 980
OVERLAP_TOKENS = 50
TOP_K_INITIAL = 20
TOP_K_RERANKED = 15
INTENT_CHUNK_LIMITS = {...}
DEFAULT_CHUNK_LIMIT = 20
```

---

## 12. DEPLOYMENT

### Docker Compose

**Services:**
- Qdrant: Vector database (port 6333)
- Ollama: LLM server (port 11434)
- FastAPI: Application server (port 8001)

**Volumes:**
- `./qdrant_storage` - Vector DB persistence
- `./cache` - Disk cache
- `./bm25_index` - BM25 persistence
- `./conversations` - Conversation history

### Startup Sequence

1. Start Qdrant
2. Start Ollama, pull gemma3:12b
3. Initialize cache manager
4. Initialize conversation manager
5. Load embedding model (BAAI/bge-m3)
6. Create Qdrant collection
7. Load BM25 index from disk (or build)
8. Initialize retrieval pipeline
9. Initialize answer generator
10. Start FastAPI server

---

## 13. FUTURE IMPROVEMENTS

### High Priority

1. **Unified chunking strategy** - Single approach, hierarchical
2. **Semantic boundary detection** - Topic shift detection
3. **Contextual chunk enrichment** - Prepend hierarchy
4. **Adaptive chunk sizing** - Content-driven, not token-driven
5. **Query reformulation** - Multi-shot retrieval

### Medium Priority

1. **Citation tracking** - Link sources to specific claims
2. **Confidence scoring** - Actual confidence calculation
3. **Metadata expansion** - More filter dimensions
4. **Re-ranking diversity** - MMR or similar
5. **Evaluation framework** - Automated quality metrics

### Low Priority

1. **Multi-turn conversation** - Use conversation history
2. **User feedback loop** - Thumbs up/down
3. **A/B testing** - Compare strategies
4. **Monitoring dashboard** - Metrics visualization
5. **Production auth** - JWT, OAuth, etc.

---

**Last Updated:** 2026-03-26
**Version:** 0.1.0
