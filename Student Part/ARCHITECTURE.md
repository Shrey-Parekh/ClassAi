# System Architecture

## Overview
This document explains the architecture of the enhanced RAG system, component interactions, and design decisions.

---

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
│                         app.py                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Chain Layer                           │
│              rag/improved_chain.py                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Query processing                                    │   │
│  │ • Context formatting                                  │   │
│  │ • LLM invocation                                      │   │
│  │ • Response generation                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Retrieval Layer                             │
│            rag/advanced_retriever.py                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Query Analyzer                                        │   │
│  │ • Intent detection                                    │   │
│  │ • Entity extraction (units, COs, questions)          │   │
│  │ • Filter generation                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Hybrid Retriever                                      │   │
│  │ • Vector search (semantic)                            │   │
│  │ • Metadata filtering                                  │   │
│  │ • Multi-signal reranking                              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vector Store Layer                          │
│                    Qdrant Client                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Vector storage (COSINE similarity)                  │   │
│  │ • Metadata indexing                                   │   │
│  │ • Efficient search                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Ingestion Pipeline                          │
│           ingest/index_documents.py                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Extract (extract_markdown.py)                      │   │
│  │    • Parse markdown files                             │   │
│  │    • Extract frontmatter metadata                     │   │
│  │    • Detect document type                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. Chunk (advanced_chunker.py)                        │   │
│  │    • Semantic-aware chunking                          │   │
│  │    • Structure preservation                           │   │
│  │    • Type-specific strategies                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. Enrich                                             │   │
│  │    • Extract keywords                                 │   │
│  │    • Identify references (CO, SO, units)             │   │
│  │    • Compute metadata                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 4. Embed (Ollama)                                     │   │
│  │    • Generate 768-dim vectors                         │   │
│  │    • Batch processing                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 5. Index (Qdrant)                                     │   │
│  │    • Store vectors                                    │   │
│  │    • Index metadata                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  External Services                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Ollama (Local)                                        │   │
│  │ • LLM: gemma3:12b                                     │   │
│  │ • Embeddings: bge-m3                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Indexing Flow
```
Markdown Files
    ↓
[Extract Metadata & Content]
    ↓
Documents with Rich Metadata
    ↓
[Semantic Chunking]
    ↓
Structured Chunks
    ↓
[Metadata Enrichment]
    ↓
Enriched Chunks
    ↓
[Embedding Generation]
    ↓
Vector Embeddings (768-dim)
    ↓
[Store in Qdrant]
    ↓
Indexed Vector Database
```

### Query Flow
```
User Query
    ↓
[Query Analysis]
    ├─ Intent Detection
    ├─ Entity Extraction
    └─ Filter Generation
    ↓
Analyzed Query + Filters
    ↓
[Vector Search]
    ├─ Semantic Search (COSINE)
    ├─ Metadata Filtering
    └─ Top-k Retrieval
    ↓
Retrieved Documents (k*2)
    ↓
[Multi-Signal Reranking]
    ├─ Keyword Matching
    ├─ Entity Matching
    ├─ Intent Alignment
    └─ Metadata Relevance
    ↓
Top-k Reranked Documents
    ↓
[Context Formatting]
    ↓
Formatted Context + Citations
    ↓
[LLM Generation]
    ↓
Final Answer with Sources
```

---

## Component Details

### 1. Markdown Extraction (`extract_markdown.py`)

**Purpose**: Parse markdown files and extract rich metadata

**Key Functions**:
- `extract_metadata_from_frontmatter()`: Parse YAML frontmatter
- `extract_syllabus_metadata()`: Extract syllabus-specific metadata
- `extract_qp_metadata()`: Extract question paper metadata
- `extract_markdown_document()`: Main extraction function

**Metadata Extracted**:
- **Syllabus**: subject, course_code, credits, semester, units, COs, SOs
- **Question Papers**: subject, exam_type, batch, year, marks, duration, questions

**Design Decisions**:
- Use frontmatter for explicit metadata
- Regex patterns for implicit metadata extraction
- Auto-detection of document type
- Graceful error handling

### 2. Advanced Chunking (`advanced_chunker.py`)

**Purpose**: Split documents while preserving semantic structure

**Chunkers**:

#### SyllabusChunker
- **Strategy**: Preserve complete units as chunks
- **Chunk Size**: 1200 chars (larger for context)
- **Features**:
  - Markdown header-based splitting
  - Unit boundary detection
  - Course outcomes extraction
  - Sub-chunking for large units

#### QuestionPaperChunker
- **Strategy**: Extract individual questions
- **Chunk Size**: 600 chars (smaller, focused)
- **Features**:
  - Question number detection
  - Marks extraction
  - CO/SO mapping
  - Section boundary detection

#### HybridChunker
- **Strategy**: Route to appropriate chunker
- **Logic**: Based on document type metadata

**Design Decisions**:
- Semantic boundaries over fixed sizes
- Preserve context with overlap
- Type-specific strategies
- Fallback to recursive splitting

### 3. Hybrid Retrieval (`advanced_retriever.py`)

**Purpose**: Retrieve most relevant documents using multiple signals

**Components**:

#### QueryAnalyzer
- **Intent Detection**: Classify query type (syllabus_unit, question_paper, etc.)
- **Entity Extraction**: Extract units, COs, SOs, questions
- **Filter Generation**: Create metadata filters

**Intents**:
- `syllabus_unit`: Queries about course units
- `question_paper`: Queries about exam questions
- `course_outcomes`: Queries about COs/SOs
- `evaluation_scheme`: Queries about grading
- `general`: Other queries

#### HybridRetriever
- **Vector Search**: Semantic similarity (COSINE)
- **Metadata Filtering**: Filter by type, subject, etc.
- **Reranking**: Multi-signal scoring

**Reranking Signals**:
1. **Keyword Matching** (weight: 2): Basic term overlap
2. **Entity Matching** (weight: 15-25): Units, COs, SOs, questions
3. **Intent Alignment** (weight: 10-15): Chunk type matches intent
4. **Metadata Relevance** (weight: 5-12): Subject, year, structure
5. **Content Quality** (weight: 2-3): Length, structure type

**Design Decisions**:
- Retrieve k*2 documents for reranking
- Entity matching gets highest boost
- Recent documents preferred
- Structured chunks preferred

### 4. RAG Chain (`improved_chain.py`)

**Purpose**: Orchestrate retrieval and generation

**Components**:

#### ImprovedRAGChain
- **Retrieval**: Uses HybridRetriever
- **Context Formatting**: Rich citations with metadata
- **Prompt Engineering**: Academic-specific instructions
- **LLM Invocation**: gemma3:12b with 8192 context
- **Response Generation**: Structured answers

**Prompt Design**:
- Clear role definition (academic assistant)
- Explicit instructions for different query types
- Citation requirements
- Fallback responses
- Academic tone guidelines

**Context Formatting**:
```
[Document 1]
Source: file.md | Subject: ML | Type: syllabus | Unit: 1

<content>

[Document 2]
...
```

**Design Decisions**:
- Large context window (8192) for comprehensive answers
- Low temperature (0.1) for factual responses
- Explicit citation in context
- Structured prompt with examples

### 5. Indexing Pipeline (`index_documents.py`)

**Purpose**: End-to-end document processing and indexing

**Steps**:
1. **Extract**: Parse markdown files
2. **Chunk**: Apply semantic chunking
3. **Enrich**: Add computed metadata
4. **Embed**: Generate vectors
5. **Index**: Store in Qdrant

**Features**:
- Progress tracking
- Batch processing
- Error handling
- Statistics reporting
- Incremental updates (--no-recreate)

**Design Decisions**:
- Modular pipeline (easy to extend)
- Clear progress feedback
- Comprehensive error messages
- Configurable parameters

---

## Design Principles

### 1. Modularity
- Each component has single responsibility
- Clear interfaces between components
- Easy to test and extend

### 2. Accuracy First
- Semantic-aware chunking
- Multi-signal reranking
- Rich metadata utilization
- Context preservation

### 3. User Experience
- Clear progress feedback
- Helpful error messages
- Comprehensive documentation
- Sample files for testing

### 4. Performance
- Batch processing
- Efficient vector search
- Caching (Qdrant client singleton)
- Optimized chunk sizes

### 5. Maintainability
- Well-documented code
- Type hints where appropriate
- Consistent naming conventions
- Modular architecture

---

## Configuration Points

### Chunking
```python
# ingest/advanced_chunker.py
SyllabusChunker(chunk_size=1200, chunk_overlap=100)
QuestionPaperChunker(chunk_size=600, chunk_overlap=80)
```

### Retrieval
```python
# rag/advanced_retriever.py
HybridRetriever(k=10, use_reranking=True)
```

### RAG Chain
```python
# rag/improved_chain.py
ImprovedRAGChain(
    llm_model="gemma3:12b",
    temperature=0.1,
    k=8
)
```

### Embeddings
```python
# Throughout
OllamaEmbeddings(model="bge-m3")
```

---

## Scalability Considerations

### Current Limits
- **Documents**: ~1000 documents
- **Chunks**: ~10,000 chunks
- **Concurrent Users**: 1-5
- **Memory**: 10-12 GB

### Scaling Strategies

#### Horizontal Scaling
1. **Qdrant Cloud**: Distributed vector storage
2. **Load Balancer**: Multiple Streamlit instances
3. **Redis Cache**: Cache frequent queries
4. **API Gateway**: Rate limiting, auth

#### Vertical Scaling
1. **GPU**: Faster embeddings and LLM
2. **More RAM**: Larger models, more cache
3. **SSD**: Faster Qdrant operations

#### Optimization
1. **Hybrid Search**: Add BM25 for keyword queries
2. **Query Cache**: Cache common queries
3. **Batch Embeddings**: Process multiple queries
4. **Model Quantization**: Smaller models

---

## Security Considerations

### Current State
- Local deployment (no network exposure)
- No authentication
- No data encryption at rest

### Production Recommendations
1. **Authentication**: Add user auth (OAuth, JWT)
2. **Authorization**: Role-based access control
3. **Encryption**: Encrypt sensitive data
4. **Input Validation**: Sanitize user inputs
5. **Rate Limiting**: Prevent abuse
6. **Audit Logging**: Track access and changes
7. **HTTPS**: Secure communication

---

## Monitoring and Observability

### Current State
- Basic logging to console
- No metrics collection
- No tracing

### Recommended Additions
1. **LangFuse**: LLM observability
2. **Prometheus**: Metrics collection
3. **Grafana**: Visualization
4. **Sentry**: Error tracking
5. **ELK Stack**: Log aggregation

### Key Metrics to Track
- Query latency (p50, p95, p99)
- Retrieval accuracy
- LLM token usage
- Error rates
- User satisfaction (feedback)

---

## Testing Strategy

### Unit Tests
- Test each component independently
- Mock external dependencies
- Cover edge cases

### Integration Tests
- Test component interactions
- Use test database
- Validate end-to-end flow

### System Tests
- Test complete system
- Use real data
- Validate performance

### Current Testing
- `test_system.py`: System validation
- Manual testing via Streamlit UI

### Recommended Additions
- Pytest test suite
- CI/CD pipeline
- Automated regression tests
- Performance benchmarks

---

## Future Architecture

### Microservices Architecture
```
┌─────────────┐
│   Web UI    │
└──────┬──────┘
       │
┌──────▼──────┐
│  API Gateway│
└──────┬──────┘
       │
   ┌───┴───┬───────┬────────┐
   │       │       │        │
┌──▼──┐ ┌─▼──┐ ┌──▼───┐ ┌──▼────┐
│Query│ │Ret-│ │Embed-│ │Index- │
│Svc  │ │riev│ │ding  │ │ing    │
│     │ │al  │ │Svc   │ │Svc    │
└─────┘ └────┘ └──────┘ └───────┘
```

### Benefits
- Independent scaling
- Technology flexibility
- Fault isolation
- Easier maintenance

---

## Conclusion

This architecture prioritizes:
1. **Accuracy**: Through semantic chunking and hybrid retrieval
2. **Usability**: Clear interfaces and documentation
3. **Maintainability**: Modular design and clean code
4. **Scalability**: Ready for production deployment

The system is production-ready for small to medium deployments and can be scaled as needed.
