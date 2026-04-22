# NMIMS Academic RAG System

A clean, simple, and accurate RAG (Retrieval-Augmented Generation) system for NMIMS university academic documents. Query syllabus and question papers using natural language.

---

## Features

- ✅ **Markdown-based**: Simple markdown files (no PDF/OCR complexity)
- ✅ **Accurate Retrieval**: Semantic search with intelligent reranking
- ✅ **Smart Chunking**: Preserves course structure, units, and questions
- ✅ **API-Ready**: Backend RAG system ready for integration
- ✅ **Fully Local**: Runs entirely on your machine using Ollama

---

## Quick Start

### 1. Install Ollama

Download from: https://ollama.com

Pull required models:
```bash
# Current models
ollama pull gemma3:12b
ollama pull bge-m3

# OR alternative models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

**Recommended**: Use Gemma 3 12B for best accuracy and bge-m3 for embeddings!
See `MODEL_RECOMMENDATIONS.md` for details.

Start Ollama server:
```bash
ollama serve
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Your Documents Are Ready!

Your markdown files are already in place:
- ✅ `data/syllabus/SVKM_NMIMS_Complete_Syllabus.md`
- ✅ `data/question_papers/Machine_Learning_Exam_Paper.md`
- ✅ `data/question_papers/Cyber_Security_Exam_Paper.md`

### 4. Index Documents

```bash
python ingest/index.py
```

Expected output:
```
✅ Successfully indexed 109 chunks!

📊 Indexing Summary
Documents:        3
Chunks:           109
Embedding Dim:    1024
Collection:       academic_rag
```

### 5. Test the System

```bash
# Test retrieval
python rag/retriever.py "What topics are covered in Unit 2 of Cyber Security?"

# Test RAG chain
python rag/chain.py "What topics are covered in Unit 2 of Cyber Security?"
```

---

## Usage

### Command Line Testing

**Test Retrieval:**
```bash
python rag/retriever.py "What topics are covered in Unit 2 of Cyber Security?"
```

**Test RAG Chain:**
```bash
python rag/chain.py "List all course outcomes for Machine Learning"
```

### Sample Queries

**Syllabus Queries:**
- "What topics are covered in Unit 2 of Cyber Security?"
- "List all course outcomes for Machine Learning"
- "What is the evaluation scheme for Cyber Security?"
- "Explain the cryptography topics in the syllabus"

**Question Paper Queries:**
- "Show me Question 2.A from the Machine Learning exam"
- "What questions were asked about gradient descent?"
- "Find all 10-mark questions"
- "What CO-2 questions appeared in the exam?"

---

## Project Structure

```
nmims-rag/
├── ingest/
│   ├── extract.py          # Markdown extraction
│   ├── chunker.py          # Semantic chunking
│   └── index.py            # Indexing pipeline
├── rag/
│   ├── retriever.py        # Document retrieval
│   └── chain.py            # RAG chain
├── data/
│   ├── syllabus/           # Syllabus markdown files
│   └── question_papers/    # Question paper markdown files
├── scripts/                # Utility scripts
└── requirements.txt        # Python dependencies
```

---

## How It Works

### 1. Extraction (`ingest/extract.py`)
- Reads markdown files
- Extracts metadata (course codes, credits, semesters, etc.)
- Auto-detects document type (syllabus vs question paper)

### 2. Chunking (`ingest/chunker.py`)
- **Syllabus**: Chunks by courses and units
- **Question Papers**: Chunks by individual questions
- Preserves structure and context

### 3. Indexing (`ingest/index.py`)
- Generates embeddings using Ollama (nomic-embed-text)
- Stores in Qdrant vector database
- Enriches with metadata (COs, SOs, units, etc.)

### 4. Retrieval (`rag/retriever.py`)
- Analyzes query (extracts units, questions, COs, etc.)
- Semantic search using vector similarity
- Reranks results based on entity matching

### 5. Generation (`rag/chain.py`)
- Formats retrieved context
- Generates answer using LLM (qwen2.5:14b)
- Returns accurate, context-based response

---

## Adding New Documents

### 1. Add Markdown Files

Place new files in:
- `data/syllabus/` - for syllabus documents
- `data/question_papers/` - for question papers

### 2. Re-index

```bash
python ingest/index.py
```

---

## Python API Usage

### Basic Usage

```python
from rag.chain import build_rag_chain

# Initialize RAG chain
chain = build_rag_chain(k=8)

# Query
answer = chain("What topics are covered in Unit 2 of Cyber Security?")
print(answer)
```

### Advanced Usage

```python
from rag.retriever import get_retriever
from rag.chain import AcademicRAG

# Custom retriever
retriever = get_retriever(
    store_path="./qdrant_db",
    collection_name="academic_rag",
    k=10
)

# Custom RAG chain
chain = AcademicRAG(
    store_path="./qdrant_db",
    collection_name="academic_rag",
    llm_model="gemma3:12b",
    k=8
)

# Query
answer = chain("List all course outcomes for Machine Learning")
print(answer)
```

---

## Configuration

### Change LLM Model

**Easy way** (recommended):
```bash
python switch_model.py
```

**Manual way** - Edit `rag/chain.py`:
```python
llm_model="gemma3:12b"  # Current default (best accuracy)
# or
llm_model="llama3.1:8b"   # Alternative
# or
llm_model="mistral:7b"   # Faster
# or
llm_model="qwen2.5:14b"    # Larger model
```

See `MODEL_RECOMMENDATIONS.md` for detailed comparison.

### Change Embeddings

Edit `rag/retriever.py` and `ingest/index.py`:
```python
embeddings = OllamaEmbeddings(model="bge-m3")  # Current default (1024-dim)
# or
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Alternative (768-dim)
```

Then re-index:
```bash
python ingest/index.py
```

### Adjust Retrieval

Edit `rag/chain.py` or pass parameter:
```python
chain = build_rag_chain(k=5)  # Retrieve fewer documents
```

### Change Chunk Sizes

Edit `ingest/chunker.py`:
```python
chunk_by_size(doc, chunk_size=1000, overlap=100)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Ollama connection error** | Run `ollama serve` in a terminal |
| **Model not found** | `ollama pull gemma3:12b` and `ollama pull bge-m3` |
| **No documents found** | Add markdown files to `data/syllabus/` or `data/question_papers/` |
| **Empty responses** | Re-run indexing: `python ingest/index.py` |
| **Out of memory** | Use smaller model: `llama3.1:8b` or `mistral:7b` |

---

## System Requirements

- **Python**: 3.9+
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB for models and data
- **OS**: Windows, Linux, or macOS

---

## Performance

- **Indexing**: ~30 seconds for 3 documents
- **Query Time**: 3-6 seconds per question
- **Accuracy**: ~85-90% retrieval accuracy
- **Memory**: ~10-12 GB during operation

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | gemma3:12b (Ollama) |
| Embeddings | bge-m3 (Ollama) |
| Vector DB | Qdrant (local) |
| Framework | LangChain |
| Interface | Python API |
| Language | Python 3.9+ |

---

## License

MIT License

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review `VALIDATION_RESULTS.md` for test results
3. Open a GitHub issue

---

## What's Different from v1?

- ✅ **Removed**: All PDF/OCR code (PyMuPDF, EasyOCR)
- ✅ **Simplified**: Clean, focused codebase
- ✅ **Optimized**: Better chunking for NMIMS format
- ✅ **Tested**: Validated with your actual files
- ✅ **Faster**: No OCR processing needed

---

**Ready to use!** Just start Ollama and run the indexing command.
