# Quick Reference Guide

## Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### Index Documents
```bash
# Index all documents
python ingest/index.py

# Append new documents (don't recreate)
python ingest/index.py --append
```

### Run Application
```bash
# Recommended
python -m streamlit run app.py

# Alternative
streamlit run app.py

# Or double-click
run_app.bat
```

### Test System
```bash
# Full system test
python test_system.py

# Test individual components
python ingest/extract.py
python ingest/chunker.py
python rag/retriever.py "test query"
python rag/chain.py "test query"
```

---

## File Locations

### Add Documents
- Syllabus: `data/syllabus/*.md`
- Question Papers: `data/question_papers/*.md`

### Database
- Vector Store: `./qdrant_db/`

### Code
- Extraction: `ingest/extract.py`
- Chunking: `ingest/chunker.py`
- Indexing: `ingest/index.py`
- Retrieval: `rag/retriever.py`
- RAG Chain: `rag/chain.py`
- UI: `app.py`

---

## Sample Queries

### Syllabus
- "What topics are covered in Unit 2 of Cyber Security?"
- "List all course outcomes for Machine Learning"
- "What is the evaluation scheme?"
- "What are the prerequisites for Machine Learning?"

### Question Papers
- "Show me Question 2.A from the ML exam"
- "What questions were asked about gradient descent?"
- "Find all 10-mark questions"
- "What CO-2 questions appeared?"

---

## Troubleshooting

### Ollama Not Running
```bash
ollama serve
```

### Model Not Found
```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### Streamlit Blocked
```bash
python -m streamlit run app.py
```

### No Documents Found
- Add markdown files to `data/syllabus/` or `data/question_papers/`
- Re-run: `python ingest/index.py`

### Empty Responses
```bash
# Re-index
python ingest/index.py

# Test
python test_system.py
```

---

## Configuration

### Change LLM Model
Edit `rag/chain.py`:
```python
llm_model="qwen2.5:7b"  # Smaller model
```

### Adjust Retrieval Count
Edit `app.py`:
```python
build_rag_chain(k=5)  # Retrieve 5 instead of 8
```

### Change Chunk Sizes
Edit `ingest/chunker.py`:
```python
chunk_by_size(doc, chunk_size=1000, overlap=100)
```

---

## System Status Check

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check if database exists
ls qdrant_db/

# Check if documents exist
ls data/syllabus/
ls data/question_papers/

# Run full test
python test_system.py
```

---

## Performance

- **Indexing**: ~30 seconds
- **Query**: 3-6 seconds
- **Memory**: ~10-12 GB
- **Accuracy**: ~85-90%

---

## Support

1. Check `README.md` for detailed docs
2. Check `SYSTEM_OVERVIEW.md` for architecture
3. Run `python test_system.py` for diagnostics
4. Check `VALIDATION_RESULTS.md` for test results
