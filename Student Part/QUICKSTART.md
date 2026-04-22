# Quick Start Guide

Get your RAG system running in 5 minutes!

## Prerequisites

- Python 3.9+
- 16GB+ RAM (32GB recommended)
- Ollama installed

## Step-by-Step Setup

### 1. Install Ollama Models (5 minutes)

```bash
# Install Ollama from https://ollama.com
# Then pull models:
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# Start Ollama server
ollama serve
```

### 2. Setup Python Environment (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add Your Documents (1 minute)

Create markdown files in:
- `data/syllabus/` - for syllabus documents
- `data/question_papers/` - for question papers

**Example syllabus** (`data/syllabus/ml.md`):

```markdown
# Machine Learning

## Unit 1: Introduction
- ML fundamentals
- Types of learning

## Unit 2: Supervised Learning
- Linear regression
- Logistic regression

## Course Outcomes
- CO-1: Understand ML basics
- CO-2: Implement algorithms
```

**Example question paper** (`data/question_papers/ml_final.md`):

```markdown
# Machine Learning - Final Exam

## Q1. Explain supervised learning. [10 marks]
**Unit 1** | **CO-1**

## Q2. Implement linear regression. [15 marks]
**Unit 2** | **CO-2**
```

### 4. Index Documents (1 minute)

```bash
python ingest/index_documents.py
```

Wait for completion message:
```
✅ Indexing pipeline completed successfully!
```

### 5. Run the App (30 seconds)

```bash
streamlit run app.py
```

Open browser: http://localhost:8501

## Test Queries

Try these queries in the app:

1. "What topics are covered in Unit 1?"
2. "Show me Question 1 from the exam"
3. "List all course outcomes"
4. "What is covered in Machine Learning?"

## Troubleshooting

### Ollama not running
```bash
# Start Ollama
ollama serve
```

### Model not found
```bash
# Pull models
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### No documents found
- Check files are in `data/syllabus/` and `data/question_papers/`
- Ensure files have `.md` extension
- Re-run indexing: `python ingest/index_documents.py`

### Out of memory
- Use smaller model: Edit `rag/improved_chain.py` and change to `qwen2.5:7b`
- Reduce retrieval: Edit `app.py` and change `k=8` to `k=5`

## Next Steps

- Add more documents to `data/` folders
- Re-index: `python ingest/index_documents.py`
- Customize prompts in `rag/improved_chain.py`
- Adjust chunking in `ingest/advanced_chunker.py`

## Need Help?

- Check README.md for detailed documentation
- Review troubleshooting section
- Open a GitHub issue
