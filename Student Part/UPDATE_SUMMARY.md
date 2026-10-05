# Quick Update Summary

## Where to Update Embeddings

### 📁 Files to Update (2 files)

```
1. rag/retriever.py (Line 33)
   └─ self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
                                                 ↓ CHANGE THIS

2. ingest/index.py (Line 122)
   └─ embeddings = OllamaEmbeddings(model="nomic-embed-text")
                                            ↓ CHANGE THIS
```

---

## Easy Way (Recommended)

```bash
python switch_embeddings.py
```

**What it does**:
- ✅ Updates both files automatically
- ✅ Checks if model is available
- ✅ Offers to download if needed
- ✅ Reminds you to re-index

---

## Manual Way

### Step 1: Edit Files

**File 1**: `rag/retriever.py`
```python
# Line 33 - Change this:
self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

# To this:
self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

**File 2**: `ingest/index.py`
```python
# Line 122 - Change this:
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# To this:
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

### Step 2: Pull Model
```bash
ollama pull mxbai-embed-large
```

### Step 3: Re-index
```bash
python ingest/index.py
```

### Step 4: Restart App
```bash
python -m streamlit run app.py
```

---

## Why Both Files?

- **`rag/retriever.py`**: Used when **querying** (searching for documents)
- **`ingest/index.py`**: Used when **indexing** (creating embeddings)

Both must use the **same model** or it won't work!

---

## Recommended Models

| Model | Dimensions | Accuracy | Best For |
|-------|-----------|----------|----------|
| **mxbai-embed-large** ⭐ | 1024 | ⭐⭐⭐⭐⭐ | Academic content |
| nomic-embed-text (current) | 768 | ⭐⭐⭐⭐ | General use |
| all-minilm | 384 | ⭐⭐⭐ | Speed |

---

## Complete Upgrade Path

### Upgrade Both LLM and Embeddings

```bash
# 1. Pull models
ollama pull llama3.1:8b
ollama pull mxbai-embed-large

# 2. Switch LLM
python switch_model.py
# Select option 1 (Llama 3.1 8B)

# 3. Switch embeddings
python switch_embeddings.py
# Select option 1 (mxbai-embed-large)

# 4. Re-index (required for embeddings change)
python ingest/index.py

# 5. Restart app
python -m streamlit run app.py
```

**Expected improvements**:
- 10-15% better accuracy
- Faster responses
- More complete "list all" results
- Better understanding of technical terms

---

## Quick Reference

### Just LLM (No Re-index Needed)
```bash
python switch_model.py
```

### Just Embeddings (Re-index Required)
```bash
python switch_embeddings.py
python ingest/index.py
```

### Both (Re-index Required)
```bash
python switch_model.py
python switch_embeddings.py
python ingest/index.py
```

---

## Files Overview

```
Your Project/
├── rag/
│   ├── retriever.py          ← UPDATE LINE 33 (embeddings)
│   └── chain.py              ← UPDATE LINE 20 (LLM)
├── ingest/
│   └── index.py              ← UPDATE LINE 122 (embeddings)
├── switch_model.py           ← Run this to change LLM
├── switch_embeddings.py      ← Run this to change embeddings
└── app.py                    ← No changes needed
```

---

## Need Help?

See detailed guides:
- **LLM Models**: `MODEL_RECOMMENDATIONS.md`
- **Embeddings**: `EMBEDDING_UPDATE_GUIDE.md`
- **Quick Ref**: `QUICK_REFERENCE.md`
