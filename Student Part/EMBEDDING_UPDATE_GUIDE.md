# How to Update Embedding Model

## Quick Method (Recommended)

```bash
python switch_embeddings.py
```

This script will:
1. Let you choose a new embedding model
2. Automatically update all 3 files
3. Remind you to re-index

---

## Manual Method

If you prefer to update manually, here's exactly where to change:

### File 1: `rag/retriever.py`

**Location**: Line ~33

**Find this**:
```python
self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

**Change to**:
```python
self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

---

### File 2: `ingest/index.py`

**Location**: Line ~122

**Find this**:
```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

**Change to**:
```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

**Also update the print statement** (Line ~125):

**Find this**:
```python
print(f"✓ Model: nomic-embed-text")
```

**Change to**:
```python
print(f"✓ Model: mxbai-embed-large")
```

**And the error message** (Line ~132):

**Find this**:
```python
print("   2. Pull model: ollama pull nomic-embed-text")
```

**Change to**:
```python
print("   2. Pull model: ollama pull mxbai-embed-large")
```

---

## After Updating

### Step 1: Pull the New Model
```bash
ollama pull mxbai-embed-large
```

### Step 2: Delete Old Database
```bash
# Windows
Remove-Item -Recurse -Force qdrant_db

# Linux/Mac
rm -rf qdrant_db
```

### Step 3: Re-index Documents
```bash
python ingest/index.py
```

This will take ~30 seconds and create new embeddings with the new model.

### Step 4: Restart App
```bash
python -m streamlit run app.py
```

---

## Why Re-index?

Different embedding models create different vector representations:
- **nomic-embed-text**: 768-dimensional vectors
- **mxbai-embed-large**: 1024-dimensional vectors
- **all-minilm**: 384-dimensional vectors

The old vectors (768-dim) won't work with a new model (1024-dim). You must re-create them.

---

## Recommended Embedding Models

### 1. mxbai-embed-large (Recommended)
```bash
ollama pull mxbai-embed-large
```
- **Dimensions**: 1024
- **Accuracy**: ⭐⭐⭐⭐⭐
- **Speed**: ⭐⭐⭐⭐
- **Memory**: ~1 GB
- **Best for**: Academic content, technical terms

### 2. nomic-embed-text (Current)
```bash
ollama pull nomic-embed-text
```
- **Dimensions**: 768
- **Accuracy**: ⭐⭐⭐⭐
- **Speed**: ⭐⭐⭐⭐
- **Memory**: ~0.8 GB
- **Best for**: General use

### 3. all-minilm (Fastest)
```bash
ollama pull all-minilm
```
- **Dimensions**: 384
- **Accuracy**: ⭐⭐⭐
- **Speed**: ⭐⭐⭐⭐⭐
- **Memory**: ~0.5 GB
- **Best for**: Speed, low memory

---

## Complete Example

```bash
# 1. Pull new embedding model
ollama pull mxbai-embed-large

# 2. Switch embeddings (automatic)
python switch_embeddings.py
# Select option 1

# 3. Re-index
python ingest/index.py

# 4. Restart app
python -m streamlit run app.py
```

---

## Verification

After re-indexing, check the output:

```
✓ Model: mxbai-embed-large
✓ Dimension: 1024
```

If you see 1024 dimensions, the new model is working!

---

## Troubleshooting

### Error: "Model not found"
```bash
ollama pull mxbai-embed-large
```

### Error: "Dimension mismatch"
You forgot to re-index. Delete the database and re-index:
```bash
Remove-Item -Recurse -Force qdrant_db
python ingest/index.py
```

### Error: "Database locked"
Close all Python scripts and restart your terminal.

---

## Summary

**3 files to update**:
1. `rag/retriever.py` (line ~33)
2. `ingest/index.py` (line ~122)

**Easy way**: `python switch_embeddings.py`

**Must do after**: Re-index with `python ingest/index.py`

**Expected improvement**: 10-15% better retrieval accuracy with mxbai-embed-large!
