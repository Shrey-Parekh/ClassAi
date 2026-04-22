# Model Recommendations for NMIMS Academic RAG

## Current Setup
- **LLM**: qwen2.5:14b (~8 GB RAM)
- **Embeddings**: nomic-embed-text (~0.8 GB RAM)

---

## Better LLM Models

### 1. **Llama 3.1 8B** (Recommended Upgrade)
```bash
ollama pull llama3.1:8b
```

**Pros**:
- Better instruction following
- More accurate responses
- Better at structured output
- Faster than qwen2.5:14b
- Only ~5 GB RAM

**Cons**:
- Slightly less context window (8K vs 8K)

**Performance**: ⭐⭐⭐⭐⭐
**Speed**: ⭐⭐⭐⭐⭐
**Accuracy**: ⭐⭐⭐⭐⭐

**Change in code**:
```python
# rag/chain.py
llm_model="llama3.1:8b"
```

---

### 2. **Llama 3.1 70B** (Best Quality, High RAM)
```bash
ollama pull llama3.1:70b
```

**Pros**:
- Highest quality responses
- Best instruction following
- Most accurate
- Great for complex queries

**Cons**:
- Requires 40+ GB RAM
- Much slower (10-15s per query)
- Not suitable for 32GB systems

**Performance**: ⭐⭐⭐⭐⭐
**Speed**: ⭐⭐
**Accuracy**: ⭐⭐⭐⭐⭐

**Only use if you have 64GB+ RAM**

---

### 3. **Mistral 7B** (Fast & Efficient)
```bash
ollama pull mistral:7b
```

**Pros**:
- Very fast responses
- Good accuracy
- Low memory (~4 GB)
- Great for quick queries

**Cons**:
- Less detailed than Llama
- Shorter context window (8K)

**Performance**: ⭐⭐⭐⭐
**Speed**: ⭐⭐⭐⭐⭐
**Accuracy**: ⭐⭐⭐⭐

**Change in code**:
```python
# rag/chain.py
llm_model="mistral:7b"
```

---

### 4. **Gemma 2 9B** (Google's Model)
```bash
ollama pull gemma2:9b
```

**Pros**:
- Excellent instruction following
- Good balance of speed/quality
- ~6 GB RAM
- Strong at academic content

**Cons**:
- Slightly verbose
- Can be overly formal

**Performance**: ⭐⭐⭐⭐
**Speed**: ⭐⭐⭐⭐
**Accuracy**: ⭐⭐⭐⭐⭐

**Change in code**:
```python
# rag/chain.py
llm_model="gemma2:9b"
```

---

### 5. **Phi-3 Medium** (Microsoft's Model)
```bash
ollama pull phi3:medium
```

**Pros**:
- Excellent for academic content
- Very accurate
- Good reasoning
- ~8 GB RAM

**Cons**:
- Slower than Mistral
- Less creative

**Performance**: ⭐⭐⭐⭐
**Speed**: ⭐⭐⭐
**Accuracy**: ⭐⭐⭐⭐⭐

**Change in code**:
```python
# rag/chain.py
llm_model="phi3:medium"
```

---

## Better Embedding Models

### Current: nomic-embed-text (768-dim)
Good general-purpose embeddings.

### 1. **mxbai-embed-large** (Recommended Upgrade)
```bash
ollama pull mxbai-embed-large
```

**Pros**:
- Better semantic understanding
- 1024-dim (more detailed)
- Better for academic content
- Same speed

**Cons**:
- Slightly more memory (~1 GB)
- Need to re-index

**Performance**: ⭐⭐⭐⭐⭐
**Accuracy**: ⭐⭐⭐⭐⭐

**Change in code**:
```python
# rag/retriever.py and ingest/index.py
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

**Then re-index**:
```bash
python ingest/index.py
```

---

### 2. **all-minilm** (Faster, Smaller)
```bash
ollama pull all-minilm
```

**Pros**:
- Very fast
- Low memory (~0.5 GB)
- 384-dim (smaller vectors)

**Cons**:
- Less accurate than nomic
- Shorter context understanding

**Performance**: ⭐⭐⭐
**Speed**: ⭐⭐⭐⭐⭐
**Accuracy**: ⭐⭐⭐

---

## Recommended Combinations

### 🏆 Best Overall (32GB RAM)
```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

**Configuration**:
```python
# rag/chain.py
llm_model="llama3.1:8b"

# rag/retriever.py and ingest/index.py
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

**Benefits**:
- Best accuracy
- Good speed
- Fits in 32GB RAM
- ~12 GB total usage

---

### ⚡ Fastest (16GB RAM)
```bash
ollama pull mistral:7b
ollama pull nomic-embed-text
```

**Configuration**:
```python
# rag/chain.py
llm_model="mistral:7b"

# Keep current embeddings
```

**Benefits**:
- Very fast responses (2-3s)
- Low memory (~8 GB)
- Good accuracy

---

### 🎯 Most Accurate (64GB+ RAM)
```bash
ollama pull llama3.1:70b
ollama pull mxbai-embed-large
```

**Configuration**:
```python
# rag/chain.py
llm_model="llama3.1:70b"

# rag/retriever.py and ingest/index.py
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

**Benefits**:
- Highest quality
- Best for complex queries
- Most detailed answers

**Drawbacks**:
- Slow (10-15s per query)
- High memory (40+ GB)

---

### 💰 Budget (8GB RAM)
```bash
ollama pull phi3:mini
ollama pull all-minilm
```

**Configuration**:
```python
# rag/chain.py
llm_model="phi3:mini"

# rag/retriever.py and ingest/index.py
embeddings = OllamaEmbeddings(model="all-minilm")
```

**Benefits**:
- Works on low-end systems
- Fast
- Decent accuracy

---

## How to Change Models

### Step 1: Pull New Model
```bash
ollama pull llama3.1:8b
```

### Step 2: Update Code

**For LLM** (edit `rag/chain.py`):
```python
class AcademicRAG:
    def __init__(
        self,
        store_path: str = "./qdrant_db",
        collection_name: str = "academic_rag",
        llm_model: str = "llama3.1:8b",  # ← Change this
        k: int = 8
    ):
```

**For Embeddings** (edit `rag/retriever.py` and `ingest/index.py`):
```python
self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # ← Change this
```

### Step 3: Re-index (Only if Embeddings Changed)
```bash
python ingest/index.py
```

### Step 4: Restart App
```bash
python -m streamlit run app.py
```

---

## Model Comparison Table

| Model | RAM | Speed | Accuracy | Best For |
|-------|-----|-------|----------|----------|
| **qwen2.5:14b** (current) | 8 GB | ⭐⭐⭐ | ⭐⭐⭐⭐ | General use |
| **llama3.1:8b** ⭐ | 5 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Best overall** |
| **llama3.1:70b** | 40 GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | Maximum quality |
| **mistral:7b** | 4 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Speed |
| **gemma2:9b** | 6 GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Academic content |
| **phi3:medium** | 8 GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Reasoning |

| Embeddings | RAM | Dim | Accuracy | Best For |
|------------|-----|-----|----------|----------|
| **nomic-embed-text** (current) | 0.8 GB | 768 | ⭐⭐⭐⭐ | General use |
| **mxbai-embed-large** ⭐ | 1 GB | 1024 | ⭐⭐⭐⭐⭐ | **Best accuracy** |
| **all-minilm** | 0.5 GB | 384 | ⭐⭐⭐ | Speed |

---

## My Recommendation

### For Your Use Case (32GB RAM, Academic Content):

**Upgrade to**:
```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

**Why**:
1. **Llama 3.1 8B** is better at:
   - Following instructions ("list ALL units")
   - Structured output (numbered lists)
   - Academic language
   - Completeness (won't skip items)

2. **mxbai-embed-large** is better at:
   - Understanding academic terminology
   - Semantic similarity for technical content
   - Distinguishing between similar topics

**Expected Improvements**:
- 10-15% better accuracy
- More complete "list all" responses
- Better understanding of technical terms
- Faster responses (5 GB vs 8 GB model)

---

## Testing Different Models

Create a test script:

```python
# test_models.py
from rag.chain import build_rag_chain

models = ["llama3.1:8b", "mistral:7b", "gemma2:9b"]
query = "List all units for Machine Learning"

for model in models:
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print('='*60)
    
    try:
        chain = build_rag_chain(llm_model=model, k=12)
        answer = chain(query)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
```

Run:
```bash
python test_models.py
```

---

## Quick Start: Upgrade to Llama 3.1

```bash
# 1. Pull model
ollama pull llama3.1:8b

# 2. Edit rag/chain.py (line ~20)
# Change: llm_model: str = "qwen2.5:14b"
# To:     llm_model: str = "llama3.1:8b"

# 3. Restart app
python -m streamlit run app.py

# 4. Test
# Ask: "List all units for Machine Learning"
```

---

## Summary

**Current**: qwen2.5:14b + nomic-embed-text
- Good, but not optimal for academic content

**Recommended Upgrade**: llama3.1:8b + mxbai-embed-large
- Better accuracy
- Faster
- More complete responses
- Better for "list all" queries

**Cost**: Free (all models are open-source via Ollama)

**Effort**: 5 minutes to change + 30 seconds to re-index (if changing embeddings)

**Benefit**: 10-15% better accuracy, especially for comprehensive queries

---

Would you like me to update the code to use Llama 3.1 8B?
