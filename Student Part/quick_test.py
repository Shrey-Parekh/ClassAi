"""Quick system validation (no LLM test)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Quick System Test")
print("=" * 70)

# Test 1: Extraction
print("\n1. Testing Extraction...")
try:
    from ingest.extract import extract_all
    docs = extract_all("data/syllabus") + extract_all("data/question_papers")
    print(f"   ✅ Extracted {len(docs)} documents")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 2: Chunking
print("\n2. Testing Chunking...")
try:
    from ingest.chunker import chunk_documents
    chunks = chunk_documents(docs)
    print(f"   ✅ Generated {len(chunks)} chunks")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Ollama
print("\n3. Testing Ollama...")
try:
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    test_vec = embeddings.embed_query("test")
    print(f"   ✅ Embeddings working ({len(test_vec)}-dim)")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    print("   Make sure Ollama is running: ollama serve")
    sys.exit(1)

# Test 4: Database
print("\n4. Testing Database...")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(path="./qdrant_db", force_disable_check_same_thread=True)
    if client.collection_exists("academic_rag"):
        info = client.get_collection("academic_rag")
        print(f"   ✅ Database ready ({info.points_count} points)")
    else:
        print(f"   ⚠️  Database not indexed yet")
        print("   Run: python ingest/index.py")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 5: Retrieval
print("\n5. Testing Retrieval...")
try:
    from rag.retriever import get_retriever
    retriever = get_retriever(k=3)
    docs = retriever.retrieve("What is in Unit 2?")
    print(f"   ✅ Retrieved {len(docs)} documents")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All tests passed! System is ready.")
print("=" * 70)
print("\nNext step:")
print("  python -m streamlit run app.py")
