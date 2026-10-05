"""
Complete system test for NMIMS Academic RAG.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_extraction():
    """Test document extraction."""
    print("\n" + "=" * 70)
    print("TEST 1: Document Extraction")
    print("=" * 70)
    
    try:
        from ingest.extract import extract_all
        
        # Test syllabus
        syllabus_docs = extract_all("data/syllabus")
        print(f"✓ Syllabus: {len(syllabus_docs)} documents")
        
        # Test question papers
        qp_docs = extract_all("data/question_papers")
        print(f"✓ Question Papers: {len(qp_docs)} documents")
        
        total = len(syllabus_docs) + len(qp_docs)
        print(f"\n✅ Extraction: {total} documents extracted")
        return True
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return False


def test_chunking():
    """Test document chunking."""
    print("\n" + "=" * 70)
    print("TEST 2: Document Chunking")
    print("=" * 70)
    
    try:
        from ingest.extract import extract_all
        from ingest.chunker import chunk_documents
        
        # Extract
        all_docs = []
        all_docs.extend(extract_all("data/syllabus"))
        all_docs.extend(extract_all("data/question_papers"))
        
        # Chunk
        chunks = chunk_documents(all_docs)
        print(f"✓ Generated: {len(chunks)} chunks")
        
        # Show distribution
        chunk_types = {}
        for chunk in chunks:
            ctype = chunk.metadata.get("chunk_type", "unknown")
            chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
        
        print("\nChunk Distribution:")
        for ctype, count in sorted(chunk_types.items()):
            print(f"  • {ctype}: {count}")
        
        print(f"\n✅ Chunking: {len(chunks)} chunks created")
        return True
        
    except Exception as e:
        print(f"\n❌ Chunking failed: {e}")
        return False


def test_ollama():
    """Test Ollama connection."""
    print("\n" + "=" * 70)
    print("TEST 3: Ollama Connection")
    print("=" * 70)
    
    try:
        from langchain_ollama import OllamaEmbeddings, ChatOllama
        
        # Test embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_vec = embeddings.embed_query("test")
        print(f"✓ Embeddings: nomic-embed-text ({len(test_vec)}-dim)")
        
        # Test LLM
        llm = ChatOllama(model="qwen2.5:14b", temperature=0.1)
        response = llm.invoke("Say OK")
        print(f"✓ LLM: qwen2.5:14b (responding)")
        
        print(f"\n✅ Ollama: All models working")
        return True
        
    except Exception as e:
        print(f"\n❌ Ollama failed: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Models are pulled:")
        print("     ollama pull nomic-embed-text")
        print("     ollama pull qwen2.5:14b")
        return False


def test_indexing():
    """Test document indexing."""
    print("\n" + "=" * 70)
    print("TEST 4: Document Indexing")
    print("=" * 70)
    
    if not os.path.exists("./qdrant_db"):
        print("⚠️  Database not found")
        print("   Run: python ingest/index.py")
        return False
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(path="./qdrant_db", force_disable_check_same_thread=True)
        
        if client.collection_exists("academic_rag"):
            collection_info = client.get_collection("academic_rag")
            print(f"✓ Collection: academic_rag")
            print(f"✓ Points: {collection_info.points_count}")
            print(f"\n✅ Indexing: Database ready")
            return True
        else:
            print("⚠️  Collection not found")
            print("   Run: python ingest/index.py")
            return False
            
    except Exception as e:
        print(f"\n❌ Indexing check failed: {e}")
        return False


def test_retrieval():
    """Test document retrieval."""
    print("\n" + "=" * 70)
    print("TEST 5: Document Retrieval")
    print("=" * 70)
    
    try:
        from rag.retriever import get_retriever
        
        retriever = get_retriever(k=3)
        docs = retriever.retrieve("What is covered in Unit 2 of Cyber Security?")
        
        print(f"✓ Retrieved: {len(docs)} documents")
        if docs:
            print(f"✓ First result: {docs[0].metadata.get('chunk_type', 'unknown')}")
        
        print(f"\n✅ Retrieval: Working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Retrieval failed: {e}")
        return False


def test_rag_chain():
    """Test complete RAG chain."""
    print("\n" + "=" * 70)
    print("TEST 6: RAG Chain")
    print("=" * 70)
    
    try:
        from rag.chain import build_rag_chain
        
        chain = build_rag_chain(k=3)
        response = chain("What topics are covered in Unit 2 of Cyber Security?")
        
        print(f"✓ Generated response ({len(response)} chars)")
        print(f"\nPreview:\n{response[:200]}...")
        
        print(f"\n✅ RAG Chain: Working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG Chain failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("NMIMS Academic RAG - System Test")
    print("=" * 70)
    
    tests = [
        ("Extraction", test_extraction),
        ("Chunking", test_chunking),
        ("Ollama", test_ollama),
        ("Indexing", test_indexing),
        ("Retrieval", test_retrieval),
        ("RAG Chain", test_rag_chain),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python -m streamlit run app.py")
        print("  2. Open: http://localhost:8501")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
