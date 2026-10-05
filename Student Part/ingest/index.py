"""
Simple and robust indexing pipeline for NMIMS academic documents.
"""
import os
import sys
from typing import List
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from ingest.extract import extract_all
from ingest.chunker import chunk_documents


def enrich_chunks(chunks: List[Document]) -> List[Document]:
    """Add computed metadata to chunks."""
    import re
    
    for chunk in chunks:
        content = chunk.page_content
        
        # Add content statistics
        chunk.metadata["content_length"] = len(content)
        chunk.metadata["word_count"] = len(content.split())
        
        # Extract all CO references
        cos = re.findall(r'CO-(\d+)', content, re.IGNORECASE)
        if cos:
            chunk.metadata["cos"] = list(set(f"CO-{co}" for co in cos))
        
        # Extract all SO references
        sos = re.findall(r'SO-(\d+)', content, re.IGNORECASE)
        if sos:
            chunk.metadata["sos"] = list(set(f"SO-{so}" for so in sos))
        
        # Extract unit references
        units = re.findall(r'Unit\s+(\d+)', content, re.IGNORECASE)
        if units:
            chunk.metadata["unit_refs"] = list(set(units))
    
    return chunks


def index_documents(
    syllabus_dir: str = "data/syllabus",
    qp_dir: str = "data/question_papers",
    qdrant_url: str = None,
    collection_name: str = "academic_rag",
    recreate: bool = True
):
    """
    Main indexing pipeline.
    """
    print("=" * 70)
    print("🚀 NMIMS Academic RAG - Indexing Pipeline")
    print("=" * 70)
    
    # Step 1: Extract
    print("\n📖 Step 1: Extracting Documents")
    print("-" * 70)
    
    all_docs = []
    
    if os.path.exists(syllabus_dir):
        syllabus_docs = extract_all(syllabus_dir)
        all_docs.extend(syllabus_docs)
        print(f"✓ Extracted {len(syllabus_docs)} syllabus document(s)")
    else:
        print(f"⚠️  Syllabus directory not found: {syllabus_dir}")
    
    if os.path.exists(qp_dir):
        qp_docs = extract_all(qp_dir)
        all_docs.extend(qp_docs)
        print(f"✓ Extracted {len(qp_docs)} question paper document(s)")
    else:
        print(f"⚠️  Question paper directory not found: {qp_dir}")
    
    if not all_docs:
        print("\n❌ No documents found!")
        print(f"   Add markdown files to: {syllabus_dir} or {qp_dir}")
        return False
    
    print(f"\n✓ Total: {len(all_docs)} documents")
    
    # Step 2: Chunk
    print("\n✂️  Step 2: Chunking Documents")
    print("-" * 70)
    
    chunks = chunk_documents(all_docs)
    print(f"✓ Generated {len(chunks)} chunks")
    
    # Show chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        ctype = chunk.metadata.get("chunk_type", "unknown")
        chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
    
    print("\nChunk Distribution:")
    for ctype, count in sorted(chunk_types.items()):
        print(f"  • {ctype}: {count}")
    
    # Step 3: Enrich
    print("\n🔍 Step 3: Enriching Metadata")
    print("-" * 70)
    
    chunks = enrich_chunks(chunks)
    print(f"✓ Enriched {len(chunks)} chunks")
    
    # Step 4: Initialize Embeddings
    print("\n🧠 Step 4: Initializing Embeddings")
    print("-" * 70)
    
    try:
        embeddings = OllamaEmbeddings(model="bge-m3")
        test_vec = embeddings.embed_query("test")
        embedding_dim = len(test_vec)
        print(f"✓ Model: bge-m3")
        print(f"✓ Dimension: {embedding_dim}")
    except Exception as e:
        print(f"\n❌ Failed to initialize embeddings!")
        print(f"   Error: {e}")
        print("\n   Make sure Ollama is running:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Pull model: ollama pull bge-m3")
        return False
    
    # Step 5: Initialize Qdrant
    print("\n💾 Step 5: Initializing Vector Store")
    print("-" * 70)
    
    # Use environment variable or parameter
    if qdrant_url is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    try:
        print(f"Connecting to Qdrant at: {qdrant_url}")
        client = QdrantClient(url=qdrant_url)
        
        # Create or recreate collection
        if client.collection_exists(collection_name):
            if recreate:
                print(f"🗑️  Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
            else:
                print(f"✓ Using existing collection: {collection_name}")
        
        if recreate or not client.collection_exists(collection_name):
            print(f"🆕 Creating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
        
        print(f"✓ Vector store ready at: {qdrant_url}")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize Qdrant!")
        print(f"   Error: {e}")
        print("\n   Make sure Qdrant is running:")
        print("   docker-compose up -d")
        return False
    
    # Step 6: Index
    print("\n📦 Step 6: Indexing Chunks")
    print("-" * 70)
    
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # Index in batches
        batch_size = 50
        total = len(chunks)
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            vector_store.add_documents(batch)
            progress = min(i + batch_size, total)
            print(f"  ✓ Indexed {progress}/{total} chunks")
        
        print(f"\n✅ Successfully indexed {total} chunks!")
        
    except Exception as e:
        print(f"\n❌ Failed to index documents!")
        print(f"   Error: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Indexing Summary")
    print("=" * 70)
    print(f"Documents:        {len(all_docs)}")
    print(f"Chunks:           {len(chunks)}")
    print(f"Embedding Dim:    {embedding_dim}")
    print(f"Collection:       {collection_name}")
    print(f"Qdrant URL:       {qdrant_url}")
    print("=" * 70)
    print("✅ Indexing Complete!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index NMIMS academic documents")
    parser.add_argument("--syllabus-dir", default="data/syllabus", help="Syllabus directory")
    parser.add_argument("--qp-dir", default="data/question_papers", help="Question papers directory")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL (default: from .env or http://localhost:6333)")
    parser.add_argument("--collection", default="academic_rag", help="Collection name")
    parser.add_argument("--append", action="store_true", help="Append to existing (don't recreate)")
    
    args = parser.parse_args()
    
    success = index_documents(
        syllabus_dir=args.syllabus_dir,
        qp_dir=args.qp_dir,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        recreate=not args.append
    )
    
    sys.exit(0 if success else 1)
