"""Append new documents to existing Qdrant collection."""
import os
import sys
import hashlib
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.extract import extract_all
from ingest.chunker import chunk_documents

def get_file_hash(filepath: str) -> str:
    """Generate MD5 hash for duplicate detection."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def append_db(
    syllabus_dir: str,
    qp_dir: str,
    qdrant_url: str = None,
    collection_name: str = "academic_rag"
):
    """Append new documents to existing index."""
    print("📖 Scanning for new/updated documents...")

    # Use environment variable or parameter
    if qdrant_url is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Initialize components
    client = QdrantClient(url=qdrant_url)
    embeddings = OllamaEmbeddings(model="bge-m3")

    # Create collection if missing
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        print("⚠️  Collection missing. Creating...")
        test_vec = embeddings.embed_query("test")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(test_vec), distance=Distance.COSINE)
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # Get existing file hashes to skip duplicates
    existing_points, _ = client.scroll(collection_name, limit=10000, with_payload=True)
    existing_hashes = {p.payload.get("file_hash") for p in existing_points if p.payload.get("file_hash")}

    new_docs = []

    # Process new documents
    for directory in [syllabus_dir, qp_dir]:
        if not os.path.exists(directory):
            continue

        for f in os.listdir(directory):
            if f.endswith(".md"):
                path = os.path.join(directory, f)
                file_hash = get_file_hash(path)
                if file_hash not in existing_hashes:
                    docs = extract_all(directory)
                    for d in docs:
                        d.metadata["file_hash"] = file_hash
                    new_docs.extend(docs)
                    print(f"📄 New document: {f}")

    if not new_docs:
        print("✅ No new or updated documents found")
        return

    # Chunk and index
    print(f"✂️ Chunking {len(new_docs)} documents...")
    chunks = chunk_documents(new_docs)

    print(f"📦 Upserting {len(chunks)} chunks to Qdrant...")
    vector_store.add_documents(chunks)
    print("✅ Append complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--syllabus-dir", default="data/syllabus")
    parser.add_argument("--qp-dir", default="data/question_papers")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default="academic_rag")
    args = parser.parse_args()

    append_db(
        syllabus_dir=args.syllabus_dir,
        qp_dir=args.qp_dir,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection
    )
