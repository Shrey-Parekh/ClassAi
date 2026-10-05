"""Reset Qdrant collection for Student Part."""
import os
from qdrant_client import QdrantClient

def reset_db(qdrant_url: str = None, collection_name: str = "academic_rag"):
    """Completely reset the Qdrant vector database."""

    # Use environment variable or parameter
    if qdrant_url is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    print(f"🗑️  Resetting Qdrant collection '{collection_name}' at {qdrant_url}...")

    try:
        # Delete collection via API
        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            client.delete_collection(collection_name)
            print(f"✅ Collection '{collection_name}' deleted via API")
        else:
            print(f"⚠️  Collection '{collection_name}' does not exist")
    except Exception as e:
        print(f"❌ API deletion failed: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker-compose up -d")

    print("✅ Database reset complete. Run ingestion to rebuild.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reset Qdrant collection")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL (default: from .env or http://localhost:6333)")
    parser.add_argument("--collection", default="academic_rag", help="Collection name")

    args = parser.parse_args()

    reset_db(qdrant_url=args.qdrant_url, collection_name=args.collection)
