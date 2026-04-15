"""
Reset the vector database and BM25 index - delete all chunks and start fresh.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def main():
    """Reset Qdrant collection and BM25 index."""
    print("=" * 70)
    print("RESET VECTOR DATABASE")
    print("=" * 70)
    print()

    collection_name = "faculty_chunks"
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    try:
        # Connect to Qdrant
        client = QdrantClient(url=qdrant_url)
        print(f"✓ Connected to Qdrant at {qdrant_url}")
        
        # Check if collection exists
        try:
            info = client.get_collection(collection_name)
            print(f"✓ Found collection '{collection_name}' with {info.points_count} chunks")
            print()
            
            # Confirm deletion
            user_response = input(f"Delete collection '{collection_name}' and all data? (yes/no): ")
            
            if user_response.lower() != 'yes':
                print("\nCancelled. No changes made.")
                return
            
            # Delete collection
            client.delete_collection(collection_name)
            print(f"\n✓ Collection '{collection_name}' deleted successfully!")
            
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"Collection '{collection_name}' does not exist.")
                print("Nothing to delete.")
            else:
                raise
        
        # Clean up BM25 index files
        bm25_dir = Path(__file__).parent.parent / "bm25_index"
        if bm25_dir.exists():
            print("\n✓ Cleaning BM25 index directory...")
            for file in bm25_dir.glob("*"):
                if file.is_file():
                    file.unlink()
                    print(f"  Deleted: {file.name}")
        
        # Clean up cache
        cache_file = Path(__file__).parent.parent / "cache" / "cache.db"
        if cache_file.exists():
            print("\n✓ Cleaning cache...")
            try:
                cache_file.unlink()
                print("  Deleted: cache.db")
            except PermissionError:
                print("  ⚠ Could not delete cache.db (file is in use by API server)")
                print(f"  Stop the API server first, then manually delete: {cache_file}")
        
        print()
        print("=" * 70)
        print("RESET COMPLETE")
        print("=" * 70)
        print()
        print("You can now run ingestion to start fresh:")
        print("python scripts/ingest_new.py --input data/raw --metadata data/metadata.json")
        print()

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
