"""
Reset the vector database - delete all chunks and start fresh.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.vector_db import VectorDBClient
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Resetting vector database...\n")
    
    try:
        # Connect to Qdrant
        vector_db = VectorDBClient()
        
        collection_name = "faculty_chunks"
        
        # Check if collection exists
        try:
            info = vector_db.get_collection_info(collection_name)
            print(f"Found collection: {collection_name}")
            print(f"Current points: {info['points_count']}")
            print(f"Status: {info['status']}\n")
            
            # Confirm deletion
            response = input(f"Delete collection '{collection_name}' and all data? (yes/no): ")
            
            if response.lower() == 'yes':
                vector_db.delete_collection(collection_name)
                print(f"\n✓ Collection '{collection_name}' deleted successfully!")
                print("\nYou can now run ingestion to start fresh:")
                print("python scripts/ingest_documents.py --input data/raw --metadata data/metadata.json")
            else:
                print("\nCancelled. No changes made.")
        
        except Exception as e:
            print(f"Collection '{collection_name}' does not exist or error: {e}")
            print("Nothing to delete.")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
