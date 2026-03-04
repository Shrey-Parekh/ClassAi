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
        
        # Try to delete collection via REST API directly (bypass pydantic validation)
        try:
            import requests
            url = "http://localhost:6333/collections/faculty_chunks"
            response = requests.delete(url, timeout=5)
            
            if response.status_code in [200, 204]:
                print(f"✓ Collection '{collection_name}' deleted successfully!")
                print("\nYou can now run ingestion to start fresh:")
                print("python scripts/ingest_documents.py --input data/raw --metadata data/metadata.json")
            elif response.status_code == 404:
                print(f"Collection '{collection_name}' does not exist.")
                print("Nothing to delete.")
            else:
                print(f"Unexpected response: {response.status_code}")
                print(response.text)
        
        except Exception as e:
            print(f"Could not delete via REST API: {e}")
            print("Trying via client...")
            
            try:
                info = vector_db.get_collection_info(collection_name)
                print(f"Found collection: {collection_name}")
                
                # Confirm deletion
                response = input(f"Delete collection '{collection_name}' and all data? (yes/no): ")
                
                if response.lower() == 'yes':
                    vector_db.delete_collection(collection_name)
                    print(f"\n✓ Collection '{collection_name}' deleted successfully!")
                else:
                    print("\nCancelled. No changes made.")
            
            except Exception as e2:
                print(f"Collection does not exist or error: {e2}")
                print("Nothing to delete.")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
