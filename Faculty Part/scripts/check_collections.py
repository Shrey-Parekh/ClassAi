"""Verify Qdrant collections before integration."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def check_collections():
    """Check both faculty and student collections in Docker Qdrant."""
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"Connecting to Qdrant at: {qdrant_url}")
    print("=" * 70)
    
    try:
        client = QdrantClient(url=qdrant_url)
        
        # Get all collections
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        print(f"\nFound {len(collections)} collection(s):")
        for c in collections:
            print(f"  - {c.name}")
        
        print("\n" + "=" * 70)
        
        # Check faculty_chunks
        print("\n📊 FACULTY_CHUNKS Collection:")
        print("-" * 70)
        
        if "faculty_chunks" in collection_names:
            faculty_info = client.get_collection("faculty_chunks")
            faculty_count = client.count("faculty_chunks").count
            
            print(f"✓ Collection exists")
            print(f"  Points count:    {faculty_count}")
            print(f"  Vector size:     {faculty_info.config.params.vectors.size}")
            print(f"  Distance metric: {faculty_info.config.params.vectors.distance}")
            print(f"  Status:          {faculty_info.status}")
            
            if faculty_count == 0:
                print("  ⚠️  WARNING: Collection is empty!")
        else:
            print("✗ Collection does NOT exist")
            print("  Run: python scripts/ingest_new.py --input data/raw --metadata data/metadata.json")
        
        print("\n" + "=" * 70)
        
        # Check academic_rag
        print("\n📊 ACADEMIC_RAG Collection (Student):")
        print("-" * 70)
        
        if "academic_rag" in collection_names:
            student_info = client.get_collection("academic_rag")
            student_count = client.count("academic_rag").count
            
            print(f"✓ Collection exists")
            print(f"  Points count:    {student_count}")
            print(f"  Vector size:     {student_info.config.params.vectors.size}")
            print(f"  Distance metric: {student_info.config.params.vectors.distance}")
            print(f"  Status:          {student_info.status}")
            
            if student_count == 0:
                print("  ⚠️  WARNING: Collection is empty!")
                print("  Run: cd 'Student Part' && python ingest/index.py")
        else:
            print("✗ Collection does NOT exist")
            print("  Run: cd 'Student Part' && python ingest/index.py")
        
        print("\n" + "=" * 70)
        
        # Compatibility check
        print("\n🔍 Compatibility Check:")
        print("-" * 70)
        
        if "faculty_chunks" in collection_names and "academic_rag" in collection_names:
            faculty_info = client.get_collection("faculty_chunks")
            student_info = client.get_collection("academic_rag")
            
            faculty_size = faculty_info.config.params.vectors.size
            student_size = student_info.config.params.vectors.size
            
            faculty_distance = faculty_info.config.params.vectors.distance
            student_distance = student_info.config.params.vectors.distance
            
            if faculty_size == student_size:
                print(f"✓ Vector sizes match: {faculty_size} dimensions")
            else:
                print(f"✗ Vector size MISMATCH: faculty={faculty_size}, student={student_size}")
                print("  STOP: Cannot proceed with integration!")
                return False
            
            if faculty_distance == student_distance:
                print(f"✓ Distance metrics match: {faculty_distance}")
            else:
                print(f"✗ Distance metric MISMATCH: faculty={faculty_distance}, student={student_distance}")
                print("  STOP: Cannot proceed with integration!")
                return False
            
            faculty_count = client.count("faculty_chunks").count
            student_count = client.count("academic_rag").count
            
            if faculty_count > 0 and student_count > 0:
                print(f"✓ Both collections have data")
                print(f"  Faculty: {faculty_count} chunks")
                print(f"  Student: {student_count} chunks")
            else:
                print("⚠️  One or both collections are empty")
                if faculty_count == 0:
                    print("  Faculty collection needs ingestion")
                if student_count == 0:
                    print("  Student collection needs ingestion")
        
        print("\n" + "=" * 70)
        print("\n✅ Prerequisite check complete!")
        
        # Final verdict
        if "faculty_chunks" in collection_names and "academic_rag" in collection_names:
            faculty_count = client.count("faculty_chunks").count
            student_count = client.count("academic_rag").count
            
            if faculty_count > 0 and student_count > 0:
                print("\n🎉 READY FOR INTEGRATION!")
                print("   Both collections exist with compatible configurations.")
                return True
            else:
                print("\n⚠️  INGESTION REQUIRED!")
                print("   Collections exist but need data.")
                return False
        else:
            print("\n❌ NOT READY!")
            print("   One or both collections missing.")
            return False
        
    except Exception as e:
        print(f"\n❌ Error connecting to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker-compose up -d")
        return False

if __name__ == "__main__":
    success = check_collections()
    sys.exit(0 if success else 1)
