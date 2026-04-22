"""Check what's in the database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from qdrant_client import QdrantClient

client = QdrantClient(path='./qdrant_db', force_disable_check_same_thread=True)
points, _ = client.scroll('academic_rag', limit=20, with_payload=True, with_vectors=False)

print("Database Contents (first 20 chunks):\n")
for i, p in enumerate(points, 1):
    meta = p.payload.get("metadata", {})
    chunk_type = meta.get("chunk_type", "unknown")
    course_name = meta.get("course_name", "N/A")
    unit_name = meta.get("unit_name", "N/A")[:40]
    
    print(f"{i}. Type: {chunk_type:15} | Course: {course_name:25} | Unit: {unit_name}")
