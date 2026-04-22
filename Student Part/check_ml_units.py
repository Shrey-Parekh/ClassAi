"""Check ML units in database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from qdrant_client import QdrantClient

client = QdrantClient(path='./qdrant_db', force_disable_check_same_thread=True)
points, _ = client.scroll('academic_rag', limit=200, with_payload=True, with_vectors=False)

ml_units = [p for p in points if p.payload.get('metadata', {}).get('course_name') == 'Machine Learning' and p.payload.get('metadata', {}).get('chunk_type') == 'syllabus_unit']

print(f"Total Machine Learning units in database: {len(ml_units)}\n")
for i, p in enumerate(ml_units, 1):
    meta = p.payload.get("metadata", {})
    unit_num = meta.get("unit_number", "?")
    unit_name = meta.get("unit_name", "Unknown")
    print(f"{i}. Unit {unit_num}: {unit_name}")
