from src.utils.vector_db import VectorDBClient

db = VectorDBClient(collection_name='faculty_chunks')
info = db.get_collection_info('faculty_chunks')
print(f"Points in collection: {info.get('points_count', 0)}")
