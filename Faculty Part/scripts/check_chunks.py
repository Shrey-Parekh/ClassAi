"""
Quick script to check chunks in Qdrant by document name.
"""

from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Get all chunks from Legal document
results = client.scroll(
    collection_name="faculty_chunks",
    scroll_filter={
        "must": [
            {"key": "document_name", "match": {"value": "NMIMS_Faculty_Employment_Agreement_Legal.pdf"}}
        ]
    },
    limit=5,
    with_payload=True
)

print(f"\nFound {len(results[0])} chunks from Legal document:\n")

for i, point in enumerate(results[0], 1):
    content = point.payload.get("content", "")
    section = point.payload.get("section_title", "N/A")
    source_type = point.payload.get("source_type", "N/A")
    
    print(f"Chunk {i}:")
    print(f"  Source Type: {source_type}")
    print(f"  Section: {section}")
    print(f"  Content preview: {content[:150]}...")
    print("---\n")
