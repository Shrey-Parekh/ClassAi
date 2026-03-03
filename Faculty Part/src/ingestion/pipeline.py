"""
Complete ingestion pipeline from raw documents to vector database.
"""

from typing import List, Dict, Any
from pathlib import Path
import json

from .document_processor import DocumentProcessor
from ..chunking.semantic_chunker import SemanticChunker


class IngestionPipeline:
    """
    End-to-end pipeline for ingesting faculty documents.
    
    Steps:
    1. Process document (extract text, OCR, etc.)
    2. Semantic chunking (3 levels)
    3. Generate embeddings
    4. Store in vector database
    5. Build BM25 index
    """
    
    def __init__(
        self,
        vector_db_client,
        embedding_model,
        collection_name: str = "faculty_chunks"
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            vector_db_client: Vector database client
            embedding_model: Model for generating embeddings
            collection_name: Target collection name
        """
        self.doc_processor = DocumentProcessor()
        self.chunker = SemanticChunker()
        self.vector_db = vector_db_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
    
    def ingest_document(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest a single document.
        
        Args:
            file_path: Path to document file
            doc_metadata: Document metadata (title, date, applies_to, etc.)
        
        Returns:
            Ingestion summary with chunk counts
        """
        # Step 1: Process document
        processed = self.doc_processor.process_document(file_path, doc_metadata)
        
        # Step 2: Semantic chunking
        chunks = self.chunker.chunk_document(
            content=processed["content"],
            doc_metadata=processed["metadata"]
        )
        
        # Step 3: Generate embeddings and store
        stored_count = 0
        failed_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = self.embedding_model.embed(chunk.content)
                
                # Store in vector DB
                self._store_chunk(chunk, embedding)
                stored_count += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(chunks)} chunks...")
                    
            except Exception as e:
                error_msg = str(e)[:200]
                chunk_preview = chunk.content[:100].replace('\n', ' ')
                print(f"  ⚠ Chunk {i+1} failed (len={len(chunk.content)}): {error_msg}")
                print(f"     Preview: {chunk_preview}...")
                failed_chunks.append({
                    "index": i + 1,
                    "length": len(chunk.content),
                    "error": error_msg
                })
                continue
        
        if failed_chunks:
            print(f"  ⚠ Failed to embed {len(failed_chunks)} chunks: {failed_chunks[:5]}...")
        
        return {
            "doc_id": doc_metadata.get("doc_id"),
            "file_path": str(file_path),
            "chunks_created": len(chunks),
            "chunks_stored": stored_count,
            "chunks_failed": len(failed_chunks),
            "has_images": processed.get("images", []),
        }
    
    def ingest_directory(
        self,
        directory: Path,
        metadata_file: Path = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest all documents in a directory.
        
        Args:
            directory: Directory containing documents
            metadata_file: Optional JSON file with per-document metadata
        
        Returns:
            List of ingestion summaries
        """
        # Load metadata if provided
        metadata_map = {}
        if metadata_file and metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_map = json.load(f)
        
        results = []
        
        # Process all supported files
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in [
                '.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md', '.csv', '.xlsx', '.xls'
            ]:
                # Get metadata for this file
                doc_metadata = metadata_map.get(file_path.name, {
                    "doc_id": file_path.stem,
                    "title": file_path.stem,
                    "applies_to": "all_faculty",
                })
                
                try:
                    result = self.ingest_document(file_path, doc_metadata)
                    results.append(result)
                    print(f"✓ Ingested: {file_path.name}")
                except Exception as e:
                    print(f"✗ Failed: {file_path.name} - {e}")
                    results.append({
                        "doc_id": doc_metadata.get("doc_id"),
                        "file_path": str(file_path),
                        "error": str(e),
                    })
        
        return results
    
    def _store_chunk(self, chunk, embedding: List[float]):
        """Store chunk with embedding in vector database."""
        import hashlib
        
        # Convert string ID to integer hash for Qdrant
        chunk_id_int = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:16], 16)
        
        # Prepare payload
        payload = {
            "content": chunk.content,
            "chunk_level": chunk.level.value,
            "content_type": chunk.content_type.value,
            "token_count": chunk.token_count,
            "parent_doc_id": chunk.parent_doc_id,
            "original_chunk_id": chunk.chunk_id,  # Keep original ID in payload
            **chunk.metadata,
        }
        
        # Store in vector DB
        self.vector_db.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": chunk_id_int,
                "vector": embedding,
                "payload": payload,
            }]
        )
