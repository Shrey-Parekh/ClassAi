"""
Complete ingestion pipeline from raw documents to vector database.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import hashlib

from .document_processor import DocumentProcessor
from ..chunking.semantic_chunker import SemanticChunker
from ..utils.chunk_preprocessor import ChunkPreprocessor
from ..utils.dual_encoder_embeddings import DualEncoderEmbeddings


class IngestionPipeline:
    """
    End-to-end pipeline for ingesting faculty documents.
    
    Steps:
    1. Process document (extract text, OCR, etc.)
    2. Semantic chunking (3 levels)
    3. Pre-process chunks (normalize, validate, split)
    4. Generate embeddings (dual-encoder with fallback)
    5. Store in vector database
    6. Build BM25 index
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
            embedding_model: Model for generating embeddings (can be DualEncoderEmbeddings)
            collection_name: Target collection name
        """
        self.doc_processor = DocumentProcessor()
        self.chunker = SemanticChunker()
        self.preprocessor = ChunkPreprocessor()
        self.vector_db = vector_db_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Check if using dual-encoder
        self.is_dual_encoder = isinstance(embedding_model, DualEncoderEmbeddings)
    
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
        
        # Step 3: Pre-process, embed, and store
        stored_count = 0
        failed_chunks = []
        split_count = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Pre-process chunk
                preprocessed_chunks = self.preprocessor.preprocess(chunk.content)
                
                for pre_chunk in preprocessed_chunks:
                    # Check if discarded
                    if not pre_chunk.is_valid:
                        if pre_chunk.discard_reason:
                            print(f"  ⚠ Chunk {i+1} discarded: {pre_chunk.discard_reason}")
                        continue
                    
                    # Track splits
                    if pre_chunk.was_split:
                        split_count += 1
                    
                    # Generate embedding (dual-encoder or single)
                    if self.is_dual_encoder:
                        embedding, metadata = self.embedding_model.embed(
                            text=pre_chunk.text,
                            chunk_id=chunk.chunk_id,
                            source_file=file_path.name,
                            chunk_length_chars=len(pre_chunk.text)
                        )
                        
                        if embedding is None:
                            failed_chunks.append({
                                "index": i + 1,
                                "reason": "Both encoders failed"
                            })
                            continue
                        
                        # Add split metadata
                        if pre_chunk.was_split:
                            metadata["was_split"] = True
                            metadata["split_index"] = pre_chunk.split_index
                    else:
                        # Single encoder (backward compatibility)
                        embedding = self.embedding_model.embed(pre_chunk.text)
                        metadata = {
                            "embedding_model": "all-mpnet-base-v2",
                            "embedding_fallback": False,
                            "fallback_reason": None,
                            "chunk_length_chars": len(pre_chunk.text),
                            "was_split": pre_chunk.was_split,
                            "split_index": pre_chunk.split_index,
                        }
                    
                    # Store in vector DB
                    self._store_chunk(chunk, pre_chunk.text, embedding, metadata)
                    stored_count += 1
                    
                    # Progress indicator
                    if stored_count % 10 == 0:
                        print(f"  Processed {stored_count} chunks...")
                        
            except Exception as e:
                error_msg = str(e)[:200]
                chunk_preview = chunk.content[:100].replace('\n', ' ')
                print(f"  ✗ Chunk {i+1} failed: {error_msg}")
                print(f"     Preview: {chunk_preview}...")
                failed_chunks.append({
                    "index": i + 1,
                    "length": len(chunk.content),
                    "error": error_msg
                })
                continue
        
        if failed_chunks:
            print(f"  ⚠ Failed to process {len(failed_chunks)} chunks")
        
        return {
            "doc_id": doc_metadata.get("doc_id"),
            "file_path": str(file_path),
            "chunks_created": len(chunks),
            "chunks_stored": stored_count,
            "chunks_failed": len(failed_chunks),
            "chunks_split": split_count,
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
            with open(metadata_file, 'r', encoding='utf-8') as f:
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
        
        # Print summary if using dual-encoder
        if self.is_dual_encoder:
            self.embedding_model.print_summary()
        
        return results
    
    def _store_chunk(
        self,
        chunk,
        cleaned_text: str,
        embedding: List[float],
        embedding_metadata: Dict[str, Any]
    ):
        """Store chunk with embedding in vector database."""
        # Convert string ID to integer hash for Qdrant
        chunk_id_int = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:16], 16)
        
        # Prepare payload
        payload = {
            "content": cleaned_text,
            "chunk_level": chunk.level.value,
            "content_type": chunk.content_type.value,
            "token_count": chunk.token_count,
            "parent_doc_id": chunk.parent_doc_id,
            "original_chunk_id": chunk.chunk_id,
            **chunk.metadata,
            **embedding_metadata,  # Add embedding metadata
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
