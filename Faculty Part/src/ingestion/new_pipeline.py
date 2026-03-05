"""
Complete ingestion pipeline using document-type-specific chunking and BAAI/bge-m3.

Replaces old multi-layer chunking strategy.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import hashlib
import logging
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import torch

from ..chunking.document_chunker import DocumentChunker
from .document_processor import DocumentProcessor
from ..utils.sparse_encoder import SparseEncoder


class NewIngestionPipeline:
    """
    Complete ingestion pipeline with document-type-specific chunking.
    
    Uses BAAI/bge-m3 (8192 token context, 1024 dimensions).
    """
    
    def __init__(
        self,
        vector_db_client,
        collection_name: str = "faculty_chunks"
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            vector_db_client: Qdrant client
            collection_name: Target collection name
        """
        self.vector_db = vector_db_client
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        print("\n" + "="*60)
        print("INITIALIZING INGESTION PIPELINE")
        print("="*60)
        
        print("\n[1/4] Initializing document processor...")
        self.doc_processor = DocumentProcessor()
        print("      ✓ Document processor ready")
        
        print("\n[2/4] Initializing chunker...")
        self.chunker = DocumentChunker()
        print("      ✓ Chunker ready")
        
        # Load embedding model
        print("\n[3/4] Loading BAAI/bge-m3 embedding model...")
        print("      This may take a few minutes on first run (downloading ~2GB model)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"      Device: {device}")
        
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-m3",
            device=device
        )
        print(f"      ✓ Embedding model loaded on {device}")
        
        # Initialize sparse encoder
        print("\n[4/4] Initializing sparse encoder...")
        try:
            self.sparse_encoder = SparseEncoder(model_name="prithivida/Splade_PP_en_v1")
            print("      ✓ Sparse encoder loaded")
        except Exception as e:
            print(f"      ⚠ Sparse encoder not available: {e}")
            self.sparse_encoder = None
        
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60 + "\n")
        
        # Statistics
        self.stats = defaultdict(int)
    
    def ingest_document(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest a single document.
        
        Args:
            file_path: Path to document file
            doc_metadata: Document metadata (title, date, etc.)
        
        Returns:
            Ingestion summary
        """
        print(f"\n{'─'*60}")
        print(f"📄 Processing: {file_path.name}")
        print(f"{'─'*60}")
        
        try:
            # Step 1: Process document (extract text)
            print(f"   [1/4] Extracting text from {file_path.suffix} file...")
            processed = self.doc_processor.process_document(file_path, doc_metadata)
            text_length = len(processed["content"])
            print(f"         ✓ Extracted {text_length:,} characters")
            
            # Step 2: Chunk document based on type
            source_type = self.chunker.detect_source_type(file_path)
            print(f"   [2/4] Chunking as '{source_type}'...")
            chunks = self.chunker.chunk_document(
                text=processed["content"],
                filepath=file_path,
                doc_metadata=processed["metadata"]
            )
            print(f"         ✓ Created {len(chunks)} chunks")
            
            # Step 3: Process and store each chunk
            print(f"   [3/4] Embedding and storing chunks...")
            stored_count = 0
            skipped_count = 0
            skip_reasons = defaultdict(int)
            
            for i, chunk in enumerate(chunks):
                # Quality filter
                should_skip, reason = self.chunker.should_skip_chunk(chunk)
                if should_skip:
                    skipped_count += 1
                    skip_reasons[reason] += 1
                    continue
                
                # Clean text
                clean_text = self.chunker.clean_chunk_text(chunk.text)
                
                # Embed
                try:
                    dense_vector = self._embed_text(clean_text)
                    sparse_vector = self._compute_sparse(clean_text)
                    
                    # Store in Qdrant
                    self._store_chunk(
                        text=clean_text,
                        dense_vector=dense_vector,
                        sparse_vector=sparse_vector,
                        metadata=chunk.metadata
                    )
                    
                    stored_count += 1
                    
                    # Update stats by source type
                    source_type = chunk.metadata.get("source_type", "unknown")
                    self.stats[source_type] += 1
                    
                    # Progress indicator
                    if stored_count % 5 == 0:
                        print(f"         • Processed {stored_count}/{len(chunks)} chunks...")
                    
                except Exception as e:
                    print(f"         ✗ Failed to embed chunk {i+1}: {str(e)[:50]}")
                    skipped_count += 1
                    skip_reasons["embedding_failed"] += 1
            
            print(f"         ✓ Stored {stored_count} chunks")
            if skipped_count > 0:
                print(f"         ⚠ Skipped {skipped_count} chunks")
            
            print(f"   [4/4] Complete!")
            print(f"         ✓ {file_path.name}: SUCCESS")
            
            return {
                "doc_id": doc_metadata.get("doc_id"),
                "file_path": str(file_path),
                "chunks_created": len(chunks),
                "chunks_stored": stored_count,
                "chunks_skipped": skipped_count,
                "skip_reasons": dict(skip_reasons),
                "source_type": self.chunker.detect_source_type(file_path)
            }
            
        except Exception as e:
            print(f"         ✗ FAILED: {str(e)[:100]}")
            return {
                "doc_id": doc_metadata.get("doc_id"),
                "file_path": str(file_path),
                "error": str(e)
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
        print("\n" + "="*60)
        print("STARTING DOCUMENT INGESTION")
        print("="*60)
        
        # Load metadata if provided
        metadata_map = {}
        if metadata_file and metadata_file.exists():
            print(f"\n📋 Loading metadata from: {metadata_file.name}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_map = json.load(f)
            print(f"   ✓ Loaded metadata for {len(metadata_map)} documents")
        
        results = []
        
        # Process all supported files
        supported_extensions = [
            '.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md',
            '.json', '.csv', '.xlsx', '.xls', '.docx'
        ]
        
        # Count files first
        all_files = [f for f in directory.rglob('*') 
                     if f.is_file() and f.suffix.lower() in supported_extensions]
        
        print(f"\n📁 Found {len(all_files)} documents to process")
        print(f"   Directory: {directory}")
        print(f"   Supported types: {', '.join(supported_extensions)}")
        
        # Process each file
        for idx, file_path in enumerate(all_files, 1):
            print(f"\n[{idx}/{len(all_files)}] ", end="")
            
            # Get metadata for this file
            doc_metadata = metadata_map.get(file_path.name, {
                "doc_id": file_path.stem,
                "title": file_path.stem,
                "applies_to": "all_faculty",
            })
            
            result = self.ingest_document(file_path, doc_metadata)
            results.append(result)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _embed_text(self, text: str) -> List[float]:
        """
        Embed text using BAAI/bge-m3.
        
        Args:
            text: Text to embed
        
        Returns:
            1024-dimensional embedding vector
        """
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
            batch_size=8,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    def _compute_sparse(self, text: str) -> Dict[int, float]:
        """
        Compute sparse vector using SPLADE.
        
        Args:
            text: Text to encode
        
        Returns:
            Sparse vector as dict
        """
        if not self.sparse_encoder:
            return {}
        
        try:
            return self.sparse_encoder.encode(text)
        except Exception as e:
            self.logger.debug(f"Sparse encoding failed: {e}")
            return {}
    
    def _store_chunk(
        self,
        text: str,
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        metadata: Dict[str, Any]
    ):
        """Store chunk with embeddings in Qdrant."""
        # Generate unique ID
        chunk_id_str = f"{metadata.get('document_name', '')}_{metadata.get('section_index', 0)}_{metadata.get('sub_index', 0)}"
        chunk_id_int = int(hashlib.md5(chunk_id_str.encode()).hexdigest()[:16], 16)
        
        # Prepare payload
        payload = {
            "text": text,
            "char_count": len(text),
            "token_count": len(text) // 4,  # Approximate
            **metadata
        }
        
        # Store in Qdrant
        self.vector_db.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": chunk_id_int,
                "vector": dense_vector,
                "payload": payload,
            }]
        )
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print ingestion summary."""
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        
        # Count by source type
        source_counts = defaultdict(int)
        for source_type, count in self.stats.items():
            source_counts[source_type] = count
        
        print("\nChunks stored by document type:")
        print(f"  Faculty profiles:       {source_counts['faculty_profile']:>5} chunks")
        print(f"  HR policy:              {source_counts['hr_policy']:>5} chunks")
        print(f"  Legal documents:        {source_counts['legal_document']:>5} chunks")
        print(f"  Guidelines:             {source_counts['guidelines']:>5} chunks")
        print(f"  Procedures:             {source_counts['procedure_document']:>5} chunks")
        print(f"  Forms:                  {source_counts['form_document']:>5} chunks")
        print(f"  General:                {source_counts['general_document']:>5} chunks")
        
        # Total stats
        total_stored = sum(source_counts.values())
        total_skipped = sum(r.get('chunks_skipped', 0) for r in results if 'chunks_skipped' in r)
        total_failed = len([r for r in results if 'error' in r])
        
        print(f"\nTotal chunks stored:    {total_stored:>5}")
        print(f"Total chunks skipped:   {total_skipped:>5}")
        print(f"Total files failed:     {total_failed:>5}")
        
        # Skip reasons
        if total_skipped > 0:
            print("\nSkip reasons:")
            skip_reasons = defaultdict(int)
            for r in results:
                if 'skip_reasons' in r:
                    for reason, count in r['skip_reasons'].items():
                        skip_reasons[reason] += count
            
            for reason, count in skip_reasons.items():
                print(f"  {reason:20s}: {count:>5}")
        
        print("=" * 60 + "\n")
