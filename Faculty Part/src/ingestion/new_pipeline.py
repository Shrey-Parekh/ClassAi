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
from ..utils.chunk_preprocessor import ChunkPreprocessor
from .document_processor import DocumentProcessor


class NewIngestionPipeline:
    """
    Complete ingestion pipeline with document-type-specific chunking.
    
    Uses BAAI/bge-m3 (8192 token context, 1024 dimensions).
    """
    
    def __init__(
        self,
        vector_db_client,
        collection_name: str = "faculty_chunks",
        manifest_path: Path = None,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            vector_db_client: Qdrant client
            collection_name: Target collection name
            manifest_path: Optional path to the JSON ingestion manifest. The
                manifest tracks `{document_name: {content_hash, chunks_stored,
                ingested_at}}` so that subsequent runs can skip files whose
                bytes have not changed and purge stale chunks for files that
                have changed. Defaults to ``data/ingest_manifest.json`` next
                to the project root.
        """
        self.vector_db = vector_db_client
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

        # Manifest for content-hash based incremental ingestion. We pick a
        # default path beside the project's `data/` dir; callers can override.
        if manifest_path is None:
            manifest_path = (
                Path(__file__).resolve().parents[2]
                / "data" / "ingest_manifest.json"
            )
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()
        
        # Initialize components
        print("\n" + "="*60)
        print("INITIALIZING INGESTION PIPELINE")
        print("="*60)
        
        print("\n[1/4] Initializing document processor...")
        self.doc_processor = DocumentProcessor()
        print("      ✓ Document processor ready")
        
        print("\n[2/4] Initializing chunker...")
        self.chunker = DocumentChunker()
        self.chunk_preprocessor = ChunkPreprocessor()
        print("      ✓ Chunker ready")
        print("      ✓ Chunk preprocessor ready (reranker-safe splitting)")
        
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
        
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60 + "\n")
        
        # Statistics
        self.stats = defaultdict(int)
    
    def ingest_document(
        self,
        file_path: Path,
        doc_metadata: Dict[str, Any],
        force_reingest: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest a single document.

        Args:
            file_path: Path to document file
            doc_metadata: Document metadata (title, date, etc.)
            force_reingest: If True, ignore the manifest content-hash check
                and re-process the document even when its bytes are unchanged.
            dry_run: If True, run chunking and report counts but do NOT embed,
                store, or modify the manifest. Useful for validating chunking
                after regex/code changes without spending compute.

        Returns:
            Ingestion summary
        """
        print(f"\n{'─'*60}")
        print(f"📄 Processing: {file_path.name}")
        print(f"{'─'*60}")

        try:
            # Content-hash check — if the file's bytes are identical to the
            # previously-ingested version AND the caller hasn't asked for a
            # forced reingest, we skip the whole pipeline. This is the most
            # expensive step in the system (embedding model at ~1s / chunk on
            # CPU) so this matters in practice.
            content_hash = self._compute_content_hash(file_path)
            prev = self.manifest.get(file_path.name)
            if (
                not force_reingest
                and not dry_run
                and prev is not None
                and prev.get("content_hash") == content_hash
            ):
                prev_chunks = prev.get("chunks_stored", 0)
                print(
                    f"   ⏭  Unchanged since last ingest "
                    f"(hash={content_hash[:10]}…, {prev_chunks} chunks). "
                    f"Skipping. Use --force to re-ingest."
                )
                return {
                    "doc_id": doc_metadata.get("doc_id"),
                    "file_path": str(file_path),
                    "skipped_unchanged": True,
                    "content_hash": content_hash,
                    "chunks_stored": prev_chunks,
                }

            # If we reach here, either (a) the file is new, (b) the bytes
            # changed, or (c) --force was passed. For (b)/(c) we purge any
            # stale chunks for this document_name so they don't coexist with
            # the freshly-produced chunks (which may have different section
            # indices / titles after chunker updates).
            if prev is not None and not dry_run:
                try:
                    n_deleted = self.vector_db.delete_by_document_name(
                        collection_name=self.collection_name,
                        document_name=file_path.name,
                    )
                    if n_deleted:
                        print(f"   🧹 Purged {n_deleted} stale chunks from previous ingest")
                except Exception as e:
                    print(f"   ⚠ Stale-chunk purge failed: {e}")

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

            # Dry-run short-circuit: report the planned chunk counts and
            # return without embedding/storing anything. We still expose the
            # per-type breakdown and a sample of the section titles for fast
            # chunker-regression validation.
            if dry_run:
                type_counts = defaultdict(int)
                sample_titles = []
                for chunk in chunks:
                    ct = chunk.metadata.get("chunk_type", "unknown")
                    type_counts[ct] += 1
                    if len(sample_titles) < 8:
                        st = chunk.metadata.get("section_title")
                        if st:
                            sample_titles.append(st)
                print(f"   🏜  DRY RUN — no embedding, no storage, manifest unchanged.")
                print(f"         chunks_created: {len(chunks)}")
                print(f"         by_type: {dict(type_counts)}")
                if sample_titles:
                    print(f"         sample_titles: {sample_titles}")
                return {
                    "doc_id": doc_metadata.get("doc_id"),
                    "file_path": str(file_path),
                    "dry_run": True,
                    "chunks_created": len(chunks),
                    "chunks_by_type": dict(type_counts),
                    "sample_section_titles": sample_titles,
                    "source_type": source_type,
                }

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

                # Reranker-safety pre-processing: splits oversized chunks at
                # sentence boundaries (MAX_TOKENS=490), drops empty/too-short
                # fragments, normalises unicode. This protects the BAAI
                # bge-reranker-v2-m3 512-token input window downstream.
                sub_chunks = self.chunk_preprocessor.preprocess(clean_text)

                valid_sub_chunks = [sc for sc in sub_chunks if sc.is_valid]
                if not valid_sub_chunks:
                    # All sub-chunks failed validation — record the reason
                    # from the first result and skip this chunk.
                    first_reason = sub_chunks[0].discard_reason if sub_chunks else "preprocess_failed"
                    skipped_count += 1
                    skip_reasons[f"preprocess_{first_reason}"] += 1
                    continue

                n_sub = len(valid_sub_chunks)
                for sub_idx, sc in enumerate(valid_sub_chunks):
                    # Propagate sub-chunk index into metadata so every point
                    # gets a unique Qdrant id (see _store_chunk hash input).
                    sub_meta = dict(chunk.metadata)
                    if sc.was_split or n_sub > 1:
                        # Preserve any existing sub_index from the chunker
                        # and add a second-level preprocessor index.
                        existing = sub_meta.get("sub_section_index",
                                                sub_meta.get("sub_index", 0))
                        sub_meta["sub_section_index"] = existing
                        sub_meta["preprocess_split_index"] = sub_idx
                        sub_meta["was_split_for_reranker"] = True

                        # Disambiguate section_title for cited sources so
                        # multiple splits of the same parent section are
                        # distinguishable in retrieval UI: "SECTION 5 (2/3)".
                        original_title = sub_meta.get("section_title", "")
                        if original_title and n_sub > 1:
                            sub_meta["section_title"] = (
                                f"{original_title} ({sub_idx + 1}/{n_sub})"
                            )
                            sub_meta["parent_section_title"] = original_title

                    try:
                        dense_vector = self._embed_text(sc.text)

                        # Store in Qdrant
                        self._store_chunk(
                            text=sc.text,
                            dense_vector=dense_vector,
                            metadata=sub_meta
                        )

                        stored_count += 1

                        # Update stats by source type
                        source_type = sub_meta.get("source_type", "unknown")
                        self.stats[source_type] += 1

                        # Progress indicator
                        if stored_count % 5 == 0:
                            print(f"         • Processed {stored_count}/{len(chunks)} chunks...")

                    except Exception as e:
                        print(f"         ✗ Failed to embed chunk {i+1}.{sub_idx}: {str(e)[:50]}")
                        skipped_count += 1
                        skip_reasons["embedding_failed"] += 1
            
            print(f"         ✓ Stored {stored_count} chunks")
            if skipped_count > 0:
                print(f"         ⚠ Skipped {skipped_count} chunks")
            
            print(f"   [4/4] Complete!")
            print(f"         ✓ {file_path.name}: SUCCESS")

            # Record successful ingest in the manifest so subsequent runs on
            # unchanged bytes skip directly. We persist after every file so a
            # crash mid-directory doesn't lose progress.
            from datetime import datetime
            self.manifest[file_path.name] = {
                "content_hash": content_hash,
                "chunks_stored": stored_count,
                "chunks_created": len(chunks),
                "ingested_at": datetime.utcnow().isoformat() + "Z",
                "source_type": source_type,
            }
            self._save_manifest()

            return {
                "doc_id": doc_metadata.get("doc_id"),
                "file_path": str(file_path),
                "content_hash": content_hash,
                "chunks_created": len(chunks),
                "chunks_stored": stored_count,
                "chunks_skipped": skipped_count,
                "skip_reasons": dict(skip_reasons),
                "source_type": source_type,
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
        metadata_file: Path = None,
        force_reingest: bool = False,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ingest all documents in a directory.

        Args:
            directory: Directory containing documents
            metadata_file: Optional JSON file with per-document metadata
            force_reingest: Re-ingest every file even if its content hash
                matches the manifest (use after chunker / embedding changes).
            dry_run: Run chunking only; do not embed, store, or update the
                manifest. Useful for chunker validation in CI.

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
            
            result = self.ingest_document(
                file_path,
                doc_metadata,
                force_reingest=force_reingest,
                dry_run=dry_run,
            )
            results.append(result)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    # ---- Manifest helpers (content-hash incremental ingestion) -----------

    def _compute_content_hash(self, file_path: Path) -> str:
        """Stable sha256 of the file's raw bytes. Used as the cache key in
        the ingestion manifest so unchanged files can be skipped without
        re-parsing, re-chunking, or re-embedding.
        """
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        except OSError as e:
            # Degrade gracefully — hash the filename + mtime so at least
            # identity is stable within a single run. We deliberately do NOT
            # return a literal empty string, because "" would accidentally
            # match any other unreadable file.
            stat = file_path.stat() if file_path.exists() else None
            fallback = f"{file_path.name}|{stat.st_mtime_ns if stat else 'missing'}|{e}"
            h.update(fallback.encode("utf-8"))
        return h.hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the ingestion manifest JSON, returning {} if it doesn't
        exist or can't be parsed.
        """
        if not self.manifest_path.exists():
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            self.logger.warning(
                "manifest at %s was not a dict; ignoring", self.manifest_path
            )
            return {}
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(
                "failed to read manifest %s (%s); starting fresh",
                self.manifest_path, e
            )
            return {}

    def _save_manifest(self) -> None:
        """Persist the manifest to disk. Best-effort — a failure here is
        logged but does NOT abort ingestion.
        """
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, sort_keys=True)
            tmp.replace(self.manifest_path)
        except OSError as e:
            self.logger.warning(
                "failed to save manifest %s: %s", self.manifest_path, e
            )

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
    
    def _store_chunk(
        self,
        text: str,
        dense_vector: List[float],
        metadata: Dict[str, Any]
    ):
        """Store chunk with embeddings in Qdrant."""
        # Generate unique ID — include preprocess_split_index so reranker
        # safety splits don't collide on the same Qdrant point id.
        chunk_id_str = "|".join([
            metadata.get("document_name", ""),
            str(metadata.get("section_index", 0)),
            str(metadata.get("sub_section_index", metadata.get("sub_index", 0))),
            str(metadata.get("section_letter", "")),
            str(metadata.get("chunk_type", "")),
            str(metadata.get("preprocess_split_index", 0)),
        ])
        chunk_id_int = int(hashlib.md5(chunk_id_str.encode()).hexdigest()[:16], 16)
        
        # Prepare payload
        payload = {
            "content": text,  # Use "content" to match BM25 index expectations
            "char_count": len(text),
            "token_count": int(len(text.split()) * 1.3),  # Word-based estimation for consistency
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
        total_unchanged = sum(1 for r in results if r.get('skipped_unchanged'))
        total_dry_run = sum(1 for r in results if r.get('dry_run'))

        print(f"\nTotal chunks stored:    {total_stored:>5}")
        print(f"Total chunks skipped:   {total_skipped:>5}")
        print(f"Files unchanged (skip): {total_unchanged:>5}")
        if total_dry_run:
            print(f"Files in dry-run mode:  {total_dry_run:>5}")
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
                matches the manifest (use after chunker / embedding changes).
            dry_run: Run chunking only; do not embed, store, or update the
                manifest. Useful for chunker validation in CI.

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

            result = self.ingest_document(
                file_path,
                doc_metadata,
                force_reingest=force_reingest,
                dry_run=dry_run,
            )
            results.append(result)

        # Print summary
        self._print_summary(results)

        return results

    # ---- Manifest helpers (content-hash incremental ingestion) -----------

    def _compute_content_hash(self, file_path: Path) -> str:
        """Stable sha256 of the file's raw bytes. Used as the cache key in
        the ingestion manifest so unchanged files can be skipped without
        re-parsing, re-chunking, or re-embedding.
        """
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        except OSError as e:
            # Degrade gracefully — hash the filename + mtime so at least
            # identity is stable within a single run.
            stat = file_path.stat() if file_path.exists() else None
            fallback = f"{file_path.name}|{stat.st_mtime_ns if stat else 'missing'}|{e}"
            h.update(fallback.encode("utf-8"))
        return h.hexdigest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the ingestion manifest JSON, returning {} if it doesn't
        exist or can't be parsed.
        """
        if not self.manifest_path.exists():
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            self.logger.warning(
                "manifest at %s was not a dict; ignoring", self.manifest_path
            )
            return {}
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(
                "failed to read manifest %s (%s); starting fresh",
                self.manifest_path, e
            )
            return {}

    def _save_manifest(self) -> None:
        """Persist the manifest to disk. Best-effort — a failure here is
        logged but does NOT abort ingestion.
        """
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, sort_keys=True)
            tmp.replace(self.manifest_path)
        except OSError as e:
            self.logger.warning(
                "failed to save manifest %s: %s", self.manifest_path, e
            )

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

    def _store_chunk(
        self,
        text: str,
        dense_vector: List[float],
        metadata: Dict[str, Any]
    ):
        """Store chunk with embeddings in Qdrant."""
        # Generate unique ID — include preprocess_split_index so reranker
        # safety splits don't collide on the same Qdrant point id.
        chunk_id_str = "|".join([
            metadata.get("document_name", ""),
            str(metadata.get("section_index", 0)),
            str(metadata.get("sub_section_index", metadata.get("sub_index", 0))),
            str(metadata.get("section_letter", "")),
            str(metadata.get("chunk_type", "")),
            str(metadata.get("preprocess_split_index", 0)),
        ])
        chunk_id_int = int(hashlib.md5(chunk_id_str.encode()).hexdigest()[:16], 16)

        # Prepare payload
        payload = {
            "content": text,  # Use "content" to match BM25 index expectations
            "char_count": len(text),
            "token_count": int(len(text.split()) * 1.3),
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
        total_unchanged = sum(1 for r in results if r.get('skipped_unchanged'))
        total_dry_run = sum(1 for r in results if r.get('dry_run'))

        print(f"\nTotal chunks stored:    {total_stored:>5}")
        print(f"Total chunks skipped:   {total_skipped:>5}")
        print(f"Files unchanged (skip): {total_unchanged:>5}")
        if total_dry_run:
            print(f"Files in dry-run mode:  {total_dry_run:>5}")
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
