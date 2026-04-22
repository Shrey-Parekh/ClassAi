"""
BM25 index persistence for faster startup.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from rank_bm25 import BM25Okapi
import hashlib
import json


class BM25PersistenceManager:
    """
    Manage BM25 index persistence.
    
    Features:
    - Save/load BM25 index to disk
    - Checksum validation
    - Automatic rebuild on corruption
    """
    
    def __init__(self, storage_dir: str = "./bm25_index", collection_name: str = "faculty_chunks"):
        """
        Initialize persistence manager.
        
        Args:
            storage_dir: Directory for BM25 index storage
            collection_name: Collection name for namespacing cache files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Namespace files by collection name
        self.index_file = self.storage_dir / f"bm25_index_{collection_name}.pkl"
        self.corpus_file = self.storage_dir / f"corpus_{collection_name}.pkl"
        self.ids_file = self.storage_dir / f"ids_{collection_name}.pkl"
        self.checksum_file = self.storage_dir / f"checksum_{collection_name}.txt"
        
        # Migration: rename legacy files if they exist and this is faculty_chunks
        if collection_name == "faculty_chunks":
            self._migrate_legacy_files()
    
    def save(
        self,
        bm25_index: BM25Okapi,
        corpus: List[List[str]],
        ids: List[str]
    ) -> bool:
        """
        Save BM25 index to disk.
        
        Args:
            bm25_index: BM25 index object
            corpus: Tokenized corpus
            ids: Document IDs
        
        Returns:
            True if successful
        """
        try:
            # Save index
            with open(self.index_file, 'wb') as f:
                pickle.dump(bm25_index, f)
            
            # Save corpus
            with open(self.corpus_file, 'wb') as f:
                pickle.dump(corpus, f)
            
            # Save IDs
            with open(self.ids_file, 'wb') as f:
                pickle.dump(ids, f)
            
            # Calculate checksum
            checksum = self._calculate_checksum(corpus, ids)
            with open(self.checksum_file, 'w') as f:
                f.write(checksum)
            
            self.logger.info(f"✓ BM25 index saved: {len(corpus)} documents")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save BM25 index: {e}")
            return False
    
    def load(self) -> Optional[Tuple[BM25Okapi, List[List[str]], List[str]]]:
        """
        Load BM25 index from disk.
        
        Returns:
            Tuple of (index, corpus, ids) or None if not found/corrupted
        """
        try:
            # Check if files exist
            if not all([
                self.index_file.exists(),
                self.corpus_file.exists(),
                self.ids_file.exists(),
                self.checksum_file.exists()
            ]):
                self.logger.info("BM25 index not found on disk")
                return None
            
            # Load corpus and IDs first
            with open(self.corpus_file, 'rb') as f:
                corpus = pickle.load(f)
            
            with open(self.ids_file, 'rb') as f:
                ids = pickle.load(f)
            
            # Verify checksum
            with open(self.checksum_file, 'r') as f:
                stored_checksum = f.read().strip()
            
            current_checksum = self._calculate_checksum(corpus, ids)
            
            if stored_checksum != current_checksum:
                self.logger.warning("BM25 index checksum mismatch, rebuild required")
                return None
            
            # Load index
            with open(self.index_file, 'rb') as f:
                bm25_index = pickle.load(f)
            
            self.logger.info(f"✓ BM25 index loaded: {len(corpus)} documents")
            return bm25_index, corpus, ids
        
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {e}")
            return None
    
    def clear(self) -> bool:
        """Clear saved BM25 index."""
        try:
            for file in [self.index_file, self.corpus_file, self.ids_file, self.checksum_file]:
                if file.exists():
                    file.unlink()
            
            self.logger.info("BM25 index cleared")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to clear BM25 index: {e}")
            return False
    
    def _calculate_checksum(self, corpus: List[List[str]], ids: List[str]) -> str:
        """Calculate checksum for corpus and IDs — hashes every id and first 64 tokens."""
        h = hashlib.sha256()
        for cid, tokens in zip(ids, corpus):
            h.update(cid.encode())
            h.update(b"\0")
            h.update((" ".join(tokens[:64])).encode())
            h.update(b"\n")
        return h.hexdigest()
    
    def _migrate_legacy_files(self):
        """Migrate legacy unnamed cache files to collection-namespaced files."""
        legacy_files = {
            "bm25_index.pkl": self.index_file,
            "corpus.pkl": self.corpus_file,
            "ids.pkl": self.ids_file,
            "checksum.txt": self.checksum_file
        }
        
        for legacy_name, new_path in legacy_files.items():
            legacy_path = self.storage_dir / legacy_name
            if legacy_path.exists() and not new_path.exists():
                try:
                    legacy_path.rename(new_path)
                    self.logger.info(f"Migrated {legacy_name} → {new_path.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to migrate {legacy_name}: {e}")
