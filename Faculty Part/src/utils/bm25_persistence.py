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
    
    def __init__(self, storage_dir: str = "./bm25_index"):
        """
        Initialize persistence manager.
        
        Args:
            storage_dir: Directory for BM25 index storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.index_file = self.storage_dir / "bm25_index.pkl"
        self.corpus_file = self.storage_dir / "corpus.pkl"
        self.ids_file = self.storage_dir / "ids.pkl"
        self.checksum_file = self.storage_dir / "checksum.txt"
    
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
        """Calculate checksum for corpus and IDs."""
        # Create deterministic representation
        data = {
            "corpus_size": len(corpus),
            "ids_size": len(ids),
            "sample_ids": ids[:10] if len(ids) > 10 else ids
        }
        
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
