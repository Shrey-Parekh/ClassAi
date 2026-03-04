"""
Single reliable encoder using BAAI/bge-large-en-v1.5.

Direct sentence-transformers loading — no HTTP, no fallback, no silent failures.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
import numpy as np

from sentence_transformers import SentenceTransformer
import torch


class DualEncoderEmbeddings:
    """
    Single encoder wrapper using BAAI/bge-large-en-v1.5.
    
    Loads model directly via sentence-transformers.
    No Ollama HTTP calls. No fallback logic. No silent failures.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        log_file: str = "embedding_log.jsonl"
    ):
        """
        Initialize single encoder.
        
        Args:
            model_name: Sentence Transformers model name
            log_file: Path to embedding log file
        """
        self.model_name = model_name
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            "embedded": 0,
            "discarded": 0,
            "split": 0,
        }
        
        # Load model
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name, device=device)
            self.logger.info(f"✓ Encoder loaded: {model_name} on {device}")
            
            # Test embedding
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            dimension = len(test_embedding)
            self.logger.info(f"✓ Encoder ready: {model_name} ({dimension}d)")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to load encoder: {e}")
            raise
    
    def embed(
        self,
        text: str,
        chunk_id: str,
        source_file: str,
        chunk_length_chars: int
    ) -> Tuple[Optional[List[float]], Dict[str, Any]]:
        """
        Embed text using single encoder.
        
        Args:
            text: Cleaned chunk text
            chunk_id: Unique chunk identifier
            source_file: Source document name
            chunk_length_chars: Character count of cleaned text
        
        Returns:
            Tuple of (embedding_vector, metadata_dict)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "chunk_id": chunk_id,
            "source_file": source_file,
            "chunk_length_chars": chunk_length_chars,
            "encoder": self.model_name,
            "encode_result": None,
            "encode_error": None,
            "final_result": None,
        }
        
        metadata = {
            "embedding_model": self.model_name,
            "chunk_length_chars": chunk_length_chars,
            "was_split": False,
            "split_index": None,
        }
        
        try:
            # Encode
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Validate
            if embedding is None or len(embedding) == 0:
                log_entry["encode_result"] = "failure"
                log_entry["encode_error"] = "Empty embedding returned"
                log_entry["final_result"] = "discarded"
                self.stats["discarded"] += 1
                self._write_log(log_entry)
                return None, metadata
            
            # Check for NaN or Inf
            if any(np.isnan(v) or np.isinf(v) for v in embedding):
                log_entry["encode_result"] = "failure"
                log_entry["encode_error"] = "Embedding contains NaN or Inf"
                log_entry["final_result"] = "discarded"
                self.stats["discarded"] += 1
                self._write_log(log_entry)
                return None, metadata
            
            # Success
            log_entry["encode_result"] = "success"
            log_entry["final_result"] = "embedded"
            self.stats["embedded"] += 1
            self._write_log(log_entry)
            
            return embedding.tolist(), metadata
        
        except Exception as e:
            log_entry["encode_result"] = "failure"
            log_entry["encode_error"] = str(e)[:200]
            log_entry["final_result"] = "discarded"
            self.stats["discarded"] += 1
            self._write_log(log_entry)
            self.logger.error(f"Embedding failed for {chunk_id}: {e}")
            return None, metadata
    
    def _write_log(self, entry: Dict[str, Any]):
        """Write log entry to JSONL file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write log entry: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get embedding statistics."""
        return self.stats.copy()
    
    def print_summary(self):
        """Print ingestion summary."""
        total = sum(self.stats.values())
        print("\n" + "="*60)
        print("EMBEDDING INGESTION COMPLETE")
        print("="*60)
        print(f"Encoder: {self.model_name}")
        print(f"✓ Embedded:  {self.stats['embedded']} chunks")
        print(f"✗ Discarded: {self.stats['discarded']} chunks")
        print(f"↔ Split:     {self.stats['split']} chunks")
        print(f"\nTotal processed: {total} chunks")
        print(f"Full log: {self.log_file}")
        print("="*60 + "\n")
