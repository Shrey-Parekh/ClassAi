"""
Sparse vector encoder using FastEmbed for Qdrant integration.

Replaces BM25 with native Qdrant sparse vectors for better performance.
"""

from typing import List, Dict, Any
import logging


class SparseEncoder:
    """
    Sparse vector encoder using FastEmbed.
    
    Generates sparse vectors for keyword-based retrieval in Qdrant.
    Replaces the old BM25 index with native sparse vector support.
    """
    
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        """
        Initialize sparse encoder.
        
        Args:
            model_name: FastEmbed model name (default: prithivida/Splade_PP_en_v1)
        """
        try:
            from fastembed import SparseTextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed required for sparse encoding. "
                "Install: pip install fastembed"
            )
        
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        try:
            self.model = SparseTextEmbedding(model_name=model_name)
            self.logger.info(f"✓ Loaded sparse encoder: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load sparse encoder: {e}")
            raise
    
    def encode(self, text: str) -> Dict[int, float]:
        """
        Encode text to sparse vector.
        
        Args:
            text: Text to encode
        
        Returns:
            Dict mapping token indices to weights
        """
        try:
            # FastEmbed 0.4.0 uses query_embed() for query encoding
            sparse_vectors = list(self.model.query_embed([text]))
            
            if sparse_vectors:
                sparse_vector = sparse_vectors[0]
                
                # Convert to Qdrant format (dict of index -> weight)
                if hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                    return dict(zip(sparse_vector.indices.tolist(), sparse_vector.values.tolist()))
                elif isinstance(sparse_vector, dict):
                    return sparse_vector
                else:
                    self.logger.warning(f"Unexpected sparse vector format: {type(sparse_vector)}")
                    return {}
            else:
                return {}
        
        except Exception as e:
            self.logger.error(f"Failed to encode sparse vector: {e}")
            return {}
    
    def encode_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Encode multiple texts to sparse vectors.
        
        Args:
            texts: List of texts to encode
        
        Returns:
            List of sparse vectors
        """
        try:
            sparse_vectors = []
            for text in texts:
                sparse_vector = self.encode(text)
                sparse_vectors.append(sparse_vector)
            return sparse_vectors
        
        except Exception as e:
            self.logger.error(f"Failed to encode batch: {e}")
            return [{} for _ in texts]
