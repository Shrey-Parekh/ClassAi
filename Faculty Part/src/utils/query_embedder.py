"""
Query embedding using BAAI/bge-m3.

Same model as document embeddings for semantic consistency.
"""

from typing import List
import logging
from sentence_transformers import SentenceTransformer
import torch


class QueryEmbedder:
    """
    Query embedding using BAAI/bge-m3.
    
    Matches document embedding model for semantic consistency.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize query embedder.
        
        Args:
            model_name: Sentence Transformers model name
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name, device=device)
            self.logger.info(f"✓ Query embedder loaded: {model_name} on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load query embedder: {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """
        Embed query text.
        
        Args:
            text: Query text
        
        Returns:
            Embedding vector (1024 dimensions)
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Query embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension (BAAI/bge-m3 = 1024)."""
        return 1024
