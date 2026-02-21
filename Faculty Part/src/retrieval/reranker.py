"""
Cross-encoder reranking for final result selection.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class RankedResult:
    """Result after reranking with relevance score."""
    chunk_id: str
    content: str
    relevance_score: float
    metadata: dict
    original_rank: int


class CrossEncoderReranker:
    """
    Rerank search results using a cross-encoder model.
    
    Cross-encoders jointly encode query and document for better
    relevance scoring than bi-encoders (used in vector search).
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with cross-encoder model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model = None  # Lazy load
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
    
    def rerank(
        self,
        query: str,
        results: List,
        top_k: int = 5
    ) -> List[RankedResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Original query text
            results: List of SearchResult objects from hybrid search
            top_k: Number of top results to return
        
        Returns:
            List of reranked results with relevance scores
        """
        if not results:
            return []
        
        self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]
        
        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Create ranked results
        ranked = []
        for idx, (result, score) in enumerate(zip(results, scores)):
            ranked.append(RankedResult(
                chunk_id=result.chunk_id,
                content=result.content,
                relevance_score=float(score),
                metadata=result.metadata,
                original_rank=idx
            ))
        
        # Sort by relevance score
        ranked.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return ranked[:top_k]
