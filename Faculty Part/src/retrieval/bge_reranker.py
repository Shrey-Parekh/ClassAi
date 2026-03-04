"""
BGE-based reranker for improved relevance scoring.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result with relevance score."""
    chunk_id: str
    content: str
    score: float
    metadata: dict
    relevance_score: float = 0.0


class BGEReranker:
    """
    Reranks search results using BGE (BAAI General Embedding) reranker.
    
    BGE reranker reads query + chunk together for better relevance scoring.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize BGE reranker.
        
        Args:
            model_name: BGE reranker model name
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required for BGE reranker. "
                "Install: pip install sentence-transformers"
            )
        
        self.model = CrossEncoder(model_name)
        print(f"✓ Initialized BGE reranker: {model_name}")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank search results using BGE cross-encoder.
        
        Args:
            query: User query
            results: List of search results from hybrid search
            top_k: Number of top results to return
        
        Returns:
            Reranked list of top-k results
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, result.content] for result in results]
        
        # Get relevance scores from BGE
        scores = self.model.predict(pairs)
        
        # Update results with new scores
        for result, score in zip(results, scores):
            result.relevance_score = float(score)
        
        # Sort by relevance score and return top-k
        reranked = sorted(
            results,
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        return reranked[:top_k]
