"""
BGE Reranker for cross-encoder based relevance scoring.
"""

from typing import List, Dict, Any
import logging


class BGEReranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.
    
    Reranks retrieved chunks using a cross-encoder model that scores
    query-document pairs directly for better relevance.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize BGE reranker.
        
        Args:
            model_name: Reranker model name
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        try:
            from FlagEmbedding import FlagReranker
            
            self.model = FlagReranker(model_name, use_fp16=True)
            self.logger.info(f"✓ Loaded reranker: {model_name}")
            
        except ImportError:
            self.logger.error("FlagEmbedding not installed. Install: pip install FlagEmbedding")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load reranker: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: int = 15
    ) -> List[Any]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Original query text
            results: List of SearchResult objects from hybrid search
            top_k: Number of top results to return
        
        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []
        
        try:
            # Prepare query-document pairs
            pairs = [[query, result.content] for result in results]
            
            # Get reranker scores
            scores = self.model.compute_score(pairs, normalize=True)
            
            # Handle single result case (scores is float, not list)
            if not isinstance(scores, list):
                scores = [scores]
            
            # Update result scores with reranker scores
            for result, score in zip(results, scores):
                result.score = float(score)
                result.source = "reranked"
            
            # Sort by reranker score and return top-k
            reranked = sorted(results, key=lambda x: x.score, reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            # Fallback: return original results
            return results[:top_k]
