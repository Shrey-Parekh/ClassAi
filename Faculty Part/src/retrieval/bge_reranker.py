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
        top_k: int = 15,
        max_per_doc: int = 3
    ) -> List[Any]:
        """
        Rerank search results using cross-encoder with diversity cap.
        
        Args:
            query: Original query text
            results: List of SearchResult objects from hybrid search
            top_k: Number of top results to return
            max_per_doc: Maximum chunks from any single document (default: 3)
        
        Returns:
            Reranked list of SearchResult objects with diversity applied
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
            
            # Sort by reranker score
            reranked = sorted(results, key=lambda x: x.score, reverse=True)
            
            # Apply diversity cap: max N chunks per doc_id
            diverse_results = self._apply_diversity_cap(reranked, max_per_doc, top_k)
            
            return diverse_results
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            # Fallback: return original results
            return results[:top_k]
    
    def _apply_diversity_cap(
        self,
        results: List[Any],
        max_per_doc: int,
        top_k: int
    ) -> List[Any]:
        """
        Apply source diversity cap to prevent document monopolization.
        
        Args:
            results: Sorted results by score
            max_per_doc: Maximum chunks from any single doc_id
            top_k: Target number of results
        
        Returns:
            Diverse results list
        """
        doc_counts = {}
        diverse_results = []
        
        for result in results:
            # Get doc_id from metadata (prioritize document_name which is always set)
            doc_id = result.metadata.get("document_name", result.metadata.get("doc_id", result.metadata.get("parent_doc_id", "unknown")))
            
            # Check if we've hit the cap for this document
            current_count = doc_counts.get(doc_id, 0)
            
            if current_count < max_per_doc:
                diverse_results.append(result)
                doc_counts[doc_id] = current_count + 1
                
                # Stop when we have enough results
                if len(diverse_results) >= top_k:
                    break
        
        self.logger.info(f"Diversity cap applied: {len(diverse_results)} results from {len(doc_counts)} documents")
        
        return diverse_results
