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
        max_per_doc: int = 3,
        intent: str = "general"
    ) -> List[Any]:
        """
        Rerank search results using cross-encoder with intent-aware diversity cap.
        """
        if not results:
            return []

        # R16: intent-aware diversity cap
        INTENT_MAX_PER_DOC = {
            "person_lookup": 8,
            "lookup": 8,
            "topic_search": 3,
            "policy_lookup": 5,
            "procedure": 6,
            "eligibility": 5,
            "general": 4,
        }
        effective_max = INTENT_MAX_PER_DOC.get(intent.lower(), max_per_doc)

        try:
            pairs = [[query, result.content] for result in results]
            scores = self.model.compute_score(pairs, normalize=True)

            if not isinstance(scores, list):
                scores = [scores]

            for result, score in zip(results, scores):
                result.score = float(score)
                result.source = "reranked"

            reranked = sorted(results, key=lambda x: x.score, reverse=True)
            return self._apply_diversity_cap(reranked, effective_max, top_k)

        except Exception as e:
            self.logger.error("Reranking failed: %s", e)
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
