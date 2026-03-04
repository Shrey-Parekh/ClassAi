"""
Hybrid search combining dense vectors and sparse vectors.

Uses Qdrant native sparse vectors (FastEmbed) instead of BM25.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from ..utils.sparse_encoder import SparseEncoder


@dataclass
class SearchResult:
    """Single search result with score and metadata."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # "dense", "sparse", or "hybrid"


class HybridSearchEngine:
    """
    Hybrid search combining dense and sparse vectors.
    
    Pipeline:
    1. Dense search (semantic similarity via embeddings)
    2. Sparse search (keyword matching via FastEmbed)
    3. Fusion of results with weighted scoring
    """
    
    def __init__(
        self,
        vector_db_client,
        collection_name: str,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            vector_db_client: Qdrant client
            collection_name: Collection name
            dense_weight: Weight for dense search (0-1) - default 60%
            sparse_weight: Weight for sparse search (0-1) - default 40%
        """
        self.vector_db = vector_db_client
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize sparse encoder
        try:
            self.sparse_encoder = SparseEncoder(model_name="Splade")
        except Exception as e:
            self.logger.warning(f"Failed to initialize sparse encoder: {e}")
            self.sparse_encoder = None
    
    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse.
        
        Args:
            query: Text query
            query_embedding: Dense vector embedding
            top_k: Number of results to return
            filters: Metadata filters
        
        Returns:
            List of search results sorted by hybrid score
        """
        # 1. Dense search
        dense_results = self._dense_search(
            query_embedding,
            top_k=top_k * 2,
            filters=filters
        )
        
        # 2. Sparse search (if encoder available)
        sparse_results = []
        if self.sparse_encoder:
            sparse_results = self._sparse_search(
                query,
                top_k=top_k * 2,
                filters=filters
            )
        
        # 3. Fuse results
        fused_results = self._fuse_results(
            dense_results,
            sparse_results,
            top_k=top_k
        )
        
        return fused_results
    
    def _dense_search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Perform dense vector search."""
        try:
            # Build filter if provided
            qdrant_filter = None
            if filters and isinstance(filters, dict):
                qdrant_filter = self._build_filter(filters)
            
            # Query Qdrant
            results = self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            )
            
            return [
                SearchResult(
                    chunk_id=hit.id,
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    metadata=hit.payload,
                    source="dense"
                )
                for hit in results
            ]
        
        except Exception as e:
            self.logger.error(f"Dense search failed: {e}")
            return []
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Perform sparse vector search."""
        try:
            if not self.sparse_encoder:
                return []
            
            # Encode query to sparse vector
            sparse_query = self.sparse_encoder.encode(query)
            
            if not sparse_query:
                return []
            
            # Build filter if provided
            qdrant_filter = None
            if filters and isinstance(filters, dict):
                qdrant_filter = self._build_filter(filters)
            
            # Query Qdrant with sparse vector
            # Note: Qdrant sparse search API may vary by version
            # This is a placeholder for the actual sparse search call
            results = self.vector_db.search_sparse(
                collection_name=self.collection_name,
                query_vector=sparse_query,
                limit=top_k,
                query_filter=qdrant_filter
            )
            
            return [
                SearchResult(
                    chunk_id=hit.id,
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    metadata=hit.payload,
                    source="sparse"
                )
                for hit in results
            ]
        
        except Exception as e:
            self.logger.warning(f"Sparse search not available or failed: {e}")
            return []
    
    def _fuse_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Fuse dense and sparse results using weighted scoring.
        """
        # Normalize scores
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)
        
        # Combine results by chunk_id
        combined: Dict[str, SearchResult] = {}
        
        for result in dense_results:
            combined[result.chunk_id] = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score * self.dense_weight,
                metadata=result.metadata,
                source="hybrid"
            )
        
        for result in sparse_results:
            if result.chunk_id in combined:
                # Add sparse score to existing
                combined[result.chunk_id].score += result.score * self.sparse_weight
            else:
                # New result from sparse only
                combined[result.chunk_id] = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score * self.sparse_weight,
                    metadata=result.metadata,
                    source="hybrid"
                )
        
        # Sort by combined score and return top-k
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        
        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        
        if max_score == min_score:
            return results
        
        for result in results:
            result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def _build_filter(self, filters: Dict[str, Any]):
        """Build Qdrant filter from metadata filters."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            
            if isinstance(value, list):
                for v in value:
                    if v is not None:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=v))
                        )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions) if conditions else None
