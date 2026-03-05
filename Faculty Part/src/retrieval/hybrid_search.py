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
    3. Fusion of results with intent-based dynamic weighting
    """
    
    # Intent-based weight mapping
    INTENT_WEIGHTS = {
        "person_lookup": {"dense": 0.40, "sparse": 0.60},
        "lookup": {"dense": 0.40, "sparse": 0.60},
        "topic_search": {"dense": 0.80, "sparse": 0.20},
        "procedure": {"dense": 0.60, "sparse": 0.40},
        "eligibility": {"dense": 0.60, "sparse": 0.40},
        "general": {"dense": 0.70, "sparse": 0.30},
    }
    
    def __init__(
        self,
        vector_db_client,
        collection_name: str
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            vector_db_client: Qdrant client
            collection_name: Collection name
        """
        self.vector_db = vector_db_client
        self.collection_name = collection_name
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize sparse encoder
        try:
            self.sparse_encoder = SparseEncoder(model_name="Splade")
        except Exception as e:
            self.logger.warning(f"Failed to initialize sparse encoder: {e}")
            self.sparse_encoder = None
    
    def search(
        self,
        original_query: str,
        expanded_query: str,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Dict[str, Any] = None,
        name_embedding: List[float] = None,
        name_boost: float = 0.0,
        intent: str = "general"
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and sparse with intent-based weighting.
        
        Args:
            original_query: Original clean query (for sparse search)
            expanded_query: Expanded query with keywords (for sparse search)
            query_embedding: Dense vector embedding (from original query)
            top_k: Number of results to return
            filters: Metadata filters
            name_embedding: Optional name-focused embedding for faculty queries
            name_boost: Boost factor for name-based results (0.0-1.0)
            intent: Query intent for dynamic weight selection
        
        Returns:
            List of search results sorted by hybrid score
        """
        # Get intent-based weights (fallback to general if not found)
        weights = self.INTENT_WEIGHTS.get(intent.lower(), self.INTENT_WEIGHTS["general"])
        dense_weight = weights["dense"]
        sparse_weight = weights["sparse"]
        
        # 1. Dense search with original query embedding
        dense_results = self._dense_search(
            query_embedding,
            top_k=top_k * 2,
            filters=filters
        )
        
        # 2. Name-boosted search (if name embedding provided)
        name_results = []
        if name_embedding is not None and name_boost > 0:
            name_results = self._dense_search(
                name_embedding,
                top_k=top_k,
                filters=filters
            )
        
        # 3. Sparse search using expanded query for keywords
        sparse_results = []
        if self.sparse_encoder:
            # Use expanded query for better keyword coverage
            sparse_results = self._sparse_search(
                expanded_query,
                top_k=top_k * 2,
                filters=filters
            )
        
        # 4. Fuse results with intent-based weights
        fused_results = self._fuse_results(
            dense_results,
            sparse_results,
            name_results,
            name_boost,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
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
        """
        Perform sparse vector search using expanded query for keywords.
        
        Note: Sparse search is currently not fully implemented in Qdrant client.
        This is a placeholder that will be activated when Qdrant sparse vectors are available.
        """
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
            
            # NOTE: Qdrant sparse search is not yet available in the client
            # This will be enabled when the API is ready
            # For now, we rely on dense search only
            self.logger.debug("Sparse search not available - using dense search only")
            return []
            
            # Future implementation:
            # results = self.vector_db.search_sparse(
            #     collection_name=self.collection_name,
            #     query_vector=sparse_query,
            #     limit=top_k,
            #     query_filter=qdrant_filter
            # )
            
        except Exception as e:
            self.logger.debug(f"Sparse search not available: {e}")
            return []
    
    def _fuse_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        name_results: List[SearchResult],
        name_boost: float,
        top_k: int,
        dense_weight: float,
        sparse_weight: float
    ) -> List[SearchResult]:
        """
        Fuse dense, sparse, and name-based results using intent-based weighted scoring.
        
        Scores are normalized to 0-1 range to maintain consistency across intents.
        
        Args:
            dense_results: Results from dense semantic search
            sparse_results: Results from sparse keyword search
            name_results: Results from name-focused search
            name_boost: Boost factor for name matches
            top_k: Number of final results to return
            dense_weight: Weight for dense results (intent-based)
            sparse_weight: Weight for sparse results (intent-based)
        """
        # Normalize scores
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)
        name_results = self._normalize_scores(name_results)
        
        # Calculate maximum possible score for normalization
        max_possible_score = dense_weight + sparse_weight
        has_name_boost = len(name_results) > 0 and name_boost > 0
        if has_name_boost:
            max_possible_score += name_boost
        
        # Combine results by chunk_id
        combined: Dict[str, SearchResult] = {}
        
        # Add dense results with intent-based weight
        for result in dense_results:
            combined[result.chunk_id] = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score * dense_weight,
                metadata=result.metadata,
                source="hybrid"
            )
        
        # Add sparse results with intent-based weight
        for result in sparse_results:
            if result.chunk_id in combined:
                # Add sparse score to existing
                combined[result.chunk_id].score += result.score * sparse_weight
            else:
                # New result from sparse only
                combined[result.chunk_id] = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score * sparse_weight,
                    metadata=result.metadata,
                    source="hybrid"
                )
        
        # Add name-based boost
        if has_name_boost:
            for result in name_results:
                if result.chunk_id in combined:
                    # Boost existing result
                    combined[result.chunk_id].score += result.score * name_boost
                else:
                    # New result from name search
                    combined[result.chunk_id] = SearchResult(
                        chunk_id=result.chunk_id,
                        content=result.content,
                        score=result.score * name_boost,
                        metadata=result.metadata,
                        source="hybrid"
                    )
        
        # Normalize all scores to 0-1 range
        for result in combined.values():
            result.score = result.score / max_possible_score
        
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
