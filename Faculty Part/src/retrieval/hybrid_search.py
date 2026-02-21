"""
Hybrid search combining vector similarity and BM25 keyword search.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    """Single search result with score and metadata."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # "vector" or "bm25" or "hybrid"


class HybridSearchEngine:
    """
    Combines vector similarity search with BM25 keyword search.
    
    Pipeline:
    1. Vector search for semantic similarity
    2. BM25 search for keyword matching
    3. Fusion of results with weighted scoring
    """
    
    def __init__(
        self,
        vector_db_client,
        collection_name: str,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            vector_db_client: Qdrant or similar vector DB client
            collection_name: Name of the vector collection
            vector_weight: Weight for vector similarity scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        self.vector_db = vector_db_client
        self.collection_name = collection_name
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # BM25 index (built from vector DB contents)
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_chunk_ids = []
    
    def build_bm25_index(self):
        """
        Build BM25 index from vector database contents.
        
        Call this after ingestion or periodically to refresh.
        """
        # Fetch all chunks from vector DB
        all_chunks = self._fetch_all_chunks()
        
        self.bm25_corpus = [chunk["content"] for chunk in all_chunks]
        self.bm25_chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]
        
        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)
    
    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25.
        
        Args:
            query: Text query
            query_embedding: Vector embedding of query
            top_k: Number of results to return
            filters: Metadata filters to apply
        
        Returns:
            List of search results sorted by hybrid score
        """
        # 1. Vector search
        vector_results = self._vector_search(
            query_embedding,
            top_k=top_k * 2,  # Get more for fusion
            filters=filters
        )
        
        # 2. BM25 search
        bm25_results = self._bm25_search(
            query,
            top_k=top_k * 2,
            filters=filters
        )
        
        # 3. Fuse results
        fused_results = self._fuse_results(
            vector_results,
            bm25_results,
            top_k=top_k
        )
        
        return fused_results
    
    def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Perform vector similarity search."""
        # Query vector database
        results = self.vector_db.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=self._build_filter(filters) if filters else None
        )
        
        return [
            SearchResult(
                chunk_id=hit.id,
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata=hit.payload,
                source="vector"
            )
            for hit in results
        ]
    
    def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Perform BM25 keyword search."""
        if not self.bm25_index:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            chunk_id = self.bm25_chunk_ids[idx]
            content = self.bm25_corpus[idx]
            score = scores[idx]
            
            # Fetch metadata from vector DB
            metadata = self._fetch_chunk_metadata(chunk_id)
            
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            results.append(SearchResult(
                chunk_id=chunk_id,
                content=content,
                score=score,
                metadata=metadata,
                source="bm25"
            ))
        
        return results
    
    def _fuse_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Fuse vector and BM25 results using weighted scoring.
        
        Uses Reciprocal Rank Fusion for combining rankings.
        """
        # Normalize scores to 0-1 range
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Combine results by chunk_id
        combined: Dict[str, SearchResult] = {}
        
        for result in vector_results:
            combined[result.chunk_id] = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score * self.vector_weight,
                metadata=result.metadata,
                source="hybrid"
            )
        
        for result in bm25_results:
            if result.chunk_id in combined:
                # Add BM25 score to existing
                combined[result.chunk_id].score += result.score * self.bm25_weight
            else:
                # New result from BM25 only
                combined[result.chunk_id] = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score * self.bm25_weight,
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
        """Build vector DB filter from metadata filters."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                for v in value:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=v))
                    )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions) if conditions else None
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True
    
    def _fetch_all_chunks(self) -> List[Dict[str, Any]]:
        """Fetch all chunks from vector DB for BM25 indexing."""
        try:
            # Scroll through all points in collection
            offset = None
            all_chunks = []
            
            while True:
                results, offset = self.vector_db.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in results:
                    all_chunks.append({
                        "chunk_id": point.id,
                        "content": point.payload.get("content", ""),
                        **point.payload
                    })
                
                if offset is None:
                    break
            
            return all_chunks
        except Exception as e:
            print(f"Warning: Could not fetch chunks for BM25: {e}")
            return []
    
    def _fetch_chunk_metadata(self, chunk_id: str) -> Dict[str, Any]:
        """Fetch metadata for a specific chunk."""
        try:
            point = self.vector_db.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True
            )
            if point:
                return point[0].payload
        except Exception:
            pass
        return {}
