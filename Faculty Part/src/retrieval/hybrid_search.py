"""
Hybrid search combining dense vectors and BM25 sparse retrieval.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import re
import threading
from rank_bm25 import BM25Okapi

from ..utils.bm25_persistence import BM25PersistenceManager


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
    
    # Intent-based weight mapping.
    # person_lookup  → names/departments: sparse-heavy for exact string hits
    # policy_lookup  → policy/form-code queries: balanced semantic+keyword
    # topic_search   → thematic discovery: dense-heavy
    # procedure      → steps/how-to: balanced
    # eligibility    → rule queries: balanced
    # general        → fallback: dense-leaning
    INTENT_WEIGHTS = {
        "person_lookup": {"dense": 0.40, "sparse": 0.60},
        "policy_lookup": {"dense": 0.60, "sparse": 0.40},
        "lookup": {"dense": 0.40, "sparse": 0.60},  # legacy alias → person-biased
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
        
        # BM25 index (built on demand)
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_ids = []
        
        # BM25 persistence manager (namespaced by collection)
        self.bm25_persistence = BM25PersistenceManager(collection_name=collection_name)
        
        # Thread safety for BM25 operations
        self._bm25_lock = threading.Lock()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 with punctuation handling.
        
        Handles:
        - Form codes like HR-LA-01
        - Abbreviations like Dr., Rs.
        - Parenthetical text like (EL)
        - Table content with pipes and multiple spaces
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of cleaned tokens
        """
        # Remove common punctuation that breaks matching
        text = re.sub(r'[(){}[\]|•·,;:!?"]', ' ', text.lower())
        # Keep hyphens in form codes (HR-LA-01) but split on standalone hyphens
        text = re.sub(r'\s+-\s+', ' ', text)
        # Keep periods in abbreviations (Dr., Rs.) but split on sentence periods
        text = re.sub(r'\.(\s|$)', ' ', text)
        # Split and filter empty tokens
        return [token for token in text.split() if len(token) > 1]
    
    def search(
        self,
        original_query: str,
        expanded_query: str,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Dict[str, Any] = None,
        name_embedding: List[float] = None,
        name_boost: float = 0.0,
        name_filters: Dict[str, Any] = None,
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
        
        # 2. Name-boosted search (R5: use name_filters, not main filters)
        name_results = []
        if name_embedding is not None and name_boost > 0:
            name_results = self._dense_search(
                name_embedding,
                top_k=top_k,
                filters=name_filters  # R5: no domain filter on name search
            )
        
        # 3. Sparse search using expanded query for keywords
        sparse_results = []
        if self.bm25_index is not None:
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
                qdrant_filter = self.vector_db._build_filter(filters)
            
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
                    # Handle both Faculty chunks (content) and Student chunks (page_content)
                    content=hit.payload.get("content") or hit.payload.get("page_content", ""),
                    score=hit.score,
                    metadata=hit.payload,
                    source="dense"
                )
                for hit in results
            ]
        
        except Exception as e:
            self.logger.error(f"Dense search failed: {e}")
            return []
    
    def build_bm25_index(self, force_rebuild: bool = False):
        """
        Build BM25 index from all documents in collection with persistence.
        
        Args:
            force_rebuild: Force rebuild even if cached index exists
        """
        try:
            # Try loading from disk first
            if not force_rebuild:
                loaded = self.bm25_persistence.load()
                if loaded:
                    self.bm25_index, self.bm25_corpus, self.bm25_ids = loaded
                    return
            
            self.logger.info("Building BM25 index from scratch...")
            
            # Scroll through all documents in collection
            all_points = []
            offset = None
            
            while True:
                results, offset = self.vector_db.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                all_points.extend(results)
                
                if offset is None:
                    break
            
            # Extract corpus and IDs
            self.bm25_corpus = []
            self.bm25_ids = []
            
            for point in all_points:
                # Handle both Faculty chunks (content) and Student chunks (page_content)
                content = point.payload.get("content") or point.payload.get("page_content", "")
                if content:
                    # Tokenize for BM25 with punctuation handling
                    tokens = self._tokenize(content)
                    self.bm25_corpus.append(tokens)
                    self.bm25_ids.append(point.id)
            
            # Build BM25 index
            if self.bm25_corpus:
                self.bm25_index = BM25Okapi(self.bm25_corpus)
                self.logger.info(f"✓ BM25 index built with {len(self.bm25_corpus)} documents")
                
                # Save to disk for next startup
                self.bm25_persistence.save(self.bm25_index, self.bm25_corpus, self.bm25_ids)
            else:
                self.logger.warning("No documents found for BM25 index")
                
        except Exception as e:
            self.logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Perform BM25 sparse search.
        
        Args:
            query: Query text
            top_k: Number of results
            filters: Metadata filters (not supported in BM25)
        
        Returns:
            List of search results
        """
        try:
            if self.bm25_index is None:
                self.logger.debug("BM25 index not built, skipping sparse search")
                return []
            
            # Tokenize query with punctuation handling
            query_tokens = self._tokenize(query)
            
            # Get BM25 scores with thread safety
            with self._bm25_lock:
                scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            # Fetch full documents from Qdrant
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include non-zero scores
                    chunk_id = self.bm25_ids[idx]
                    
                    # Fetch from Qdrant
                    try:
                        point = self.vector_db.client.retrieve(
                            collection_name=self.collection_name,
                            ids=[chunk_id],
                            with_payload=True
                        )[0]
                        
                        results.append(SearchResult(
                            chunk_id=chunk_id,
                            # Handle both Faculty chunks (content) and Student chunks (page_content)
                            content=point.payload.get("content") or point.payload.get("page_content", ""),
                            score=float(scores[idx]),
                            metadata=point.payload,
                            source="sparse"
                        ))
                    except Exception as e:
                        self.logger.debug(f"Failed to retrieve chunk {chunk_id}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
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
        Fuse dense, sparse, and name results using Reciprocal Rank Fusion (RRF).

        RRF score = Σ weight_i / (k + rank_i)  where k=60.
        Scale-free — handles score distribution skew without per-list normalization.
        Intent weights scale each list's contribution.
        """
        RRF_K = 60

        # Build a lookup: chunk_id → best SearchResult (for content/metadata)
        best: Dict[str, SearchResult] = {}
        scores: Dict[str, float] = {}

        def _add_list(results: List[SearchResult], weight: float) -> None:
            for rank, result in enumerate(results, start=1):
                cid = str(result.chunk_id)
                scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank)
                if cid not in best:
                    best[cid] = result

        _add_list(dense_results, dense_weight)
        _add_list(sparse_results, sparse_weight)
        if name_results and name_boost > 0:
            _add_list(name_results, name_boost)

        # Build final list sorted by RRF score
        sorted_ids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
        fused = []
        for cid in sorted_ids:
            r = best[cid]
            fused.append(SearchResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=scores[cid],
                metadata=r.metadata,
                source="hybrid"
            ))
        return fused
