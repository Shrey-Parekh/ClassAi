"""
Complete retrieval pipeline orchestrating all components.
"""

from typing import Dict, Any, List
from config.chunking_config import TOP_K_RERANKED

from .query_understanding import QueryAnalyzer
from .hybrid_search import HybridSearchEngine
from .bge_reranker import BGEReranker


class RetrievalPipeline:
    """
    Enhanced retrieval pipeline with query understanding and BGE reranking.
    
    Pipeline steps:
    1. Query understanding (intent + domain + entities)
    2. Metadata pre-filtering (domain + is_current)
    3. Query embedding (BAAI/bge-large-en-v1.5)
    4. Hybrid search (dense + sparse) → top 40
    5. BGE reranking → top 5
    6. Return final chunks
    
    Embedding model: BAAI/bge-large-en-v1.5 (1024 dimensions)
    - Documents: Embedded with BAAI/bge-large-en-v1.5
    - Queries: Embedded with BAAI/bge-large-en-v1.5
    - Consistency: Guaranteed (same model for both)
    """
    
    def __init__(
        self,
        vector_db_client,
        embedding_model,
        collection_name: str = "faculty_chunks"
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            vector_db_client: Vector database client (Qdrant)
            embedding_model: Query embedding model (BAAI/bge-large-en-v1.5)
            collection_name: Vector DB collection name
        """
        self.query_analyzer = QueryAnalyzer()
        self.search_engine = HybridSearchEngine(
            vector_db_client=vector_db_client,
            collection_name=collection_name
        )
        self.reranker = BGEReranker()
        self.embedding_model = embedding_model
        self.vector_db = vector_db_client
        self.collection_name = collection_name
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute complete retrieval pipeline.
        
        Args:
            query: Faculty query text
            top_k: Number of final results to return (default: 5)
        
        Returns:
            Dict containing:
            - chunks: List of retrieved chunks (top 5 after reranking)
            - intent: Detected intent type
            - domain: Detected domain
            - entities: Extracted entities
            - metadata: Pipeline metadata
        """
        # Step 1: Query understanding
        understanding = self.query_analyzer.analyze(query)
        
        # Step 2: Generate query embedding using EXPANDED query for better retrieval
        query_embedding = self.embedding_model.embed(understanding.expanded_query)
        
        # Step 3: Adjust retrieval parameters based on intent
        # For procedures, retrieve more chunks for better context
        if understanding.intent == "procedure":
            top_k_search = 72  # More candidates for procedures
            top_k_rerank = 12  # More final chunks for procedures
        else:
            top_k_search = 64  # Default for lookups
            top_k_rerank = 8  # Optimized for lookups
        
        # Step 4: Hybrid search with metadata pre-filtering using expanded query
        search_results = self.search_engine.search(
            query=understanding.expanded_query,
            query_embedding=query_embedding,
            top_k=top_k_search,
            filters=understanding.metadata_filters
        )
        
        # Step 5: BGE reranking using ORIGINAL query for relevance scoring
        reranked_results = self.reranker.rerank(
            query=query,
            results=search_results,
            top_k=top_k_rerank
        )
        
        # Format results
        chunks = [
            {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "score": result.relevance_score,
                "metadata": result.metadata,
            }
            for result in reranked_results
        ]
        
        return {
            "chunks": chunks,
            "intent": understanding.intent,
            "domain": understanding.domain,
            "entities": understanding.entities,
            "metadata": {
                "initial_results": len(search_results),
                "final_results": len(chunks),
                "filters_applied": understanding.metadata_filters,
                "is_current_only": understanding.is_current_only
            }
        }
