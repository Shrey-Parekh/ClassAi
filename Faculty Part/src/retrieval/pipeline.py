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
    4. Hybrid search (dense + sparse) → 70-80 candidates
    5. BGE reranking → 12 final chunks
    6. Return final chunks
    
    Token Budget Optimization (8,192 total):
    - System prompt: ~800 tokens
    - Query: ~200 tokens
    - Context: 12 × 490 = 5,880 tokens (~6,100 with overhead)
    - Output reserve: ~1,000 tokens
    - Total: ~8,000 tokens (98% utilization)
    
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
        
        # Step 2: Generate query embedding using ORIGINAL query for clean semantic signal
        # CRITICAL: Do NOT use expanded query for embedding - it dilutes semantic meaning
        query_embedding = self.embedding_model.embed(query)
        
        # Step 3: For faculty name queries, create name-focused embedding
        name_embedding = None
        name_boost = 0.0
        
        if understanding.intent == "lookup" and understanding.entities:
            # Extract faculty name (first entity)
            faculty_name = understanding.entities[0]
            
            # Create name-focused query for better matching
            # Format: "Faculty: [Name]" to match chunk format
            name_query = f"Faculty: {faculty_name}"
            name_embedding = self.embedding_model.embed(name_query)
            name_boost = 0.3  # 30% boost for name-based matches
        
        # Step 4: Adjust retrieval parameters based on intent
        # Optimized to maximize 8,192 token context window usage
        # Available tokens: 8,192 - 800 (system) - 200 (query) - 1,000 (output) = 6,192 tokens
        
        if understanding.intent == "procedure":
            # Procedure queries: Use ~6,100 tokens for comprehensive context
            top_k_search = 80  # More candidates for procedures
            top_k_rerank = 12  # 12 × 490 = 5,880 tokens + overhead = ~6,100 tokens
        else:
            # Lookup queries: Use ~6,000 tokens for detailed answers
            top_k_search = 70  # More candidates for lookups
            top_k_rerank = 12  # 12 × 490 = 5,880 tokens + overhead = ~6,000 tokens
        
        # Step 5: Hybrid search with metadata pre-filtering
        # - Dense search uses ORIGINAL query embedding (clean semantic signal)
        # - Sparse search uses EXPANDED query (keyword coverage)
        # - Name search uses NAME-FOCUSED embedding (for faculty queries)
        search_results = self.search_engine.search(
            original_query=query,  # For logging/debugging
            expanded_query=understanding.expanded_query,  # For sparse/keyword search
            query_embedding=query_embedding,  # From ORIGINAL query
            top_k=top_k_search,
            filters=understanding.metadata_filters,
            name_embedding=name_embedding,  # Optional name-focused boost
            name_boost=name_boost
        )
        
        # Step 6: BGE reranking using ORIGINAL query for relevance scoring
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
                "is_current_only": understanding.is_current_only,
                "name_boost_applied": name_boost > 0
            }
        }
