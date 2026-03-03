"""
Complete retrieval pipeline orchestrating all components.
"""

from typing import Dict, Any, List
from config.chunking_config import TOP_K_RERANKED

from .intent_classifier import IntentClassifier
from .hybrid_search import HybridSearchEngine


class RetrievalPipeline:
    """
    Complete retrieval pipeline implementing the intent-based strategy.
    
    Pipeline steps:
    1. Detect query intent
    2. Apply metadata filters
    3. Run hybrid search (vector + BM25)
    4. Return top-k chunks
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
            vector_db_client: Vector database client (Qdrant, etc.)
            embedding_model: Model for generating query embeddings
            collection_name: Vector DB collection name
        """
        self.intent_classifier = IntentClassifier()
        self.search_engine = HybridSearchEngine(
            vector_db_client=vector_db_client,
            collection_name=collection_name
        )
        self.embedding_model = embedding_model
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RERANKED
    ) -> Dict[str, Any]:
        """
        Execute complete retrieval pipeline.
        
        Args:
            query: Faculty query text
            top_k: Number of final results to return
        
        Returns:
            Dict containing:
            - chunks: List of retrieved chunks
            - intent: Detected intent type
            - metadata: Pipeline metadata
        """
        # Step 1: Detect intent
        intent = self.intent_classifier.classify(query)
        
        # Step 2: Get metadata filters
        filters = self.intent_classifier.get_metadata_filters(query, intent)
        
        # Add chunk level preference based on intent (only for document-based chunks)
        # Skip for CSV/Excel data which doesn't have chunk levels
        # target_levels = self.intent_classifier.get_target_levels(intent)
        # filters["chunk_level"] = [level.value for level in target_levels]
        
        # Step 3: Generate query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Step 4: Hybrid search
        search_results = self.search_engine.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,  # Get final count directly
            filters=filters
        )
        
        # Format results
        chunks = [
            {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata,
            }
            for result in search_results
        ]
        
        return {
            "chunks": chunks,
            "intent": intent.value,
            "metadata": {
                "final_results": len(chunks),
                "filters_applied": filters,
            }
        }
