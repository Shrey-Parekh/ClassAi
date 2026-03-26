"""
Complete retrieval pipeline orchestrating all components.
"""

from typing import Dict, Any, List
from config.chunking_config import TOP_K_RERANKED
import logging

from .query_understanding import QueryAnalyzer
from .hybrid_search import HybridSearchEngine
from .bge_reranker import BGEReranker


class RetrievalPipeline:
    """
    Enhanced retrieval pipeline with query understanding and BGE reranking.
    
    Pipeline steps:
    1. Query understanding (intent + domain + entities)
    2. Metadata pre-filtering (domain + is_current)
    3. Query embedding (BAAI/bge-m3)
    4. Hybrid search with intent-based weighting → 20 candidates
    5. BGE reranking → 15 final chunks
    6. Return final chunks
    
    Embedding model: BAAI/bge-m3 (1024 dimensions)
    - Documents: Embedded with BAAI/bge-m3
    - Queries: Embedded with BAAI/bge-m3
    - Consistency: Guaranteed (same model for both)
    """
    
    def __init__(
        self,
        vector_db_client,
        embedding_model,
        collection_name: str = "faculty_chunks",
        llm_client=None
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            vector_db_client: Vector database client (Qdrant)
            embedding_model: Query embedding model (BAAI/bge-m3)
            collection_name: Vector DB collection name
            llm_client: Optional LLM client for HyDE (topic search only)
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
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
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
        
        # Step 1.5: Attempt direct metadata match for name queries
        # This bypasses vector search entirely if we find an exact name match
        if understanding.intent == "lookup" and understanding.entities:
            try:
                direct_results = self._attempt_direct_name_match(understanding.entities[0])
                if direct_results:
                    self.logger.info(f"Direct metadata match found for: {understanding.entities[0]}")
                    return {
                        "chunks": direct_results,
                        "intent": understanding.intent,
                        "domain": understanding.domain,
                        "entities": understanding.entities,
                        "metadata": {
                            "retrieval_path": "direct_metadata_match",
                            "initial_results": len(direct_results),
                            "final_results": len(direct_results),
                            "filters_applied": {},
                            "is_current_only": understanding.is_current_only,
                            "name_boost_applied": False
                        }
                    }
            except Exception as e:
                self.logger.warning(f"Direct metadata match failed, falling back to hybrid search: {e}")
        
        # Step 2: Strip titles from query for embedding
        query_clean = self.query_analyzer._strip_titles_for_embedding(query)
        
        # Step 3: Generate query embedding
        # For TOPIC_SEARCH: Use HyDE (hypothetical document embedding)
        # For LOOKUP: Use dual embedding (original + name-focused)
        # For others: Use standard embedding
        query_embedding = None
        name_embedding = None
        name_boost = 0.0
        
        if understanding.intent == "topic_search" and self.llm_client:
            # HyDE: Generate hypothetical faculty description for topic search
            try:
                query_embedding = self._generate_hyde_embedding(query_clean)
                self.logger.info(f"HyDE embedding generated for topic search: {query_clean}")
            except Exception as e:
                self.logger.warning(f"HyDE generation failed, falling back to standard embedding: {e}")
                query_embedding = self.embedding_model.embed(query_clean)
        else:
            # Standard embedding for all other intents
            query_embedding = self.embedding_model.embed(query_clean)
        
        # Step 4: For faculty name queries, create name-focused embedding
        if understanding.intent == "lookup" and understanding.entities:
            # Extract faculty name (first entity) and strip titles
            faculty_name_clean = self.query_analyzer._strip_titles_for_embedding(understanding.entities[0])
            
            # Create name-focused query for better matching
            # Format: "Faculty: [Name]" to match chunk format
            name_query = f"Faculty: {faculty_name_clean}"
            name_embedding = self.embedding_model.embed(name_query)
            name_boost = 0.3  # 30% boost for name-based matches
        
        # Step 5: Set retrieval parameters
        # top 20 → rerank → keep top 15
        top_k_search = 20
        top_k_rerank = 15
        
        # Step 6: Hybrid search with metadata pre-filtering and intent-based weighting
        # - Dense search uses CLEANED query embedding (titles stripped)
        # - Sparse search uses EXPANDED query (titles stripped, keywords added)
        # - Name search uses NAME-FOCUSED embedding (titles stripped)
        # - Intent determines dense/sparse weight balance
        search_results = self.search_engine.search(
            original_query=query_clean,  # Cleaned query for logging
            expanded_query=understanding.expanded_query,  # Already has titles stripped
            query_embedding=query_embedding,  # From CLEANED query
            top_k=top_k_search,
            filters=understanding.metadata_filters,
            name_embedding=name_embedding,  # Optional name-focused boost
            name_boost=name_boost,
            intent=understanding.intent  # For dynamic weight selection
        )
        
        # Step 7: BGE reranking using CLEANED query for relevance scoring
        reranked_results = self.reranker.rerank(
            query=query_clean,
            results=search_results,
            top_k=top_k_rerank
        )
        
        # Step 8: Check confidence - trigger second pass if needed
        if reranked_results and reranked_results[0].score < 0.4:
            self.logger.info(f"Low confidence ({reranked_results[0].score:.3f}), triggering second pass")
            
            # Extract signal terms and retry
            signal_terms = self._extract_signal_terms(query_clean, understanding.entities)
            if signal_terms:
                self.logger.info(f"Second pass with signal terms: {signal_terms}")
                
                # Re-run hybrid search with signal terms only
                signal_query = " ".join(signal_terms)
                signal_embedding = self.embedding_model.embed(signal_query)
                
                second_search_results = self.search_engine.search(
                    original_query=signal_query,
                    expanded_query=signal_query,
                    query_embedding=signal_embedding,
                    top_k=top_k_search,
                    filters=understanding.metadata_filters,
                    intent=understanding.intent
                )
                
                second_reranked = self.reranker.rerank(
                    query=signal_query,
                    results=second_search_results,
                    top_k=top_k_rerank
                )
                
                # Use second pass if better
                if second_reranked and second_reranked[0].score > reranked_results[0].score:
                    self.logger.info(f"Second pass improved score: {second_reranked[0].score:.3f}")
                    reranked_results = second_reranked
                
                # If still low confidence, return no-confidence response
                if reranked_results[0].score < 0.4:
                    self.logger.info("Both passes failed, returning no-confidence response")
                    return self._generate_no_confidence_response(query, understanding.intent)
        
        # Format results
        chunks = [
            {
                "chunk_id": result.chunk_id,
                "content": result.content,
                "score": result.score,
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
                "retrieval_path": "hybrid_search",
                "initial_results": len(search_results),
                "final_results": len(chunks),
                "filters_applied": understanding.metadata_filters,
                "is_current_only": understanding.is_current_only,
                "name_boost_applied": name_boost > 0
            }
        }
    
    def _generate_hyde_embedding(self, query: str) -> List[float]:
        """
        Generate HyDE (Hypothetical Document Embedding) for topic search.
        
        Creates a hypothetical faculty description matching the topic,
        then embeds that description instead of the raw query.
        
        This improves topic-based faculty search by matching against
        how faculty profiles are actually written.
        
        Args:
            query: Clean query (titles already stripped)
        
        Returns:
            Embedding vector for the hypothetical document
        """
        hyde_prompt = f"""Write one sentence describing a faculty member who works on this topic: {query}

Format exactly like this example:
"Dr. X is a faculty member in the Computer Science department specializing in machine learning and neural networks with publications in deep learning."

Write only the sentence, nothing else."""
        
        # Generate hypothetical description
        hypothetical = self.llm_client.generate(
            hyde_prompt,
            temperature=0.5,
            max_tokens=80
        )
        
        # Embed the hypothetical description
        return self.embedding_model.embed(hypothetical.strip())
    
    def _attempt_direct_name_match(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Attempt direct metadata filter match for faculty name queries.
        
        This bypasses vector search entirely if we find exact name matches
        in the name_variants field.
        
        Args:
            entity_name: Extracted entity (faculty name with possible titles)
        
        Returns:
            List of matching chunks, or empty list if no match
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        # Strip titles and normalize
        name_clean = self.query_analyzer._strip_titles_for_embedding(entity_name)
        name_parts = name_clean.split()
        
        if not name_parts:
            return []
        
        try:
            # Attempt scroll with name_variants filter
            results, _ = self.vector_db.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="name_variants",
                            match=MatchAny(any=name_parts)
                        )
                    ]
                ),
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            if results:
                # Format results to match standard chunk format
                chunks = [
                    {
                        "chunk_id": point.id,
                        "content": point.payload.get("content", ""),
                        "score": 1.0,  # Direct match gets perfect score
                        "metadata": point.payload,
                    }
                    for point in results
                ]
                return chunks
            
        except Exception as e:
            # name_variants field may not exist yet - log and return empty
            self.logger.debug(f"name_variants field not available: {e}")
            return []
        
        return []

    
    def _extract_signal_terms(self, query: str, entities: List[str]) -> List[str]:
        """
        Extract 2-3 highest-signal nouns/entities from query.
        
        Used for second-pass retrieval when first pass has low confidence.
        """
        signal_terms = []
        
        if entities:
            signal_terms.extend(entities[:2])
        
        # Extract nouns (capitalized or long words)
        words = query.split()
        potential_nouns = [
            word for word in words
            if (len(word) > 4 and word[0].isupper()) or len(word) > 6
        ]
        
        for noun in potential_nouns:
            if noun.lower() not in [t.lower() for t in signal_terms]:
                signal_terms.append(noun)
                if len(signal_terms) >= 3:
                    break
        
        return signal_terms[:3]
    
    def _generate_no_confidence_response(self, query: str, intent: str) -> Dict[str, Any]:
        """Generate no-confidence response with reformulation suggestions."""
        suggestions = []
        
        if intent == "lookup":
            suggestions = [
                "Try using the full name (e.g., 'Dr. John Smith')",
                "Include the department name",
                "Check the spelling of the name"
            ]
        elif intent == "procedure":
            suggestions = [
                "Be more specific about which procedure",
                "Include the document or policy name",
                "Try using keywords like 'application', 'form', or 'process'"
            ]
        elif intent == "eligibility":
            suggestions = [
                "Specify what you want to be eligible for",
                "Include relevant details (years of service, role, etc.)",
                "Try rephrasing as 'What are the requirements for...'"
            ]
        else:
            suggestions = [
                "Try being more specific",
                "Include relevant keywords or document names",
                "Rephrase your question differently"
            ]
        
        return {
            "chunks": [],
            "intent": intent,
            "domain": "general",
            "entities": [],
            "metadata": {
                "retrieval_path": "no_confidence",
                "initial_results": 0,
                "final_results": 0,
                "filters_applied": {},
                "is_current_only": False,
                "name_boost_applied": False,
                "reformulation_suggestions": suggestions
            }
        }
