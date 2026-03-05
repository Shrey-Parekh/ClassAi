"""
LLM-based answer generation grounded in retrieved chunks.
"""

from typing import List, Dict, Any
from config.chunking_config import ContentType
from .prompt_templates import get_prompt


class AnswerGenerator:
    """
    Generate answers using LLM with intent-based prompt templates.
    
    Key behaviors:
    - Uses intent-specific templates for structured output
    - Answer only from retrieved chunks, never assume
    - Temperature set to 0.2 for consistent, factual responses
    """
    
    def __init__(self, llm_client):
        """Initialize with LLM client (Ollama)."""
        self.llm = llm_client
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        intent_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate answer from query and retrieved chunks using smart system prompt.
        
        Args:
            query: Original faculty query
            retrieved_chunks: Top-k chunks from retrieval pipeline
            intent_type: Detected intent (for metadata only, not used in prompt)
        
        Returns:
            Dict with answer, sources, and confidence
        """
        # Build context from chunks
        context = self._build_context(retrieved_chunks)
        
        # Get prompt with context and query
        prompt = get_prompt(context, query)
        
        # Generate answer with temperature=0.2 for factual consistency
        response = self.llm.generate(prompt, temperature=0.2)
        
        # Extract sources
        sources = self._extract_sources(retrieved_chunks)
        
        return {
            "answer": response,
            "sources": sources,
            "chunks_used": len(retrieved_chunks),
            "intent": intent_type,
        }
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Organizes chunks by type and preserves document boundaries
        for better cross-document reasoning.
        """
        context_parts = []
        
        # Group chunks by content type
        procedures = []
        rules = []
        facts = []
        
        for chunk in chunks:
            content_type = chunk.get("metadata", {}).get("content_type")
            content = chunk.get("content", "")
            doc_title = chunk.get("metadata", {}).get("title", "Unknown Document")
            
            # Wrap content with document source for cross-document clarity
            wrapped_content = f"[Source: {doc_title}]\n{content}"
            
            if content_type == ContentType.PROCEDURE.value:
                procedures.append(wrapped_content)
            elif content_type in [ContentType.RULE.value, ContentType.POLICY.value]:
                rules.append(wrapped_content)
            else:
                facts.append(wrapped_content)
        
        # Build organized context
        if procedures:
            context_parts.append("PROCEDURES:\n" + "\n\n".join(procedures))
        
        if rules:
            context_parts.append("RULES AND POLICIES:\n" + "\n\n".join(rules))
        
        if facts:
            context_parts.append("RELEVANT INFORMATION:\n" + "\n\n".join(facts))
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source document information from chunks."""
        sources = []
        seen_docs = set()
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("doc_id")
            
            if doc_id and doc_id not in seen_docs:
                sources.append({
                    "doc_id": doc_id,
                    "title": metadata.get("title", "Unknown Document"),
                    "date": metadata.get("date", ""),
                    "applies_to": metadata.get("applies_to", ""),
                })
                seen_docs.add(doc_id)
        
        return sources
