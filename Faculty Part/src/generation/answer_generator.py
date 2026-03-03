"""
LLM-based answer generation grounded in retrieved chunks.
"""

from typing import List, Dict, Any
from config.chunking_config import ContentType


class AnswerGenerator:
    """
    Generate answers using LLM, strictly grounded in retrieved chunks.
    
    Key behaviors:
    - Answer only from retrieved chunks, never assume
    - Present procedures in order with clear steps
    - State conditions before consequences for rules
    - Explicitly say when answer is not found
    - Keep answers concise but complete
    """
    
    def __init__(self, llm_client):
        """Initialize with LLM client (OpenAI, Anthropic, etc.)."""
        self.llm = llm_client
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        intent_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate answer from query and retrieved chunks.
        
        Args:
            query: Original faculty query
            retrieved_chunks: Top-k chunks from retrieval pipeline
            intent_type: Detected intent (optional, for prompt customization)
        
        Returns:
            Dict with answer, sources, and confidence
        """
        # Build context from chunks
        context = self._build_context(retrieved_chunks)
        
        # Build prompt based on intent
        prompt = self._build_prompt(query, context, intent_type)
        
        # Generate answer
        response = self.llm.generate(prompt)
        
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
        
        Organizes chunks by type for better LLM comprehension.
        """
        context_parts = []
        
        # Group chunks by content type
        procedures = []
        rules = []
        facts = []
        
        for chunk in chunks:
            content_type = chunk.get("metadata", {}).get("content_type")
            content = chunk.get("content", "")
            
            if content_type == ContentType.PROCEDURE.value:
                procedures.append(content)
            elif content_type in [ContentType.RULE.value, ContentType.POLICY.value]:
                rules.append(content)
            else:
                facts.append(content)
        
        # Build organized context
        if procedures:
            context_parts.append("PROCEDURES:\n" + "\n\n".join(procedures))
        
        if rules:
            context_parts.append("RULES AND POLICIES:\n" + "\n\n".join(rules))
        
        if facts:
            context_parts.append("RELEVANT INFORMATION:\n" + "\n\n".join(facts))
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        intent_type: str = None
    ) -> str:
        """
        Build LLM prompt with instructions and context.
        
        Customizes instructions based on intent type.
        """
        base_instructions = """You are a helpful and friendly faculty assistant. Your role is to help users find information about faculty members in a natural, conversational way.

IMPORTANT GUIDELINES:
1. Be conversational and warm - avoid robotic or overly formal language
2. Answer based ONLY on the provided context - never make up information
3. If you don't have the information, say so naturally: "I don't have that information available right now"
4. Use natural language - say "Dr. Kumar researches AI" instead of "According to the data, Dr. Kumar's research interests include..."
5. For multiple results, present them in a friendly list format
6. Keep responses concise but complete - users want quick, actionable information
7. When mentioning faculty, include relevant details like their research areas or notable achievements
8. Use contractions and natural phrasing (I'm, you're, they're, etc.)

RESPONSE STYLE:
- Start directly with the answer, no preamble like "Based on the context..."
- Use bullet points for multiple items
- Highlight key information naturally
- End with a helpful note if relevant (e.g., "Would you like to know more about their publications?")

"""
        
        # Add intent-specific instructions
        if intent_type == "procedure":
            base_instructions += "\nThis is a HOW-TO question. Provide clear, friendly step-by-step guidance.\n"
        elif intent_type == "eligibility":
            base_instructions += "\nThis is about ELIGIBILITY. Clearly explain who qualifies in a straightforward way.\n"
        elif intent_type == "lookup":
            base_instructions += "\nThis is a FACTUAL question. Provide the specific information in a natural, conversational tone.\n"
        
        prompt = f"""{base_instructions}

CONTEXT:
{context}

USER QUESTION: {query}

YOUR RESPONSE:"""
        
        return prompt
    
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
