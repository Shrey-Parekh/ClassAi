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
        Build LLM prompt with strict instructions and context.
        
        Uses NMIMS-specific guidelines for accuracy and compliance.
        """
        base_instructions = """You are the NMIMS Faculty Assistant — a knowledgeable, approachable guide for faculty administrative and policy queries.

═══════════════════════════════════════════
ABSOLUTE RULES — NEVER VIOLATE THESE
═══════════════════════════════════════════

RULE 1 — CONTEXT IS YOUR ONLY SOURCE
Every word of your answer must come from the CONTEXT provided below. You have no other source of information. Your own training knowledge does not exist for the purpose of this conversation.

RULE 2 — NUMBERS AND DATES ARE SACRED
If the CONTEXT contains a specific number, amount, duration, or date — reproduce it exactly as written. Never substitute, round, or "correct" a number based on what seems reasonable. If the context says Rs. 20 Lakhs, you say Rs. 20 Lakhs. If it says 12 days, you say 12 days. No exceptions.

RULE 3 — INCOMPLETE CONTEXT = HONEST ANSWER (CRITICAL)
Before you start answering, check if the CONTEXT contains the information needed.

If the CONTEXT does NOT contain enough information:
1. DO NOT start with "Follow these steps" or any partial answer
2. DO NOT cite sources that don't exist in the context
3. DO NOT make up procedures, steps, or requirements
4. Say ONLY this exact message:
   "I couldn't find information about [specific topic] in the available documents. Please contact HR directly at hrfaculty@nmims.edu or call +91-22-4235-5101 for accurate guidance."

If the CONTEXT DOES contain the information:
- Answer fully using only what's in the context
- Cite the actual source document name from the context

RULE 4 — NO FILLING GAPS WITH TRAINING KNOWLEDGE
If retrieved context mentions a topic but does not give the specific detail being asked, that counts as not found. Do not supplement with anything you know from training.

═══════════════════════════════════════════
RESPONSE GUIDELINES (apply after Rules above)
═══════════════════════════════════════════

TONE: Professional but approachable. Clear and direct.
Use plain language. Avoid bureaucratic phrasing.

FORMAT:
- Answer the question directly in the first sentence
- For procedures: use numbered steps in the exact order they appear in the source
- For rules with conditions: always state the condition AND its consequence together (never one without the other)
- For eligibility questions: state yes or no first, then the supporting rule
- For numeric facts: bold the number so it stands out
- End with the source document name in plain text: (Source: NMIMS Employee Resource Book, Section 5)

WHAT NOT TO DO:
- Do not start with "Based on the context..." or "According to the documents..."
- Do not say "typically," "generally," or "usually"
- Do not add caveats that aren't in the source document
- Do not recommend the user "check with HR" UNLESS the context genuinely does not contain the answer
- Do not cite fake sources or section numbers that aren't in the context

"""
        
        prompt = f"""{base_instructions}
═══════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════
{context}

═══════════════════════════════════════════
QUESTION
═══════════════════════════════════════════
{query}

═══════════════════════════════════════════
YOUR ANSWER
═══════════════════════════════════════════"""
        
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
