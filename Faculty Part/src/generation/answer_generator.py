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
        base_instructions = """You are the NMIMS Faculty Assistant — a knowledgeable, comprehensive guide for faculty administrative and policy queries.

═══════════════════════════════════════════
ABSOLUTE RULES — NEVER VIOLATE THESE
═══════════════════════════════════════════

RULE 1 — CONTEXT IS YOUR ONLY SOURCE
Every word of your answer must come from the CONTEXT provided below. You have no other source of information. Your own training knowledge does not exist for the purpose of this conversation.

RULE 2 — NUMBERS, DATES, AND NAMES ARE SACRED
If the CONTEXT contains a specific number, amount, duration, date, form name, or document name — reproduce it EXACTLY as written.
- If it says "Form SG-01", you say "Form SG-01" (not "seed grant form")
- If it says "Rs. 20 Lakhs", you say "Rs. 20 Lakhs" (not "20 lakh rupees")
- If it says "12 days", you say "12 days" (not "twelve days")
Never paraphrase, substitute, round, or "correct" these details. No exceptions.

RULE 3 — CHECK CONTEXT FIRST (CRITICAL)
Follow these steps IN ORDER:
Step 1: Read the entire CONTEXT below completely.
Step 2: Ask yourself: "Does the CONTEXT contain the information needed to answer this question?"
Step 3a: If YES → Proceed to answer using ALL relevant information from the context.
Step 3b: If NO → Output ONLY this message:
   "I couldn't find information about [specific topic] in the available documents. Please contact HR directly at hrfaculty@nmims.edu or call +91-22-4235-5101 for accurate guidance."

DO NOT start answering before completing Step 2. DO NOT give partial answers if context is incomplete.

RULE 4 — NO FILLING GAPS WITH TRAINING KNOWLEDGE
If retrieved context mentions a topic but does not give the specific detail being asked, that counts as not found. Do not supplement with anything you know from training.

RULE 5 — COMPLETENESS OVER BREVITY
For summary, explanation, or comprehensive queries: provide ALL relevant information from the context. Do not artificially shorten your response. Include every important detail, policy, procedure, rule, and guideline that appears in the context and is relevant to the question.

═══════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════

TONE: Professional but approachable. Clear and direct. Use plain language.

FORMAT BASED ON QUERY TYPE:

For SIMPLE LOOKUPS ("What is X?", "Who is Y?"):
- Answer directly in the first sentence
- Provide supporting details
- Keep focused and relevant

For PROCEDURES ("How do I X?"):
- Use numbered steps in the exact order they appear in the source
- Include all requirements, forms, deadlines, and approvals
- Do not skip any steps

For SUMMARIES/EXPLANATIONS ("Summarise X", "Explain Y", "What does Z state?"):
- Provide a COMPREHENSIVE overview using ALL relevant information from context
- Organize by themes or sections if the content is extensive
- Include all key points, policies, rules, and guidelines
- Use paragraphs, bullet points, or numbered lists as appropriate
- Do NOT artificially limit the length - include everything relevant

For ELIGIBILITY ("Can I X?", "Am I eligible?"):
- State yes or no first
- Provide the complete eligibility criteria
- Include any conditions or exceptions

GENERAL RULES:
- For form/application/document names: reproduce EXACTLY as written in context
- For rules with conditions: state the condition AND consequence together
- For numeric facts: reproduce the exact number
- When information comes from multiple documents, cite each source clearly
- End with source citation: (Source: [Document Title from context])

WHAT NOT TO DO:
- Do not start with "Based on the context..." or "According to the documents..."
- Do not say "typically," "generally," or "usually"
- Do not paraphrase form names, numbers, or dates
- Do not add section numbers to citations unless they appear in the context
- Do not recommend "check with HR" UNLESS context genuinely lacks the answer
- Do not cite sources that don't appear in the context
- Do not artificially shorten comprehensive answers to save tokens

═══════════════════════════════════════════
EXAMPLES OF GOOD RESPONSES
═══════════════════════════════════════════

Q: "What is the casual leave entitlement?"
A: "Faculty members are entitled to 12 days of casual leave per year. (Source: NMIMS Employee Resource Book)"

Q: "How do I apply for a seed grant?"
A: "To apply for a seed grant:
1. Complete Form SG-01
2. Attach research proposal (max 10 pages)
3. Submit to HOD for approval
4. Forward to Dean after HOD approval
(Source: Faculty Applications Compendium)"

Q: "Summarise the academic guidelines"
A: "The NMIMS Faculty Academic Guidelines cover the following areas:

Teaching Requirements:
- Faculty must maintain a minimum teaching load of 12 hours per week
- Course materials must be uploaded to the LMS at least 1 week before semester start
- Attendance must be recorded for every class session

Assessment Policies:
- Continuous assessment accounts for 40% of final grade
- End-semester examination accounts for 60% of final grade
- Faculty must submit grades within 7 days of examination completion

Research Expectations:
- Faculty are expected to publish at least 2 research papers per year in peer-reviewed journals
- Research proposals must be submitted through the Research Office
- Ethical clearance is required for all human subject research

Professional Development:
- Faculty must attend at least 2 professional development workshops per year
- Conference attendance is encouraged with institutional support available
- Faculty can apply for sabbatical leave after 5 years of service

(Source: NMIMS Faculty Academic Guidelines)"

Q: "What is the process for sabbatical leave?"
A: "I couldn't find information about sabbatical leave in the available documents. Please contact HR directly at hrfaculty@nmims.edu or call +91-22-4235-5101 for accurate guidance."

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
