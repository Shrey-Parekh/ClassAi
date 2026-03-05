"""
LLM-based answer generation with structured JSON output.
"""

from typing import List, Dict, Any
import json
import re
import logging
from pydantic import ValidationError

from .prompt_templates import get_prompt
from .response_schema import StructuredResponse
from config.chunking_config import INTENT_CHUNK_LIMITS, DEFAULT_CHUNK_LIMIT


# Schema normalization mappings
# Accept Gemma's creative names and map them to valid schema types
TYPE_ALIASES = {
    # Gemma's invented names → valid types
    "list": "bullets",
    "unordered": "bullets",
    "ordered": "steps",
    "numbered": "steps",
    "text": "paragraph",
    "prose": "paragraph",
    "info": "alert",
    "warning": "alert",
    "note": "alert",
    "callout": "alert"
}

SECTION_FIELD_FIXES = {
    # If Gemma uses wrong field names → fix them
    "content": "items",  # for bullet/steps sections
    "points": "items",
    "list": "items",
    "body": "content",  # for paragraph sections
    "text": "content",
    "description": "content"
}


class AnswerGenerator:
    """
    Generate structured JSON answers using LLM.
    
    Key behaviors:
    - Forces LLM to return strict JSON
    - Validates JSON against Pydantic schema
    - Returns clean fallback on parse failures
    - Temperature set to 0.1 for JSON consistency
    - Intent-based chunk limiting for optimal context usage
    """
    
    def __init__(self, llm_client):
        """Initialize with LLM client (Ollama)."""
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        intent_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate structured JSON answer from query and retrieved chunks.
        
        Args:
            query: Original faculty query
            retrieved_chunks: Top-k chunks from retrieval pipeline
            intent_type: Detected intent
        
        Returns:
            Dict with structured response, sources, and metadata
        """
        # Determine chunk limit based on intent
        chunk_limit = INTENT_CHUNK_LIMITS.get(intent_type.lower(), DEFAULT_CHUNK_LIMIT)
        
        # Limit chunks to intent-specific amount
        chunks_to_use = retrieved_chunks[:chunk_limit]
        
        print(f"=== CHUNK LIMITING ===")
        print(f"Intent: {intent_type}")
        print(f"Chunks available: {len(retrieved_chunks)}")
        print(f"Chunks using: {len(chunks_to_use)} (limit: {chunk_limit})")
        print("=== END LIMITING ===")
        
        # Build context from chunks
        context = self._build_context(chunks_to_use)
        
        # Get JSON prompt for intent
        prompt = get_prompt(intent_type, context, query)
        
        # DEBUG: Count tokens being sent to Gemma
        prompt_chars = len(prompt)
        estimated_input_tokens = prompt_chars // 4
        context_chars = len(context)
        estimated_context_tokens = context_chars // 4
        
        print("=== TOKEN USAGE ESTIMATE ===")
        print(f"Prompt total characters: {prompt_chars}")
        print(f"Estimated INPUT tokens: {estimated_input_tokens}")
        print(f"Context characters: {context_chars}")
        print(f"Estimated CONTEXT tokens: {estimated_context_tokens}")
        print(f"Number of chunks: {len(chunks_to_use)}")
        print(f"Gemma3:12b context window: 32768 tokens (32K)")
        print(f"Usage: {(estimated_input_tokens/32768)*100:.1f}% of context window")
        print("=== END TOKEN ESTIMATE ===")
        
        # Generate with very low temperature for JSON consistency
        # Increased max_tokens to 4096 for detailed responses
        # format="json" forces Ollama to constrain output to valid JSON at token sampling level
        raw_response = self.llm.generate(prompt, temperature=0.1, max_tokens=4096, format="json")
        
        # Parse and validate JSON
        structured = self._parse_json_response(raw_response, intent_type, query)
        
        # Extract sources
        sources = self._extract_sources(chunks_to_use)
        
        return {
            "structured": structured.dict(),
            "sources": sources,
            "chunks_used": len(chunks_to_use),
            "intent": intent_type,
        }
    
    def _parse_json_response(
        self,
        raw_text: str,
        intent: str,
        query: str
    ) -> StructuredResponse:
        """
        Parse and validate LLM JSON response.
        
        Args:
            raw_text: Raw LLM output
            intent: Query intent
            query: Original query
        
        Returns:
            Validated StructuredResponse or fallback
        """
        try:
            # Clean common LLM JSON mistakes
            clean = raw_text.strip()
            
            # Remove markdown code blocks if LLM added them
            if clean.startswith("```"):
                clean = re.sub(r'^```(?:json)?\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean)
            
            # Remove any text before first {
            if '{' in clean:
                clean = clean[clean.index('{'):]
            
            # Remove any text after last }
            if '}' in clean:
                clean = clean[:clean.rindex('}') + 1]
            
            # Parse JSON
            data = json.loads(clean)
            
            # Normalize schema before validation
            data = self._normalize_schema(data)
            
            # Validate against schema
            return StructuredResponse(**data)
        
        except (json.JSONDecodeError, ValidationError) as e:
            # Log the failure for debugging
            self.logger.error(f"JSON parse failed for intent '{intent}': {e}")
            self.logger.error(f"Raw response was: {raw_text[:500]}")
            
            # Return clean fallback response
            return StructuredResponse(
                intent=intent,
                title="Response Error",
                subtitle=None,
                sections=[],
                footer=None,
                confidence="none",
                fallback=(
                    "I was unable to format a response properly. "
                    "Please rephrase your question or contact HR directly."
                )
            )
    
    def _normalize_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Gemma's creative JSON to match our schema.
        
        Fixes:
        - Type aliases (list → bullets, text → paragraph)
        - Field name mismatches (points → items, body → content)
        - Missing required fields (adds defaults)
        
        Args:
            data: Raw parsed JSON from Gemma
        
        Returns:
            Normalized JSON matching StructuredResponse schema
        """
        # Normalize sections
        if "sections" in data and isinstance(data["sections"], list):
            normalized_sections = []
            
            for section in data["sections"]:
                if not isinstance(section, dict):
                    continue
                
                # Normalize type field
                if "type" in section:
                    section_type = section["type"].lower()
                    section["type"] = TYPE_ALIASES.get(section_type, section_type)
                
                # Fix field names based on section type
                section_type = section.get("type", "paragraph")
                
                if section_type in ["bullets", "steps"]:
                    # These need "items" field
                    for wrong_field, correct_field in SECTION_FIELD_FIXES.items():
                        if correct_field == "items" and wrong_field in section:
                            section["items"] = section.pop(wrong_field)
                    
                    # Ensure items is a list
                    if "items" in section and not isinstance(section["items"], list):
                        section["items"] = [str(section["items"])]
                
                elif section_type == "paragraph":
                    # These need "content" field
                    for wrong_field, correct_field in SECTION_FIELD_FIXES.items():
                        if correct_field == "content" and wrong_field in section:
                            section["content"] = section.pop(wrong_field)
                
                elif section_type == "alert":
                    # Ensure severity exists
                    if "severity" not in section:
                        # Infer from heading or default to info
                        heading = section.get("heading", "").lower()
                        if "warning" in heading or "caution" in heading:
                            section["severity"] = "warning"
                        elif "important" in heading or "critical" in heading:
                            section["severity"] = "important"
                        else:
                            section["severity"] = "info"
                
                normalized_sections.append(section)
            
            data["sections"] = normalized_sections
        
        return data
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Organizes chunks with document boundaries for clarity.
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Text is stored in metadata.text, not content
            content = chunk.get("metadata", {}).get("text", "")
            doc_title = chunk.get("metadata", {}).get("title", "Unknown Document")
            
            # Wrap content with source for clarity
            wrapped = f"[Source {i}: {doc_title}]\n{content}"
            context_parts.append(wrapped)
        
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
