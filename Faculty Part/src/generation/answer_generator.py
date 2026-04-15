"""
LLM-based answer generation with structured JSON output.
"""

from typing import List, Dict, Any
import json
import os
import re
import logging
from pydantic import ValidationError

from .prompt_templates import get_prompt
from .response_schema import StructuredResponse
from config.chunking_config import INTENT_CHUNK_LIMITS, DEFAULT_CHUNK_LIMIT


def _context_budget_for_provider() -> Dict[str, int]:
    """
    Pick context/output token budget based on the active LLM provider.
    Ollama's gemma3:12b runs with num_ctx=8192 in llm.py, so Gemini's 32K
    budget would silently truncate prompts. Return (context, output) pairs
    sized for the backend actually in use.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "gemini":
        return {"context": 32768, "output": 2048, "prompt": 500}
    # Ollama / default: match num_ctx=8192 in utils/llm.py
    return {"context": 7168, "output": 1024, "prompt": 500}


# Schema normalization mappings
# Accept LLM's creative names and map them to valid schema types
TYPE_ALIASES = {
    # LLM's invented names → valid types
    "list": "bullets",
    "unordered": "bullets",
    "ordered": "steps",
    "numbered": "steps",
    "text": "paragraph",
    "prose": "paragraph",
    "info": "alert",
    "warning": "alert",
    "note": "alert",
    "callout": "alert",
    "bullet_points": "bullets",
    "bullet": "bullets",
    "numbered_list": "steps",
    "step_by_step": "steps",
    "body": "paragraph"
}

SECTION_FIELD_FIXES = {
    # If LLM uses wrong field names → fix them
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
    - Dynamic token-based chunk limiting for optimal context usage
    """
    
    # Context window allocation — branches on LLM_PROVIDER env var so the
    # same code works for both Ollama (8K ctx) and Gemini (32K+ ctx).
    _BUDGET = _context_budget_for_provider()
    MAX_CONTEXT_TOKENS = _BUDGET["context"]
    SYSTEM_PROMPT_TOKENS = _BUDGET["prompt"]
    OUTPUT_TOKENS = _BUDGET["output"]
    AVAILABLE_FOR_CHUNKS = MAX_CONTEXT_TOKENS - SYSTEM_PROMPT_TOKENS - OUTPUT_TOKENS
    
    def __init__(self, llm_client):
        """Initialize with LLM client (Gemini 2.0 Flash)."""
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (word count * 1.3 for subword inflation)."""
        return int(len(text.split()) * 1.3) if text else 0
    
    def _select_chunks_by_tokens(
        self,
        chunks: List[Dict[str, Any]],
        intent_type: str
    ) -> List[Dict[str, Any]]:
        """
        Select chunks dynamically based on token budget and relevance.
        
        Strategy:
        1. Sort by relevance score (already done by reranker)
        2. Add chunks until token budget exhausted
        3. Prioritize high-relevance chunks
        4. Stop when budget reached or quality drops
        
        Args:
            chunks: Retrieved chunks sorted by relevance
            intent_type: Query intent
        
        Returns:
            Selected chunks that fit in context window
        """
        selected = []
        total_tokens = 0
        
        # Get intent-specific max chunks as upper bound
        max_chunks = INTENT_CHUNK_LIMITS.get(intent_type.lower(), DEFAULT_CHUNK_LIMIT)
        
        for i, chunk in enumerate(chunks):
            # Stop if we've hit max chunks
            if i >= max_chunks:
                break
            
            # Estimate tokens for this chunk
            chunk_text = chunk.get("text", "")
            chunk_tokens = self._estimate_tokens(chunk_text)
            
            # Check if adding this chunk would exceed budget
            if total_tokens + chunk_tokens > self.AVAILABLE_FOR_CHUNKS:
                self.logger.info(f"Token budget exhausted at chunk {i+1}/{len(chunks)}")
                break
            
            # Quality threshold: stop if relevance drops too low
            # Reranker scores are typically 0-1, stop if < 0.3
            score = chunk.get("score", 1.0)
            if i > 5 and score < 0.3:  # After first 5, enforce quality threshold
                self.logger.info(f"Quality threshold not met at chunk {i+1} (score: {score:.3f})")
                break
            
            selected.append(chunk)
            total_tokens += chunk_tokens
        
        self.logger.info(
            f"Selected {len(selected)}/{len(chunks)} chunks, "
            f"~{total_tokens} tokens (~{(total_tokens/self.AVAILABLE_FOR_CHUNKS)*100:.1f}% of budget)"
        )
        
        return selected
    
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
            retrieved_chunks: Top-k chunks from retrieval pipeline (sorted by relevance)
            intent_type: Detected intent
        
        Returns:
            Dict with structured response, sources, and metadata
        """
        # Select chunks dynamically based on token budget and relevance
        chunks_to_use = self._select_chunks_by_tokens(retrieved_chunks, intent_type)
        
        self.logger.debug(f"=== CHUNK RETRIEVAL DIAGNOSTIC ===")
        self.logger.debug(f"Intent: {intent_type}")
        self.logger.debug(f"Chunks available: {len(retrieved_chunks)}")
        self.logger.debug(f"Chunks selected: {len(chunks_to_use)}")
        if chunks_to_use:
            self.logger.debug(f"Score range: {chunks_to_use[0].get('score', 0):.3f} - {chunks_to_use[-1].get('score', 0):.3f}")
            for i, chunk in enumerate(chunks_to_use[:3], 1):
                self.logger.debug(f"\nCHUNK {i}:")
                self.logger.debug(f"  Available keys: {list(chunk.keys())}")
                self.logger.debug(f"  Score: {chunk.get('score', 'N/A')}")
                self.logger.debug(f"  Source: {chunk.get('metadata', {}).get('document_name', 'unknown')}")
                self.logger.debug(f"  Section: {chunk.get('metadata', {}).get('section_title', 'N/A')}")
                
                # Try multiple possible content keys
                content = (
                    chunk.get("text") or 
                    chunk.get("content") or 
                    chunk.get("chunk_text") or
                    chunk.get("payload", {}).get("text") or
                    chunk.get("payload", {}).get("content") or
                    "*** EMPTY - KEY MISMATCH ***"
                )
                self.logger.debug(f"  Content preview: {str(content)[:200]}")
        else:
            self.logger.debug("WARNING: No chunks selected!")
        self.logger.debug("=== END CHUNK DIAGNOSTIC ===")
        
        # Build context from selected chunks
        context = self._build_context(chunks_to_use)
        
        self.logger.debug(f"=== CONTEXT DIAGNOSTIC ===")
        self.logger.debug(f"Context length: {len(context)} chars")
        self.logger.debug(f"Context preview (first 500 chars):\n{context[:500]}")
        self.logger.debug("=== END CONTEXT ===")
        
        # Get JSON prompt for intent
        prompt = get_prompt(intent_type, context, query)
        
        self.logger.debug(f"=== PROMPT DIAGNOSTIC ===")
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        self.logger.debug(f"Prompt preview (first 800 chars):\n{prompt[:800]}")
        self.logger.debug("=== END PROMPT ===")
        
        # DEBUG: Count tokens being sent to Gemini
        prompt_chars = len(prompt)
        estimated_input_tokens = self._estimate_tokens(prompt)
        context_chars = len(context)
        estimated_context_tokens = self._estimate_tokens(context)
        
        self.logger.debug("=== TOKEN USAGE ESTIMATE ===")
        self.logger.debug(f"Prompt total characters: {prompt_chars}")
        self.logger.debug(f"Estimated INPUT tokens: {estimated_input_tokens}")
        self.logger.debug(f"Context characters: {context_chars}")
        self.logger.debug(f"Estimated CONTEXT tokens: {estimated_context_tokens}")
        self.logger.debug(f"Number of chunks: {len(chunks_to_use)}")
        self.logger.debug(f"Gemini 2.0 Flash context window: 1M tokens (using {self.MAX_CONTEXT_TOKENS} for this request)")
        self.logger.debug(f"Usage: {(estimated_input_tokens/self.MAX_CONTEXT_TOKENS)*100:.1f}% of allocated context")
        self.logger.debug("=== END TOKEN ESTIMATE ===")

        # Generate with Gemini 2.0 Flash
        # TODO: Update format parameter for Gemini client wrapper (may need response_mime_type="application/json")
        raw_response = self.llm.generate(
            prompt,
            temperature=0.2,
            max_tokens=2048,  # Increased for complex JSON responses with citations
            format="json"  # TODO: Verify this parameter works with Gemini client wrapper
        )
        
        # DIAGNOSTIC: Log what Gemini actually returns
        self.logger.debug(f"=== RAW LLM RESPONSE ===")
        self.logger.debug(f"Length: {len(raw_response)} chars")
        self.logger.debug(f"First 500 chars: {raw_response[:500]}")
        self.logger.debug(f"Last 100 chars: {raw_response[-100:]}")
        self.logger.debug("=== END RAW RESPONSE ===")

        # Parse and validate JSON
        structured = self._parse_json_response(raw_response, intent_type, query)
        
        # Auto-generate footer from source documents if not provided by LLM
        if not structured.footer:
            source_docs = set()
            for chunk in chunks_to_use:
                doc_name = chunk.get("metadata", {}).get("document_name", "")
                if doc_name:
                    clean_name = doc_name.replace(".pdf", "").replace(".json", "").replace("_", " ")
                    source_docs.add(clean_name)
            
            if source_docs:
                structured.footer = "Based on: " + ", ".join(sorted(source_docs))

        # Extract sources
        sources = self._extract_sources(chunks_to_use)

        # Calculate confidence
        confidence = self._calculate_confidence(chunks_to_use, structured)
        
        # Log context usage
        self._log_context_usage(query, chunks_to_use, estimated_context_tokens)
        
        # Use model_dump() for Pydantic v2 compatibility
        try:
            structured_dict = structured.model_dump()
        except AttributeError:
            # Fallback for Pydantic v1
            structured_dict = structured.dict()
        
        return {
            "structured": structured_dict,
            "sources": sources,
            "chunks_used": len(chunks_to_use),
            "intent": intent_type,
            "confidence": confidence,
            "tokens_used": estimated_context_tokens
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
        Normalize LLM's creative JSON to match our schema.
        
        Fixes:
        - Type aliases (list → bullets, text → paragraph)
        - Field name mismatches (points → items, body → content)
        - Missing required fields (adds defaults)
        
        Args:
            data: Raw parsed JSON from LLM
        
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
        
        Each chunk includes source information for reference.
        """
        context_parts = []
        
        for chunk in chunks:
            # Text is at top level, not in metadata
            content = chunk.get("text", "")
            doc_name = chunk.get("metadata", {}).get("document_name", "Unknown Document")
            section = chunk.get("metadata", {}).get("section_title", "")
            
            # Build source header
            source_header = f"Source: {doc_name}"
            if section:
                source_header += f" — {section}"
            
            # Format without numbering
            formatted = f"{source_header}\n{content}"
            context_parts.append(formatted)
        
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

    
    def _calculate_confidence(
        self,
        chunks: List[Dict[str, Any]],
        structured: Any
    ) -> str:
        """
        Calculate confidence level based on chunk scores and answer quality.
        
        Args:
            chunks: Retrieved chunks with scores
            structured: Parsed structured response
        
        Returns:
            Confidence level: "high", "medium", "low", or "none"
        """
        if not chunks:
            return "none"
        
        # Get top chunk score
        top_score = chunks[0].get("score", 0.0)
        
        # Get number of chunks used
        num_chunks = len(chunks)
        
        # Check if answer has content
        has_content = False
        if hasattr(structured, 'sections') and structured.sections:
            has_content = True
        elif hasattr(structured, 'fallback') and structured.fallback:
            return "none"
        
        # Calculate confidence based on multiple factors
        if top_score >= 0.7 and num_chunks >= 3 and has_content:
            return "high"
        elif top_score >= 0.4 and num_chunks >= 2 and has_content:
            return "medium"
        elif top_score >= 0.3 and has_content:
            return "low"
        else:
            return "none"

    
    def _log_context_usage(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        tokens_used: int
    ) -> None:
        """
        Log context usage to file for analysis.
        
        Args:
            query: Original query
            chunks: Chunks used
            tokens_used: Estimated tokens used
        """
        import json
        from datetime import datetime
        from pathlib import Path
        
        try:
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Append to log file
            log_file = log_dir / "context_usage.jsonl"

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "chunks_used": len(chunks),
                "tokens_used": tokens_used,
                "max_context_tokens": self.MAX_CONTEXT_TOKENS,
                "available_for_chunks": self.AVAILABLE_FOR_CHUNKS,
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to log context usage: {e}")
