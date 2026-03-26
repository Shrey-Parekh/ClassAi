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
    - Dynamic token-based chunk limiting for optimal context usage
    """
    
    # Context window allocation (8K total for faster processing)
    MAX_CONTEXT_TOKENS = 8192
    SYSTEM_PROMPT_TOKENS = 500  # Prompt template overhead
    OUTPUT_TOKENS = 1024  # Reserve for LLM response (reduced for speed)
    AVAILABLE_FOR_CHUNKS = MAX_CONTEXT_TOKENS - SYSTEM_PROMPT_TOKENS - OUTPUT_TOKENS  # ~6.6K
    
    def __init__(self, llm_client):
        """Initialize with LLM client (Ollama)."""
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 4 chars for English)."""
        return len(text) // 4
    
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
            chunk_text = chunk.get("content", "")
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
        
        print(f"=== CHUNK RETRIEVAL DIAGNOSTIC ===")
        print(f"Intent: {intent_type}")
        print(f"Chunks available: {len(retrieved_chunks)}")
        print(f"Chunks selected: {len(chunks_to_use)}")
        if chunks_to_use:
            print(f"Score range: {chunks_to_use[0].get('score', 0):.3f} - {chunks_to_use[-1].get('score', 0):.3f}")
            for i, chunk in enumerate(chunks_to_use[:3], 1):
                print(f"\nCHUNK {i}:")
                print(f"  Available keys: {list(chunk.keys())}")
                print(f"  Score: {chunk.get('score', 'N/A')}")
                print(f"  Source: {chunk.get('metadata', {}).get('document_name', 'unknown')}")
                print(f"  Section: {chunk.get('metadata', {}).get('section_title', 'N/A')}")
                
                # Try multiple possible content keys
                content = (
                    chunk.get("text") or 
                    chunk.get("content") or 
                    chunk.get("chunk_text") or
                    chunk.get("payload", {}).get("text") or
                    chunk.get("payload", {}).get("content") or
                    "*** EMPTY - KEY MISMATCH ***"
                )
                print(f"  Content preview: {str(content)[:200]}")
        else:
            print("WARNING: No chunks selected!")
        print("=== END CHUNK DIAGNOSTIC ===")
        
        # Build context from selected chunks
        context = self._build_context(chunks_to_use)
        
        print(f"=== CONTEXT DIAGNOSTIC ===")
        print(f"Context length: {len(context)} chars")
        print(f"Context preview (first 500 chars):\n{context[:500]}")
        print("=== END CONTEXT ===")
        
        # Get JSON prompt for intent
        prompt = get_prompt(intent_type, context, query)
        
        print(f"=== PROMPT DIAGNOSTIC ===")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"Prompt preview (first 800 chars):\n{prompt[:800]}")
        print("=== END PROMPT ===")
        
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

        # Generate with optimized settings for speed
        # Reduced context window (8K) and max_tokens (512) for 2-3x faster generation
        # format="json" forces Ollama to output valid JSON
        raw_response = self.llm.generate(
            prompt,
            temperature=0.2,
            max_tokens=512,  # Reduced from 1024 for even faster generation
            format="json"
        )
        
        # DIAGNOSTIC: Log what Ollama actually returns
        print(f"=== RAW OLLAMA RESPONSE ===")
        print(f"Length: {len(raw_response)} chars")
        print(f"First 500 chars: {raw_response[:500]}")
        print(f"Last 100 chars: {raw_response[-100:]}")
        print("=== END RAW RESPONSE ===")

        # Parse and validate JSON
        structured = self._parse_json_response(raw_response, intent_type, query)
        
        # Auto-generate footer from cited sources if not provided by LLM
        if not structured.footer:
            sources_used = set()
            for section in structured.sections:
                # Extract citation numbers from content and items
                content_text = section.content or ""
                items_text = " ".join(section.items or [])
                all_text = content_text + " " + items_text
                
                citations = re.findall(r'\[(\d+)\]', all_text)
                for citation in citations:
                    idx = int(citation) - 1
                    if 0 <= idx < len(chunks_to_use):
                        doc_name = chunks_to_use[idx].get("metadata", {}).get("document_name", "")
                        if doc_name:
                            # Clean up document name
                            clean_name = doc_name.replace(".pdf", "").replace("_", " ")
                            sources_used.add(clean_name)
            
            if sources_used:
                structured.footer = "Sources: " + ", ".join(sorted(sources_used))

        # Extract sources
        sources = self._extract_sources(chunks_to_use)

        # Calculate confidence
        confidence = self._calculate_confidence(chunks_to_use, structured)
        
        # Log context usage
        self._log_context_usage(query, chunks_to_use, estimated_context_tokens)
        
        return {
            "structured": structured.dict(),
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
        Build numbered context string from retrieved chunks for citation tracking.
        
        Each chunk is numbered [1], [2], etc. for citation in the response.
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Text is at top level, not in metadata
            content = chunk.get("text", "")
            doc_name = chunk.get("metadata", {}).get("document_name", "Unknown Document")
            section = chunk.get("metadata", {}).get("section_title", "")
            
            # Build source header
            source_header = f"Source: {doc_name}"
            if section:
                source_header += f" — {section}"
            
            # Number chunk for citation
            numbered = f"[{i}] {source_header}\n{content}"
            context_parts.append(numbered)
        
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
                "query": query[:100],  # Truncate for privacy
                "chunks_used": len(chunks),
                "tokens_used": tokens_used,
                "tokens_available": self.AVAILABLE_FOR_CHUNKS,
                "utilization": tokens_used / self.AVAILABLE_FOR_CHUNKS
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        except Exception as e:
            self.logger.debug(f"Failed to log context usage: {e}")
