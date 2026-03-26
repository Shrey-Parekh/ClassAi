"""
Semantic chunker that creates chunks by meaning, not token count.
Implements the three-level chunking strategy.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

from config.chunking_config import (
    ChunkLevel,
    ContentType,
    MAX_LEVEL2_TOKENS,
    OVERLAP_TOKENS,
)


@dataclass
class Chunk:
    """Represents a semantic chunk with metadata."""
    content: str
    level: ChunkLevel
    content_type: ContentType
    metadata: Dict[str, Any]
    token_count: int
    chunk_id: str
    parent_doc_id: str
    superseded_by: Optional[str] = None  # For outdated documents


class SemanticChunker:
    """
    Creates meaning-complete chunks from faculty documents.
    
    Core principle: A chunk must answer a question independently.
    """
    
    def __init__(self, tokenizer=None, llm_client=None):
        """
        Initialize with optional custom tokenizer and LLM client.
        
        Args:
            tokenizer: Custom tokenizer function
            llm_client: Optional LLM client for generating overviews
        """
        self.tokenizer = tokenizer or self._default_tokenizer
        self.llm_client = llm_client
    
    def chunk_document(
        self,
        content: str,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Main entry point: chunk a complete document into all three levels.
        
        Args:
            content: Full document text
            doc_metadata: Document-level metadata (title, date, applies_to, etc.)
        
        Returns:
            List of chunks at all three levels
        """
        chunks = []
        doc_id = doc_metadata.get("doc_id", "unknown")
        
        # Level 1: Create document overview
        overview_chunk = self._create_overview_chunk(content, doc_metadata)
        chunks.append(overview_chunk)
        
        # Level 2: Extract complete procedures/policies
        sections = self._split_into_sections(content)
        for section in sections:
            procedure_chunks = self._create_procedure_chunks(
                section, doc_id, doc_metadata
            )
            chunks.extend(procedure_chunks)
        
        # Level 3: Extract atomic facts
        atomic_chunks = self._extract_atomic_facts(content, doc_id, doc_metadata)
        chunks.extend(atomic_chunks)
        
        return chunks
    
    def _create_overview_chunk(
        self,
        content: str,
        doc_metadata: Dict[str, Any]
    ) -> Chunk:
        """
        Level 1: Create a single overview chunk for the entire document.
        
        Captures: what this is about, who it applies to, what topics it covers.
        """
        # Extract key information for overview
        title = doc_metadata.get("title", "Untitled Document")
        applies_to = doc_metadata.get("applies_to", "All faculty")
        
        # Create summary (in production, use LLM to generate this)
        overview_text = f"""Document: {title}
Applies to: {applies_to}

This document covers: {self._extract_topics(content)}

Use this document when you need information about: {self._extract_use_cases(content)}
"""
        
        return Chunk(
            content=overview_text,
            level=ChunkLevel.OVERVIEW,
            content_type=ContentType.POLICY,  # Default, refine based on doc type
            metadata={
                **doc_metadata,
                "is_overview": True,
            },
            token_count=self.tokenizer(overview_text),
            chunk_id=f"{doc_metadata.get('doc_id')}_overview",
            parent_doc_id=doc_metadata.get("doc_id", "unknown"),
        )
    
    def _split_into_sections(self, content: str) -> List[str]:
        """
        Split document at natural boundaries (headings, section breaks).
        
        NEVER splits mid-procedure or mid-rule.
        """
        # Split on common heading patterns
        section_pattern = r'\n(?=(?:[A-Z][A-Z\s]+:|\d+\.\s+[A-Z]|#{1,3}\s))'
        sections = re.split(section_pattern, content)
        
        return [s.strip() for s in sections if s.strip()]
    
    def _create_procedure_chunks(
        self,
        section: str,
        doc_id: str,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Level 2: Create complete procedure/policy chunks.
        
        This is the primary retrieval unit. Keep under 600 tokens but
        NEVER break meaning to hit that limit.
        """
        chunks = []
        
        # Detect if this is a procedure (has steps)
        is_procedure = self._is_procedure(section)
        
        if is_procedure:
            # Keep all steps together - this is ONE chunk
            content_type = ContentType.PROCEDURE
        else:
            # Check for other content types
            content_type = self._detect_content_type(section)
        
        token_count = self.tokenizer(section)
        
        # Only split if significantly over limit AND can be split meaningfully
        if token_count > MAX_LEVEL2_TOKENS * 1.5:
            # Try to find natural sub-sections
            sub_chunks = self._split_large_section(section, doc_id, doc_metadata)
            chunks.extend(sub_chunks)
        else:
            # Keep as single chunk
            chunk = Chunk(
                content=section,
                level=ChunkLevel.PROCEDURE,
                content_type=content_type,
                metadata={
                    **doc_metadata,
                    "has_steps": is_procedure,
                },
                token_count=token_count,
                chunk_id=f"{doc_id}_proc_{len(chunks)}",
                parent_doc_id=doc_id,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_atomic_facts(
        self,
        content: str,
        doc_id: str,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Level 3: Extract standalone facts, rules, deadlines, definitions.
        
        Examples:
        - "Casual leave entitlement is 12 days per year"
        - "Submit application 7 days before leave starts"
        """
        chunks = []
        
        # Extract definitions
        definitions = self._extract_definitions(content)
        for idx, definition in enumerate(definitions):
            chunks.append(Chunk(
                content=definition,
                level=ChunkLevel.ATOMIC,
                content_type=ContentType.DEFINITION,
                metadata=doc_metadata,
                token_count=self.tokenizer(definition),
                chunk_id=f"{doc_id}_def_{idx}",
                parent_doc_id=doc_id,
            ))
        
        # Extract deadlines
        deadlines = self._extract_deadlines(content)
        for idx, deadline in enumerate(deadlines):
            chunks.append(Chunk(
                content=deadline,
                level=ChunkLevel.ATOMIC,
                content_type=ContentType.DEADLINE,
                metadata=doc_metadata,
                token_count=self.tokenizer(deadline),
                chunk_id=f"{doc_id}_deadline_{idx}",
                parent_doc_id=doc_id,
            ))
        
        # Extract standalone rules
        rules = self._extract_rules(content)
        for idx, rule in enumerate(rules):
            chunks.append(Chunk(
                content=rule,
                level=ChunkLevel.ATOMIC,
                content_type=ContentType.RULE,
                metadata=doc_metadata,
                token_count=self.tokenizer(rule),
                chunk_id=f"{doc_id}_rule_{idx}",
                parent_doc_id=doc_id,
            ))
        
        return chunks
    
    # Helper methods
    
    def _is_procedure(self, text: str) -> bool:
        """Detect if text contains a procedure with steps."""
        step_patterns = [
            r'step\s+\d+',
            r'\d+\.\s+[A-Z]',
            r'first.*second.*third',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in step_patterns)
    
    def _detect_content_type(self, text: str) -> ContentType:
        """Detect the primary content type of a section."""
        text_lower = text.lower()
        
        if 'form' in text_lower and 'field' in text_lower:
            return ContentType.FORM
        elif 'circular' in text_lower or 'notification' in text_lower:
            return ContentType.CIRCULAR
        elif 'if' in text_lower and 'then' in text_lower:
            return ContentType.RULE
        elif 'policy' in text_lower or 'applies to' in text_lower:
            return ContentType.POLICY
        else:
            return ContentType.POLICY  # Default
    
    def _extract_topics(self, content: str) -> str:
        """Extract main topics from document."""
        if self.llm_client:
            try:
                prompt = f"""Extract 3-5 main topics from this document in a comma-separated list.

Document excerpt:
{content[:1000]}

Topics (comma-separated):"""
                
                topics = self.llm_client.generate(prompt, max_tokens=100, temperature=0.3)
                return topics.strip()
            except Exception:
                pass
        
        # Fallback: rule-based extraction
        keywords = ["leave", "policy", "procedure", "application", "eligibility", "faculty", "research"]
        found = [kw for kw in keywords if kw in content.lower()]
        return ", ".join(found[:5]) if found else "general policies and procedures"
    
    def _extract_use_cases(self, content: str) -> str:
        """Extract use cases from document."""
        if self.llm_client:
            try:
                prompt = f"""What questions can this document answer? List 2-3 use cases.

Document excerpt:
{content[:1000]}

Use cases (comma-separated):"""
                
                use_cases = self.llm_client.generate(prompt, max_tokens=100, temperature=0.3)
                return use_cases.strip()
            except Exception:
                pass
        
        # Fallback: rule-based extraction
        if "leave" in content.lower():
            return "applying for leave, checking leave eligibility, understanding leave policies"
        elif "research" in content.lower():
            return "research funding, publication guidelines, research policies"
        else:
            return "faculty policies, procedures, and guidelines"
    
    def _split_large_section(
        self,
        section: str,
        doc_id: str,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split a large section while preserving meaning."""
        # This is a fallback - try to find paragraph boundaries
        paragraphs = section.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if self.tokenizer(current_chunk + para) < MAX_LEVEL2_TOKENS:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(self._make_chunk(current_chunk, doc_id, doc_metadata, len(chunks)))
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, doc_id, doc_metadata, len(chunks)))
        
        return chunks
    
    def _make_chunk(self, content: str, doc_id: str, metadata: Dict, idx: int) -> Chunk:
        """Helper to create a chunk object."""
        return Chunk(
            content=content,
            level=ChunkLevel.PROCEDURE,
            content_type=self._detect_content_type(content),
            metadata=metadata,
            token_count=self.tokenizer(content),
            chunk_id=f"{doc_id}_proc_{idx}",
            parent_doc_id=doc_id,
        )
    
    def _extract_definitions(self, content: str) -> List[str]:
        """Extract definition statements."""
        # Pattern: "X is defined as Y" or "X means Y"
        pattern = r'([A-Z][^.!?]*(?:is defined as|means|refers to)[^.!?]*[.!?])'
        matches = re.findall(pattern, content)
        return matches
    
    def _extract_deadlines(self, content: str) -> List[str]:
        """Extract deadline statements."""
        # Pattern: mentions of time + action
        pattern = r'([^.!?]*(?:\d+\s+days?|before|after|deadline)[^.!?]*[.!?])'
        matches = re.findall(pattern, content, re.IGNORECASE)
        return matches
    
    def _extract_rules(self, content: str) -> List[str]:
        """Extract conditional rules (if-then statements)."""
        # Pattern: "If X then Y" or "When X, Y"
        pattern = r'((?:If|When)[^.!?]*(?:then|,)[^.!?]*[.!?])'
        matches = re.findall(pattern, content, re.IGNORECASE)
        return matches
    
    def _default_tokenizer(self, text: str) -> int:
        """Simple word-based token counter (replace with actual tokenizer)."""
        return len(text.split())
