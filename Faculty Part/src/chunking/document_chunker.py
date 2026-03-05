"""
Complete chunking strategy for all document types using BAAI/bge-m3.

Replaces the old 512 token multi-layer strategy with document-type-specific chunking.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class Chunk:
    """Represents a single chunk with text and metadata."""
    text: str
    metadata: Dict[str, Any]
    char_count: int
    token_count: int


class DocumentChunker:
    """
    Document-type-specific chunking for BAAI/bge-m3 (8192 token context).
    
    Detects document type and applies appropriate chunking strategy.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seen_hashes = set()  # For duplicate detection
    
    def detect_source_type(self, filepath: Path) -> str:
        """
        Detect document source type from filepath and extension.
        
        Returns one of: faculty_profile, hr_policy, legal_document,
        guidelines, procedure_document, form_document, general_document
        """
        path_lower = str(filepath).lower()
        filename = filepath.name.lower()
        
        # Faculty profiles (JSON/CSV)
        if "faculty" in path_lower and filepath.suffix in [".json", ".csv"]:
            return "faculty_profile"
        
        # HR policies
        if any(x in path_lower for x in ["hr", "leave", "salary", "payroll", "employee_resource"]):
            return "hr_policy"
        
        # Legal documents
        if any(x in path_lower for x in ["legal", "compliance", "act", "statute", "agreement", "employment_agreement"]):
            return "legal_document"
        
        # Guidelines and handbooks
        if any(x in path_lower for x in ["guideline", "handbook", "manual", "resource", "compendium"]):
            return "guidelines"
        
        # Procedures
        if any(x in path_lower for x in ["procedure", "process", "sop", "application"]):
            return "procedure_document"
        
        # Forms
        if any(x in path_lower for x in ["form", "template"]):
            return "form_document"
        
        return "general_document"
    
    def chunk_document(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Main entry point: chunk document based on detected type.
        
        Args:
            text: Full document text
            filepath: Path to source file
            doc_metadata: Document-level metadata
        
        Returns:
            List of chunks with metadata
        """
        source_type = self.detect_source_type(filepath)
        self.logger.info(f"Chunking {filepath.name} as {source_type}")
        
        if source_type == "faculty_profile":
            return self._chunk_faculty_profile(text, filepath, doc_metadata)
        elif source_type == "hr_policy":
            return self._chunk_hr_policy(text, filepath, doc_metadata)
        elif source_type == "legal_document":
            return self._chunk_legal_document(text, filepath, doc_metadata)
        elif source_type == "guidelines":
            return self._chunk_guidelines(text, filepath, doc_metadata)
        elif source_type == "procedure_document":
            return self._chunk_procedure(text, filepath, doc_metadata)
        elif source_type == "form_document":
            return self._chunk_form(text, filepath, doc_metadata)
        else:
            return self._chunk_general(text, filepath, doc_metadata)
    
    # ========== FACULTY PROFILE CHUNKING ==========
    
    def _chunk_faculty_profile(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk faculty profiles: One complete chunk per faculty member.
        
        Only split if profile exceeds 8192 tokens.
        """
        chunks = []
        
        # Faculty profiles are already formatted by document_processor
        # Each profile is separated by "=" * 60
        profiles = text.split("=" * 60)
        
        for profile_text in profiles:
            profile_text = profile_text.strip()
            if not profile_text:
                continue
            
            # Extract faculty metadata
            metadata = self._extract_faculty_metadata(profile_text, filepath, doc_metadata)
            
            # Check token count
            token_count = self._count_tokens(profile_text)
            
            if token_count <= 8000:  # Safe margin below 8192
                # Single chunk
                chunks.append(Chunk(
                    text=profile_text,
                    metadata=metadata,
                    char_count=len(profile_text),
                    token_count=token_count
                ))
            else:
                # Split into 2 chunks: Bio + Publications
                chunks.extend(self._split_large_faculty_profile(
                    profile_text, metadata, filepath
                ))
        
        return chunks
    
    def _extract_faculty_metadata(
        self,
        profile_text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata from faculty profile text."""
        # Extract name (first line after "Faculty:")
        name_match = re.search(r'Faculty:\s*(.+?)(?:\n|$)', profile_text)
        raw_name = name_match.group(1).strip() if name_match else ""
        
        # Strip titles for clean name
        clean_name = self._strip_titles(raw_name).lower().strip()
        name_parts = [p for p in clean_name.split() if len(p) > 1]
        
        # Extract department
        dept_match = re.search(r'Department:\s*(.+?)(?:\n|$)', profile_text, re.IGNORECASE)
        department = dept_match.group(1).strip().lower() if dept_match else ""
        
        # Extract email
        email_match = re.search(r'Email:\s*(.+?)(?:\n|$)', profile_text, re.IGNORECASE)
        email = email_match.group(1).strip() if email_match else ""
        
        # Extract research tags
        research_tags = self._extract_research_tags(profile_text)
        
        return {
            "source_type": "faculty_profile",
            "chunk_type": "full_profile",
            "document_name": filepath.name,
            "original_name": raw_name,
            "person_name": clean_name,
            "name_variants": name_parts,
            "department": department,
            "email": email,
            "research_tags": research_tags,
            "topic_tags": ["faculty", "profile"] + research_tags[:5],
            **doc_metadata
        }
    
    def _extract_research_tags(self, text: str) -> List[str]:
        """Extract research interest keywords from profile."""
        # Look for Research Interests section
        match = re.search(r'Research Interests?:\s*(.+?)(?:\n\n|\nPublications?:|\nAwards?:|$)', 
                         text, re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        
        interests = match.group(1).strip()
        # Split by common delimiters
        tags = re.split(r'[,;•\n]', interests)
        # Clean and lowercase
        tags = [tag.strip().lower() for tag in tags if tag.strip()]
        # Remove common words
        stopwords = {'and', 'or', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        tags = [tag for tag in tags if tag not in stopwords and len(tag) > 2]
        return tags[:10]  # Limit to 10 tags
    
    def _split_large_faculty_profile(
        self,
        profile_text: str,
        metadata: Dict[str, Any],
        filepath: Path
    ) -> List[Chunk]:
        """Split large faculty profile into 2 chunks."""
        # Find split point (before Publications or Awards)
        split_patterns = [
            r'\nPublications?:',
            r'\nAwards?:',
            r'\nExperience:',
        ]
        
        split_pos = None
        for pattern in split_patterns:
            match = re.search(pattern, profile_text, re.IGNORECASE)
            if match:
                split_pos = match.start()
                break
        
        if not split_pos:
            # No good split point, split at midpoint
            split_pos = len(profile_text) // 2
        
        chunk_a = profile_text[:split_pos].strip()
        chunk_b = profile_text[split_pos:].strip()
        
        # Add name to chunk B for context
        name = metadata.get("original_name", "")
        if name and not chunk_b.startswith("Faculty:"):
            chunk_b = f"Faculty: {name}\n\n{chunk_b}"
        
        # Update metadata for both chunks
        metadata_a = {**metadata, "chunk_part": "A", "has_sibling": True}
        metadata_b = {**metadata, "chunk_part": "B", "has_sibling": True}
        
        return [
            Chunk(
                text=chunk_a,
                metadata=metadata_a,
                char_count=len(chunk_a),
                token_count=self._count_tokens(chunk_a)
            ),
            Chunk(
                text=chunk_b,
                metadata=metadata_b,
                char_count=len(chunk_b),
                token_count=self._count_tokens(chunk_b)
            )
        ]
    
    # ========== HR POLICY CHUNKING ==========
    
    def _chunk_hr_policy(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk HR policy documents by section.
        Max 2000 tokens, 200 token overlap.
        """
        return self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="hr_policy",
            chunk_type="policy_section",
            max_tokens=2000,
            overlap_tokens=200
        )
    
    # ========== LEGAL DOCUMENT CHUNKING ==========
    
    def _chunk_legal_document(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk legal documents by article/clause.
        Max 1500 tokens, 150 token overlap.
        """
        return self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="legal_document",
            chunk_type="legal_clause",
            max_tokens=1500,
            overlap_tokens=150
        )
    
    # ========== GUIDELINES CHUNKING ==========
    
    def _chunk_guidelines(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk guidelines and handbooks by topic.
        Max 2500 tokens, 250 token overlap.
        """
        return self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="guidelines",
            chunk_type="guideline_section",
            max_tokens=2500,
            overlap_tokens=250
        )
    
    # ========== PROCEDURE CHUNKING ==========
    
    def _chunk_procedure(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk procedure documents - keep procedures complete.
        Max 3000 tokens, 300 token overlap.
        """
        return self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="procedure_document",
            chunk_type="procedure",
            max_tokens=3000,
            overlap_tokens=300
        )
    
    # ========== FORM CHUNKING ==========
    
    def _chunk_form(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk form documents - each form as one chunk.
        Max 1000 tokens, no overlap.
        """
        # Forms are typically small, treat as single chunk
        chunks = []
        
        # Split by form boundaries if multiple forms
        form_sections = re.split(r'\n(?=Form Name:|FORM\s+\w+)', text)
        
        for section in form_sections:
            section = section.strip()
            if not section:
                continue
            
            token_count = self._count_tokens(section)
            
            # Extract form name
            form_name_match = re.search(r'Form Name:\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
            form_name = form_name_match.group(1).strip() if form_name_match else ""
            
            metadata = {
                "source_type": "form_document",
                "chunk_type": "form",
                "document_name": filepath.name,
                "form_name": form_name,
                "topic_tags": ["form", "application"],
                **doc_metadata
            }
            
            chunks.append(Chunk(
                text=section,
                metadata=metadata,
                char_count=len(section),
                token_count=token_count
            ))
        
        return chunks
    
    # ========== GENERAL DOCUMENT CHUNKING ==========
    
    def _chunk_general(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk general documents.
        Max 1500 tokens, 150 token overlap.
        """
        return self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="general_document",
            chunk_type="general_section",
            max_tokens=1500,
            overlap_tokens=150
        )
    
    # ========== HELPER METHODS ==========
    
    def _chunk_by_sections(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any],
        source_type: str,
        chunk_type: str,
        max_tokens: int,
        overlap_tokens: int
    ) -> List[Chunk]:
        """
        Generic section-based chunking with size limits.
        
        Priority:
        1. Split on headers
        2. If section too large, split on paragraphs
        3. If still too large, split by size with overlap
        """
        chunks = []
        
        # Split by headers first
        sections = self._split_by_headers(text)
        
        for i, section in enumerate(sections):
            section_text = section["text"].strip()
            section_title = section["title"]
            
            if not section_text:
                continue
            
            token_count = self._count_tokens(section_text)
            
            if token_count <= max_tokens:
                # Section fits in one chunk
                metadata = {
                    "source_type": source_type,
                    "chunk_type": chunk_type,
                    "document_name": filepath.name,
                    "section_title": section_title,
                    "section_index": i,
                    "topic_tags": self._extract_topic_tags(section_text),
                    "has_steps": self._has_numbered_steps(section_text),
                    "has_forms": "form" in section_text.lower(),
                    **doc_metadata
                }
                
                chunks.append(Chunk(
                    text=section_text,
                    metadata=metadata,
                    char_count=len(section_text),
                    token_count=token_count
                ))
            else:
                # Section too large, split further
                sub_chunks = self._split_by_size(
                    section_text,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens
                )
                
                for j, sub_text in enumerate(sub_chunks):
                    metadata = {
                        "source_type": source_type,
                        "chunk_type": chunk_type,
                        "document_name": filepath.name,
                        "section_title": section_title,
                        "section_index": i,
                        "sub_index": j,
                        "topic_tags": self._extract_topic_tags(sub_text),
                        "has_steps": self._has_numbered_steps(sub_text),
                        "has_forms": "form" in sub_text.lower(),
                        **doc_metadata
                    }
                    
                    chunks.append(Chunk(
                        text=sub_text,
                        metadata=metadata,
                        char_count=len(sub_text),
                        token_count=self._count_tokens(sub_text)
                    ))
        
        return chunks
    
    def _split_by_headers(self, text: str) -> List[Dict[str, str]]:
        """Split text by section headers."""
        sections = []
        
        # Header patterns (in priority order)
        header_patterns = [
            r'\n([A-Z][A-Z\s]{5,}):?\n',  # ALL CAPS headers
            r'\n((?:Section|Chapter|Article)\s+\d+[:\.]?\s*.+?)\n',  # Section/Chapter/Article
            r'\n(\d+\.\s+[A-Z].+?)\n',  # Numbered headers "1. Title"
            r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\n',  # Title Case headers
        ]
        
        # Find all headers
        header_positions = []
        for pattern in header_patterns:
            for match in re.finditer(pattern, text):
                header_positions.append((match.start(), match.end(), match.group(1).strip()))
        
        # Sort by position
        header_positions.sort(key=lambda x: x[0])
        
        if not header_positions:
            # No headers found, return entire text as one section
            return [{"title": "", "text": text}]
        
        # Extract sections
        for i, (start, end, title) in enumerate(header_positions):
            # Get text until next header or end
            if i < len(header_positions) - 1:
                section_text = text[end:header_positions[i + 1][0]]
            else:
                section_text = text[end:]
            
            sections.append({"title": title, "text": section_text.strip()})
        
        # Add text before first header if exists
        if header_positions[0][0] > 0:
            intro_text = text[:header_positions[0][0]].strip()
            if intro_text:
                sections.insert(0, {"title": "Introduction", "text": intro_text})
        
        return sections
    
    def _split_by_size(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int
    ) -> List[str]:
        """Split text by size with overlap."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens <= max_tokens:
                current_chunk += para + "\n\n"
                current_tokens += para_tokens
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunks and overlap_tokens > 0:
                    # Add last part of previous chunk as overlap
                    overlap_text = self._get_last_n_tokens(chunks[-1], overlap_tokens)
                    current_chunk = overlap_text + "\n\n" + para + "\n\n"
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = para + "\n\n"
                    current_tokens = para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_last_n_tokens(self, text: str, n_tokens: int) -> str:
        """Get approximately last n tokens from text."""
        # Rough approximation: 4 chars per token
        n_chars = n_tokens * 4
        return text[-n_chars:] if len(text) > n_chars else text
    
    def _extract_topic_tags(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        # Simple keyword extraction
        text_lower = text.lower()
        
        keywords = []
        
        # Common faculty/HR keywords
        keyword_list = [
            "leave", "salary", "policy", "procedure", "application",
            "form", "faculty", "research", "publication", "award",
            "eligibility", "requirement", "deadline", "approval",
            "department", "hr", "legal", "compliance"
        ]
        
        for keyword in keyword_list:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords[:10]  # Limit to 10 tags
    
    def _has_numbered_steps(self, text: str) -> bool:
        """Check if text contains numbered steps."""
        # Look for patterns like "1.", "Step 1", etc.
        patterns = [
            r'\n\s*\d+\.\s+',
            r'\n\s*Step\s+\d+',
            r'\n\s*\(\d+\)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _strip_titles(self, name: str) -> str:
        """Strip titles from name."""
        titles = [
            "associate professor", "assistant professor", "professor",
            "prof.", "prof", "doctor", "dr.", "dr",
            "mrs.", "mrs", "mr.", "mr", "ms.", "ms",
            "sir", "ma'am", "madam"
        ]
        
        name_lower = name.lower()
        for title in titles:
            name_lower = re.sub(r'\b' + re.escape(title) + r'\b', '', name_lower)
        
        return ' '.join(name_lower.split()).strip()
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4
    
    def should_skip_chunk(self, chunk: Chunk) -> Tuple[bool, Optional[str]]:
        """
        Quality filter: determine if chunk should be skipped.
        
        Returns: (should_skip, reason)
        """
        # Too short
        if chunk.token_count < 50:
            return True, "too_short"
        
        # No alphabetic content
        if not any(c.isalpha() for c in chunk.text):
            return True, "no_alphabetic_content"
        
        # Duplicate check
        chunk_hash = hashlib.md5(chunk.text.encode()).hexdigest()
        if chunk_hash in self.seen_hashes:
            return True, "duplicate"
        
        self.seen_hashes.add(chunk_hash)
        
        return False, None
    
    def clean_chunk_text(self, text: str) -> str:
        """Clean chunk text before embedding."""
        # Strip excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove repeated newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode common unicode issues
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', '')
        
        return text.strip()
