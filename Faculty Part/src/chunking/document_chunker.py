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
    chunk_level: str = "section"  # Default: "overview" | "section" | "atomic"
    chunk_id: str = ""  # Generated unique ID


class DocumentChunker:
    """
    Document-type-specific chunking for BAAI/bge-m3 (8192 token context).
    
    Detects document type and applies appropriate chunking strategy.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Document-scoped deduplication: reset by creating new chunker per document
        self.seen_hashes = set()
    
    def _generate_chunk_id(self, filepath: Path, index: int, sub_index: int = None) -> str:
        """Generate unique chunk ID from document name and position."""
        base = f"{filepath.stem}_chunk_{index}"
        if sub_index is not None:
            base += f"_{sub_index}"
        return base
    
    def detect_source_type(self, filepath: Path) -> str:
        """
        Detect document source type from filepath and extension.

        Uses weighted keyword scoring so files matching multiple categories
        (e.g. "faculty_leave_policy.pdf") resolve to the highest-scoring type.
        Tie-breaks by priority order.

        Returns one of: faculty_profile, hr_policy, legal_document,
        guidelines, procedure_document, form_document, general_document
        """
        path_lower = str(filepath).lower()

        # Faculty profiles only match JSON/CSV
        if (any(x in path_lower for x in ["faculty", "facult"])
                and filepath.suffix in [".json", ".csv"]):
            return "faculty_profile"

        keyword_map = {
            "form_document":       ["form", "template", "compendium", "application"],
            "procedure_document":  ["procedure", "process", "sop"],
            "hr_policy":           ["hr", "leave", "salary", "payroll", "employee_resource"],
            "legal_document":      ["legal", "compliance", "act", "statute", "agreement", "employment_agreement"],
            "guidelines":          ["guideline", "handbook", "manual", "resource"],
        }

        # Priority order for tie-breaking (lower index = higher priority)
        priority = ["form_document", "procedure_document", "hr_policy",
                    "legal_document", "guidelines", "faculty_profile", "general_document"]

        scores: Dict[str, int] = {}
        for source_type, keywords in keyword_map.items():
            scores[source_type] = sum(1 for kw in keywords if kw in path_lower)

        self.logger.debug(f"detect_source_type scores for {filepath.name}: {scores}")

        best_type = max(
            (t for t in scores if scores[t] > 0),
            key=lambda t: (scores[t], -priority.index(t)),
            default=None
        )

        return best_type if best_type else "general_document"
    
    def chunk_document(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Main entry point: chunk document based on detected type.
        
        After type-specific chunking, assigns chunk levels (overview/section/atomic)
        based on position and content characteristics.
        
        Args:
            text: Full document text
            filepath: Path to source file
            doc_metadata: Document-level metadata
        
        Returns:
            List of chunks with metadata and level labels
        """
        source_type = self.detect_source_type(filepath)
        self.logger.info(f"Chunking {filepath.name} as {source_type}")
        # C6: reset per-document to prevent cross-document boilerplate suppression
        self.seen_hashes = set()
        
        if source_type == "faculty_profile":
            chunks = self._chunk_faculty_profile(text, filepath, doc_metadata)
        elif source_type == "hr_policy":
            chunks = self._chunk_hr_policy(text, filepath, doc_metadata)
        elif source_type == "legal_document":
            chunks = self._chunk_legal_document(text, filepath, doc_metadata)
        elif source_type == "guidelines":
            chunks = self._chunk_guidelines(text, filepath, doc_metadata)
        elif source_type == "procedure_document":
            chunks = self._chunk_procedure(text, filepath, doc_metadata)
        elif source_type == "form_document":
            chunks = self._chunk_form(text, filepath, doc_metadata)
        else:
            chunks = self._chunk_general(text, filepath, doc_metadata)
        
        # Assign chunk levels after type-specific chunking
        chunks = self._assign_chunk_levels(chunks, text)
        
        return chunks
    
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
        # Each profile is separated by "=" * 60 (use regex to tolerate 50-65 equals)
        profiles = re.split(r'={50,}', text)
        
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
                    token_count=token_count,
                    chunk_id=self._generate_chunk_id(filepath, 0)
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
        # Keep all parts including single-letter initials (A. K. Gupta → ["a", "k", "gupta"])
        name_parts = clean_name.split()
        # Also store concatenated initials form for robust matching ("akgupta")
        initials = "".join(p[0] for p in name_parts if p)
        last_name = name_parts[-1] if name_parts else ""
        if initials and last_name and initials != last_name:
            name_parts = name_parts + [initials + last_name]
        
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
        """Extract research interest keywords from profile.

        Splits only on commas, semicolons, and newlines to preserve
        multi-word phrases like 'drug discovery' or 'machine learning'.
        Single-word stopwords are removed but multi-word phrases are kept.
        """
        match = re.search(
            r'Research Interests?:\s*(.+?)(?:\n\n|\nPublications?:|\nAwards?:|$)',
            text, re.IGNORECASE | re.DOTALL
        )

        tags = []

        if match:
            interests = match.group(1).strip()
            # Split only on commas/semicolons/newlines — NOT on spaces
            # This preserves "drug discovery", "machine learning", etc.
            raw_tags = re.split(r'[,;\n•]', interests)
            stopwords = {'and', 'or', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'a', 'an'}
            for tag in raw_tags:
                tag = tag.strip().lower()
                if not tag:
                    continue
                # Keep multi-word phrases as-is; filter single stopwords
                words = tag.split()
                if len(words) == 1 and tag in stopwords:
                    continue
                if len(tag) > 2:
                    tags.append(tag)

        if not tags or (len(tags) == 1 and "not specified" in tags[0].lower()):
            tags = self._extract_research_tags_from_publications(text)

        return tags[:10]
    
    def _extract_research_tags_from_publications(self, text: str) -> List[str]:
        """Extract research topics from publication titles when research_interests is empty."""
        # Extract publications section
        pubs_match = re.search(r'Publications?:\s*(.+?)(?:\nAwards?:|\nExperience:|$)', 
                              text, re.IGNORECASE | re.DOTALL)
        
        if not pubs_match:
            # Try JSON format
            pubs_match = re.search(r'"publications"\s*:\s*\{(.+?)\}', text, re.DOTALL)
        
        if not pubs_match:
            return []
        
        pub_text = pubs_match.group(1).lower()
        
        # Domain keywords mapping
        domain_keywords = {
            "finance": ["stock", "market", "financial", "investment", "currency", "bond", "crypto", "trading"],
            "machine learning": ["ml", "neural", "deep learning", "ai", "language model", "nlp", "computer vision"],
            "chemistry": ["membrane", "nanofiltration", "polymer", "fatty acid", "synthesis", "catalyst"],
            "economics": ["sme", "gdp", "trade", "monetary", "economic", "fiscal", "inflation"],
            "management": ["strategy", "leadership", "organizational", "hrm", "operations"],
            "marketing": ["consumer", "brand", "advertising", "digital marketing", "social media"],
        }
        
        found = []
        for domain, keywords in domain_keywords.items():
            if any(kw in pub_text for kw in keywords):
                found.append(domain)
        
        return found[:5]
    
    def _split_large_faculty_profile(
        self,
        profile_text: str,
        metadata: Dict[str, Any],
        filepath: Path
    ) -> List[Chunk]:
        """Split large faculty profile into 2 chunks."""
        # Find split point (before Publications or Awards)
        # Support both JSON and text formats
        split_patterns = [
            r'"publications"\s*:',  # JSON format
            r'"awards"\s*:',  # JSON format
            r'\nPublications?:',  # Text format
            r'\nAwards?:',  # Text format
            r'\nExperience:',
        ]
        
        split_pos = None
        for pattern in split_patterns:
            match = re.search(pattern, profile_text, re.IGNORECASE)
            if match:
                split_pos = match.start()
                break
        
        if not split_pos:
            # C2: walk from midpoint to nearest paragraph break instead of raw char slice
            mid = len(profile_text) // 2
            # Try to find a \n\n after the midpoint
            para_break = profile_text.find('\n\n', mid)
            if para_break == -1:
                # Fall back to nearest \n
                para_break = profile_text.find('\n', mid)
            split_pos = para_break if para_break != -1 else mid
        
        chunk_a = profile_text[:split_pos].strip()
        chunk_b = profile_text[split_pos:].strip()

        # Add compact context header to chunk B for cross-chunk retrieval (item 9)
        name = metadata.get("original_name", "")
        dept = metadata.get("department", "")
        research_tags = metadata.get("research_tags", [])
        top_tags = ", ".join(research_tags[:3]) if research_tags else ""

        context_header = f"Faculty: {name}"
        if dept:
            context_header += f" | Dept: {dept}"
        if top_tags:
            context_header += f" | Research: {top_tags}"

        if not chunk_b.startswith("Faculty:"):
            chunk_b = f"{context_header}\n\n{chunk_b}"

        metadata_a = {**metadata, "chunk_part": "A", "has_sibling": True}
        metadata_b = {**metadata, "chunk_part": "B", "has_sibling": True}

        chunks = [
            Chunk(
                text=chunk_a,
                metadata=metadata_a,
                char_count=len(chunk_a),
                token_count=self._count_tokens(chunk_a),
                chunk_id=self._generate_chunk_id(filepath, 0, 0)
            )
        ]

        # C3: recursively split chunk B if still over 7500 tokens
        if self._count_tokens(chunk_b) > 7500:
            chunks.extend(self._split_large_faculty_profile(chunk_b, metadata_b, filepath))
        else:
            chunks.append(Chunk(
                text=chunk_b,
                metadata=metadata_b,
                char_count=len(chunk_b),
                token_count=self._count_tokens(chunk_b),
                chunk_id=self._generate_chunk_id(filepath, 0, 1)
            ))

        return chunks
    
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
        
        Detects form references and tags chunks accordingly.
        """
        chunks = self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="hr_policy",
            chunk_type="policy_section",
            max_tokens=2000,
            overlap_tokens=200
        )
        
        # Detect and tag form references in chunks
        # Broader pattern to catch Annexures, Proformas, and descriptive references
        coded_pattern = r'\b(?:form|application|annexure|proforma)\s+[A-Za-z]{0,3}-?\d{1,3}\b'
        descriptive_pattern = r'\b(?:application form|prescribed form|requisite form|relevant form)\b'
        
        for chunk in chunks:
            form_matches = re.findall(coded_pattern, chunk.text, re.IGNORECASE)
            descriptive_matches = re.findall(descriptive_pattern, chunk.text, re.IGNORECASE)
            all_forms = form_matches + descriptive_matches
            
            if all_forms:
                chunk.metadata["has_forms"] = True
                chunk.metadata["form_references"] = list(set(all_forms))
        
        return chunks
    
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
        
        Detects form references and tags chunks accordingly.
        """
        chunks = self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="guidelines",
            chunk_type="guideline_section",
            max_tokens=2500,
            overlap_tokens=250
        )
        
        # Detect and tag form references in chunks
        # Broader pattern to catch Annexures, Proformas, and descriptive references
        coded_pattern = r'\b(?:form|application|annexure|proforma)\s+[A-Za-z]{0,3}-?\d{1,3}\b'
        descriptive_pattern = r'\b(?:application form|prescribed form|requisite form|relevant form)\b'
        
        for chunk in chunks:
            form_matches = re.findall(coded_pattern, chunk.text, re.IGNORECASE)
            descriptive_matches = re.findall(descriptive_pattern, chunk.text, re.IGNORECASE)
            all_forms = form_matches + descriptive_matches
            
            if all_forms:
                chunk.metadata["has_forms"] = True
                chunk.metadata["form_references"] = list(set(all_forms))
        
        return chunks
    
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
        
        Detects form references and tags chunks accordingly.
        """
        chunks = self._chunk_by_sections(
            text=text,
            filepath=filepath,
            doc_metadata=doc_metadata,
            source_type="procedure_document",
            chunk_type="procedure",
            max_tokens=3000,
            overlap_tokens=300
        )
        
        # Detect and tag form references in chunks
        # Broader pattern to catch Annexures, Proformas, and descriptive references
        coded_pattern = r'\b(?:form|application|annexure|proforma)\s+[A-Za-z]{0,3}-?\d{1,3}\b'
        descriptive_pattern = r'\b(?:application form|prescribed form|requisite form|relevant form)\b'
        
        for chunk in chunks:
            form_matches = re.findall(coded_pattern, chunk.text, re.IGNORECASE)
            descriptive_matches = re.findall(descriptive_pattern, chunk.text, re.IGNORECASE)
            all_forms = form_matches + descriptive_matches
            
            if all_forms:
                chunk.metadata["has_forms"] = True
                chunk.metadata["form_references"] = list(set(all_forms))
        
        return chunks
    
    # ========== FORM CHUNKING ==========
    
    def _chunk_form(
        self,
        text: str,
        filepath: Path,
        doc_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk form documents — each form as one complete chunk.
        
        Detects form boundaries using:
        1. ALL-CAPS form titles (e.g., "LEAVE APPLICATION FORM")
        2. Form code lines (e.g., "Form Code:\\nHR-LA-01")
        3. Institution header repeats that signal a new form
        
        Designed for the NMIMS Faculty Applications Compendium.
        """
        chunks = []
        
        # Split by form boundaries
        # Each form in the Compendium starts with the institution header:
        #   "NMIMS — Narsee Monjee Institute of Management Studies"
        #   followed by a form title in ALL CAPS ending with "FORM"
        #
        # Alternative boundary: Form Code line (HR-LA-01, HR-CO-01, etc.)
        
        # Primary split: institution header that precedes each form.
        # The institution header also repeats as a page header on every page
        # of a given form, so we must NOT split on every occurrence — we only
        # want occurrences that actually start a new form.
        #
        # A new form is marked by the institution header followed (within a
        # short window) by either:
        #   (a) a "Form Code:" line with an HR-XX-NN style code, OR
        #   (b) an ALL-CAPS line ending in FORM / APPLICATION
        form_boundary = (
            r'\n(?=NMIMS[\s\S]{1,10}Narsee Monjee Institute of Management Studies'
            r'(?:[\s\S]{0,600})'
            r'(?:Form\s+Code|[A-Z][A-Z\s/]{3,}(?:FORM|APPLICATION)))'
        )
        form_sections = re.split(form_boundary, text)

        # Fallback: if the strict boundary found nothing, fall back to the
        # loose institution-header split (also DOTALL-safe), then to ALL-CAPS form title split.
        if len(form_sections) <= 2:
            form_boundary_loose = r'\n(?=NMIMS[\s\S]{1,10}Narsee Monjee Institute of Management Studies)'
            form_sections = re.split(form_boundary_loose, text)

        if len(form_sections) <= 2:
            form_boundary_alt = r'\n(?=[A-Z][A-Z\s]+FORM\s*\n)'
            form_sections = re.split(form_boundary_alt, text)

        # Re-merge any section that does NOT contain a form code AND does not
        # contain a FORM/APPLICATION title — these are continuation pages that
        # got split by a repeated page header.
        merged = []
        code_re = re.compile(r'\b[A-Z]{2,3}-[A-Z]{1,3}-\d{1,3}\b')
        title_re = re.compile(r'^[A-Z][A-Z\s/]{3,}(?:FORM|APPLICATION)\b', re.MULTILINE)
        for sec in form_sections:
            if merged and not code_re.search(sec) and not title_re.search(sec):
                merged[-1] = merged[-1] + '\n' + sec
            else:
                merged.append(sec)
        form_sections = merged
        
        # First section is the Compendium cover/TOC
        for i, section in enumerate(form_sections):
            section = section.strip()
            if not section:
                continue
            
            token_count = self._count_tokens(section)
            
            # Extract form code (HR-LA-01, HR-CO-01, etc.)
            form_code_match = re.search(
                r'(?:Form\s+Code[:\s]*\n?\s*)?([A-Z]{2,3}-[A-Z]{1,3}-\d{1,3})',
                section, re.IGNORECASE
            )
            form_code = form_code_match.group(1).upper() if form_code_match else ""
            
            # Extract form title (ALL CAPS line containing "FORM" or "APPLICATION")
            form_title_match = re.search(
                r'^([A-Z][A-Z\s/]+(?:FORM|APPLICATION)[A-Z\s/]*)',
                section, re.MULTILINE
            )
            form_title = form_title_match.group(1).strip() if form_title_match else ""
            
            # Determine chunk type
            is_toc = i == 0 and "form code" in section.lower()[:200] and "form title" in section.lower()[:200]
            
            metadata = {
                "source_type": "form_document",
                "chunk_type": "form_toc" if is_toc else "form_template",
                "document_name": filepath.name,
                "section_title": form_title or form_code or f"Form Section {i}",
                "section_index": i,
                "form_code": form_code,
                "form_title": form_title,
                "topic_tags": self._extract_topic_tags(section),
                "has_forms": True,
                **doc_metadata
            }
            
            chunks.append(Chunk(
                text=section,
                metadata=metadata,
                char_count=len(section),
                token_count=token_count,
                chunk_id=self._generate_chunk_id(filepath, i)
            ))

            # Emit per-SECTION sub-chunks for FAC forms so queries that target
            # a single section (e.g. "Section B of HR-LA-01") can retrieve
            # just that block. Skip the TOC chunk.
            if not is_toc and form_code:
                # Match "Section A: Applicant Details", "SECTION B -",
                # "Part A:", "Part 1:", "Schedule I:", "Section 1:" etc.
                sub_pattern = re.compile(
                    r'(?m)^\s*(?:'
                    r'SECTION\s+([A-Z\d]+)'          # SECTION A / SECTION 1
                    r'|Part\s+([A-Z\d]+)'             # Part A / Part 1
                    r'|Schedule\s+([A-Z\d]+)'         # Schedule I / Schedule A
                    r')\b[^\n]*$',
                    re.IGNORECASE,
                )
                hits = list(sub_pattern.finditer(section))
                for j, m in enumerate(hits):
                    start = m.start()
                    end = hits[j + 1].start() if j + 1 < len(hits) else len(section)
                    sub_text = section[start:end].strip()
                    if not sub_text or self._count_tokens(sub_text) < 20:
                        continue
                    letter = next(g for g in m.groups() if g is not None).upper()
                    # C1: prepend form identifier so dense+BM25 can match "Section B of HR-LA-01"
                    breadcrumb = f"{form_code} — {form_title} · Section {letter}" if form_title else f"{form_code} · Section {letter}"
                    sub_text_with_header = f"{breadcrumb}\n\n{sub_text}"
                    sub_meta = {
                        **metadata,
                        "chunk_type": "form_section",
                        "section_letter": letter,
                        "section_title": f"{form_code} Section {letter}",
                        "parent_form_code": form_code,
                        "parent_section_index": i,
                        "sub_section_index": j,
                    }
                    chunks.append(Chunk(
                        text=sub_text_with_header,
                        metadata=sub_meta,
                        char_count=len(sub_text_with_header),
                        token_count=self._count_tokens(sub_text_with_header),
                        chunk_id=self._generate_chunk_id(filepath, f"{i}_{letter}")
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
        
        # Split by headers — NOW passes max_tokens for two-pass splitting
        sections = self._split_by_headers(text, max_tokens=max_tokens)
        
        for i, section in enumerate(sections):
            section_text = section["text"].strip()
            section_title = section["title"]

            if not section_text:
                continue

            # Build breadcrumb header for embedding context (items 2 & 5)
            doc_name = filepath.stem.replace("_", " ")
            form_code = doc_metadata.get("form_code", "")
            breadcrumb_parts = [doc_name]
            if form_code:
                breadcrumb_parts.append(form_code)
            if section_title:
                breadcrumb_parts.append(section_title)
            breadcrumb = "[" + " · ".join(breadcrumb_parts) + "]"

            token_count = self._count_tokens(section_text)

            # Item 16: use coded pattern with IGNORECASE for OCR-mixed-case form codes
            has_form_code = bool(re.search(r'\b[A-Z]{2,3}-[A-Z]{1,3}-\d{1,3}\b', section_text, re.IGNORECASE))

            if token_count <= max_tokens:
                # Section fits in one chunk — prepend breadcrumb
                chunk_text = f"{breadcrumb}\n{section_text}"
                metadata = {
                    "source_type": source_type,
                    "chunk_type": chunk_type,
                    "document_name": filepath.name,
                    "section_title": section_title,
                    "section_index": i,
                    "topic_tags": self._extract_topic_tags(section_text),
                    "has_steps": self._has_numbered_steps(section_text),
                    "has_forms": has_form_code,
                    **doc_metadata
                }

                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=metadata,
                    char_count=len(chunk_text),
                    token_count=self._count_tokens(chunk_text),
                    chunk_id=self._generate_chunk_id(filepath, i)
                ))
            else:
                # Section too large, split further
                sub_chunks = self._split_by_size(
                    section_text,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens
                )

                for j, sub_text in enumerate(sub_chunks):
                    # Prepend breadcrumb to every sub-chunk (item 2)
                    chunk_text = f"{breadcrumb}\n{sub_text}"
                    has_form_code_sub = bool(re.search(r'\b[A-Z]{2,3}-[A-Z]{1,3}-\d{1,3}\b', sub_text))
                    metadata = {
                        "source_type": source_type,
                        "chunk_type": chunk_type,
                        "document_name": filepath.name,
                        "section_title": section_title,
                        "section_index": i,
                        "sub_index": j,
                        "topic_tags": self._extract_topic_tags(sub_text),
                        "has_steps": self._has_numbered_steps(sub_text),
                        "has_forms": has_form_code_sub,
                        **doc_metadata
                    }

                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata=metadata,
                        char_count=len(chunk_text),
                        token_count=self._count_tokens(chunk_text),
                        chunk_id=self._generate_chunk_id(filepath, i, j)
                    ))
        
        return chunks
    
    def _split_by_headers(self, text: str, max_tokens: int = 2000) -> List[Dict[str, str]]:
        """
        Two-pass hierarchical header splitting.
        
        Pass 1: Split on major structural headers only
        (SECTION, CHAPTER, Annexure, numbered ALL-CAPS titles).
        
        Pass 2: For sections exceeding max_tokens, sub-split on
        subsection headers (X.Y, X.Y.Z).
        
        This prevents the fragmentation bug where subsection headers
        (4.1, 5.2.1) create micro-sections that kill their parent
        section content via the too_short filter.
        
        Args:
            text: Full document text
            max_tokens: Threshold for triggering Pass 2 sub-splitting
        
        Returns:
            List of {"title": str, "text": str} dicts
        """
        # ── Pass 1: Major structural headers ──
        major_patterns = [
            # SECTION N: TITLE with optional continuation line (ERB style: "SECTION 6: COMPENSATION &\n BENEFITS")
            r'\n\s*(SECTION\s+\d+\s*:\s*.+?(?:\n[ \t]*[A-Z &]+)?)',
            # CHAPTER N: TITLE (Guidelines style)
            r'\n\s*(CHAPTER\s+\d+\s*:\s*[^\n]+)',
            # Annexure A: Title
            r'\n\s*(Annexure\s+[A-Z]+\s*:\s*[^\n]+)',
            # N. ALL CAPS TITLE (Legal doc style: "4. WORKING OBLIGATIONS")
            r'\n(\d{1,2}\.\s{1,3}[A-Z][A-Z][A-Z\s/&()\-,]+)\s*\n',
            # N. Title Case Heading — C5: require blank line after to avoid matching list items
            r'\n\s*(\d{1,2}\.\s+[A-Z][a-zA-Z][A-Za-z\s&/()\-,]{2,68})\s*\n\s*\n',
            # Roman numeral headers: "I. Purpose", "II. Scope", "IX. Termination"
            r'\n\s*((?:I{1,3}|IV|VI{0,3}|IX|XI{0,3}|V|X)\.\s+[A-Z][A-Za-z\s&/()\-,]{2,68})\s*\n',
            # Bare ALL-CAPS heading on its own line (surrounded by blank lines to avoid matching mid-sentence caps)
            r'\n\s*\n([A-Z][A-Z][A-Z][A-Z\s&/\-]{2,56})\s*\n\s*\n',
            # Markdown-style heading: "## Eligibility", "### Required Documents"
            r'\n(#{1,3}\s+[^\n]{3,78})\s*\n',
        ]
        
        major_positions = self._find_header_matches(text, major_patterns)
        major_positions = self._deduplicate_header_positions(major_positions)
        
        if not major_positions:
            return [{"title": "", "text": text}]
        
        # Extract sections between major headers
        sections = []
        
        # Intro text before first header
        if major_positions[0][0] > 0:
            intro = text[:major_positions[0][0]].strip()
            if intro and self._count_tokens(intro) >= 10:
                sections.append({"title": "Introduction", "text": intro})
        
        for i, (start, end, title) in enumerate(major_positions):
            next_start = major_positions[i + 1][0] if i < len(major_positions) - 1 else len(text)
            section_text = text[end:next_start].strip()
            
            # Prepend title to text so the section header is part of the chunk content
            full_text = f"{title}\n{section_text}" if section_text else title
            sections.append({"title": title, "text": full_text.strip()})
        
        # ── Pass 2: Sub-split oversized sections ──
        sub_patterns = [
            r'\n(\d+\.\d+\.\d+\s+[A-Za-z][A-Za-z ]{2,})',   # X.Y.Z: "5.2.1 Earned Leave" (min 3 chars)
            r'\n(\d+\.\d+\s+[A-Za-z][^\n]{5,})',              # X.Y: "5.2 Types of Leave"
        ]
        
        final_sections = []
        for section in sections:
            token_count = self._count_tokens(section["text"])
            if token_count > max_tokens:
                sub_sections = self._sub_split_section(section, sub_patterns)
                final_sections.extend(sub_sections)
            else:
                final_sections.append(section)
        
        return final_sections
    
    def _find_header_matches(self, text: str, patterns: List[str]) -> List[Tuple[int, int, str]]:
        """Find all header matches across a list of regex patterns."""
        positions = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                title = match.group(1).strip()
                if len(title) < 4:
                    continue
                positions.append((match.start(), match.end(), title))
        
        positions.sort(key=lambda x: x[0])
        return positions
    
    def _deduplicate_header_positions(self, positions: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """
        Remove overlapping header matches.
        
        When two headers overlap (e.g., "SECTION 4:" at pos 100 and "4. WORKING"
        at pos 105), keep the more informative one (longer title).
        """
        if not positions:
            return positions
        
        deduped = [positions[0]]
        for pos in positions[1:]:
            prev = deduped[-1]
            # Overlapping if new match starts before previous ends (+ small gap)
            if pos[0] < prev[1] + 10:
                # Keep the one with the longer (more informative) title
                if len(pos[2]) > len(prev[2]):
                    deduped[-1] = pos
            else:
                deduped.append(pos)
        
        return deduped
    
    def _sub_split_section(
        self,
        section: Dict[str, str],
        sub_patterns: List[str]
    ) -> List[Dict[str, str]]:
        """
        Split an oversized section by subsection headers (X.Y, X.Y.Z).
        
        Preserves parent section title as prefix in sub-chunk titles.
        
        Args:
            section: {"title": str, "text": str} of the oversized section
            sub_patterns: Regex patterns for subsection headers
        
        Returns:
            List of sub-sections, or [section] if no subsections found
        """
        text = section["text"]
        parent_title = section["title"]
        
        sub_positions = self._find_header_matches(text, sub_patterns)
        sub_positions = self._deduplicate_header_positions(sub_positions)
        
        if not sub_positions:
            return [section]
        
        sub_sections = []
        
        # Intro text before first subsection header
        if sub_positions[0][0] > 0:
            intro_text = text[:sub_positions[0][0]].strip()
            if intro_text and self._count_tokens(intro_text) >= 10:
                sub_sections.append({
                    "title": parent_title,
                    "text": intro_text
                })
        
        for i, (start, end, sub_title) in enumerate(sub_positions):
            next_start = sub_positions[i + 1][0] if i < len(sub_positions) - 1 else len(text)
            sub_text = text[end:next_start].strip()
            
            full_text = f"{sub_title}\n{sub_text}" if sub_text else sub_title
            sub_sections.append({
                "title": f"{parent_title} > {sub_title}",
                "text": full_text.strip()
            })
        
        return sub_sections if sub_sections else [section]
    
    def _split_by_size(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int
    ) -> List[str]:
        """
        Split text by size with overlap, respecting semantic boundaries.
        
        Rules:
        - Never split mid-sentence (walk back to `. ` + capital)
        - Never split inside numbered lists
        - Keep lists whole even if oversized
        """
        # Group table rows first to keep them atomic
        text = self._group_table_rows(text)
        
        chunks = []
        
        # Detect if text contains numbered list
        has_numbered_list = bool(re.search(r'\n\s*\d+\.\s+', text))
        
        if has_numbered_list:
            # Check if entire text with list fits in limit
            text_tokens = self._count_tokens(text)
            if text_tokens <= max_tokens * 1.5:  # Allow 50% overflow for lists
                return [text]
            
            # Split before/after list, not inside
            list_start = re.search(r'\n\s*\d+\.\s+', text)
            if list_start:
                before_list = text[:list_start.start()].strip()
                list_and_after = text[list_start.start():].strip()
                
                # Try to keep list together
                if before_list and self._count_tokens(before_list) <= max_tokens:
                    chunks.append(before_list)
                    chunks.append(list_and_after)
                    return chunks
        
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
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Check if single paragraph exceeds limit
                if para_tokens > max_tokens:
                    # Split at sentence boundaries
                    para_chunks = self._split_at_sentence_boundary(para, max_tokens)
                    chunks.extend(para_chunks[:-1])  # Add all but last
                    current_chunk = para_chunks[-1] + "\n\n"  # Last becomes start of next
                    current_tokens = self._count_tokens(para_chunks[-1])
                else:
                    # Start new chunk with overlap
                    if chunks and overlap_tokens > 0:
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
    
    def _split_at_sentence_boundary(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text at sentence boundaries, skipping common abbreviations.

        C7: Rewritten without in-place list mutation — iterates parts into a
        new list, merging when the preceding word is an abbreviation.
        """
        _ABBREVS = re.compile(
            r'\b(?:Dr|Prof|Mr|Mrs|Ms|Miss|Sr|Jr|St|Lt|Capt|Col|Gen'
            r'|No|Vol|vs|etc|e\.g|i\.e|viz|approx|dept|govt|univ'
            r'|Ltd|Pvt|Inc|Corp|Co|Fig|Ref|Sec|Art|Cl|Para'
            r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
            r'|[A-Z])\.$',
            re.IGNORECASE,
        )

        # Split on ". " followed by a capital — produces alternating [text, sep, text, sep, ...]
        parts = re.split(r'(\.\s+)(?=[A-Z])', text)

        # Merge parts into sentences, re-joining false splits caused by abbreviations
        sentences: List[str] = []
        buf = ""
        i = 0
        while i < len(parts):
            part = parts[i]
            # Is the next element a separator?
            if i + 1 < len(parts) and re.match(r'^\.\s+$', parts[i + 1]):
                sep = parts[i + 1]
                last_word = part.rstrip().rsplit(None, 1)[-1] if part.strip() else ""
                if _ABBREVS.match(last_word + "."):
                    # Abbreviation — merge and continue without emitting
                    buf += part + sep
                    i += 2
                    continue
                else:
                    # Real sentence end — emit
                    sentences.append(buf + part + sep)
                    buf = ""
                    i += 2
                    continue
            # No separator follows — last fragment
            buf += part
            i += 1

        if buf.strip():
            sentences.append(buf)

        if not sentences:
            sentences = [text]

        chunks: List[str] = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]
    
    def _get_last_n_tokens(self, text: str, n_tokens: int) -> str:
        """Get approximately last n tokens from text, ending at a sentence boundary."""
        n_chars = n_tokens * 4
        if len(text) <= n_chars:
            return text
        candidate = text[-n_chars:]
        # Walk forward to the nearest sentence terminator so overlap starts clean
        match = re.search(r'[.!?]\s+', candidate)
        if match:
            return candidate[match.end():]
        return candidate
    
    def _group_table_rows(self, text: str) -> str:
        """Group consecutive table-like rows into single paragraphs.

        C4: Require 3+ space-run occurrences on the line (two columns + gap)
        OR the pattern must appear on 2+ consecutive lines before treating
        any of them as table rows — avoids false positives from PDF prose
        with spurious double-spaces.
        """
        lines = text.split('\n')
        result = []
        table_buffer = []

        # Named-header rows that are always table rows
        _HEADER_RE = re.compile(
            r'^(Parameter|Field|Grade|Designation|Location|Leave Type|Day'
            r'|Pay Component|Score Range|Action|Publication Type)\s',
        )
        # C4: require 3+ space runs (not just 2+)
        _TABLE_ROW_RE = re.compile(r'^[A-Za-z\d\(][^\n]*\s{3,}[^\s]')

        def _is_table(stripped: str) -> bool:
            if _HEADER_RE.match(stripped):
                return True
            return bool(_TABLE_ROW_RE.match(stripped)) and len(stripped) > 20

        for idx, line in enumerate(lines):
            stripped = line.strip()
            # C4: only mark as table if this line AND the next (or prev) also match
            prev_is_table = idx > 0 and _is_table(lines[idx - 1].strip())
            next_is_table = idx < len(lines) - 1 and _is_table(lines[idx + 1].strip())
            candidate = _is_table(stripped)

            if candidate and (prev_is_table or next_is_table or _HEADER_RE.match(stripped)):
                table_buffer.append(line)
            else:
                if table_buffer:
                    result.append('\n'.join(table_buffer))
                    table_buffer = []
                result.append(line)

        if table_buffer:
            result.append('\n'.join(table_buffer))

        return '\n\n'.join(result)
    
    def _extract_topic_tags(self, text: str) -> List[str]:
        """Extract topic keywords from text using word-boundary matching."""
        keyword_list = [
            # Core HR/policy
            "leave", "salary", "policy", "procedure", "application",
            "form", "faculty", "research", "publication", "award",
            "eligibility", "requirement", "deadline", "approval",
            "department", "legal", "compliance",
            # NMIMS-specific domain terms
            "sabbatical", "gratuity", "provident fund", "pf", "tds",
            "promotion", "appraisal", "teaching load", "consultancy",
            "attendance", "increment", "ltc", "medical reimbursement",
            "seed grant", "conference", "travel", "reimbursement",
            "noc", "deputation", "transfer", "resignation", "termination",
            "probation", "confirmation", "contract", "agreement",
            "phd", "research grant", "publication incentive",
            "workload", "timetable", "examination", "evaluation",
            "feedback", "mentoring", "counselling", "grievance",
            "maternity", "paternity", "casual leave", "earned leave",
            "medical leave", "duty leave", "special leave",
        ]
        # Use \b word boundaries to avoid substring collisions (e.g. "hr" in "whether")
        keywords = [
            kw for kw in keyword_list
            if re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE)
        ]
        return keywords[:15]
    
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
        """
        Estimate token count using tiktoken cl100k_base.

        More accurate than word×1.3 for hyphenated form codes (HR-LA-01),
        Indian names with initials, and compound institutional terms.
        Falls back to word×1.3 if tiktoken is unavailable.
        """
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            return int(len(text.split()) * 1.3)
    
    def _assign_chunk_levels(self, chunks: List[Chunk], full_text: str) -> List[Chunk]:
        """
        Assign chunk levels (overview/section/atomic) after type-specific chunking.
        
        Rules:
        - First chunk if it's short and contains document title/intro: overview
        - Chunks with definitions, single facts, deadlines: atomic
        - Everything else: section
        
        Args:
            chunks: List of chunks from type-specific chunker
            full_text: Original document text
        
        Returns:
            Chunks with chunk_level assigned
        """
        if not chunks:
            return chunks
        
        for i, chunk in enumerate(chunks):
            # Default to section
            level = "section"
            
            # First chunk: check if it's an overview
            if i == 0 and chunk.token_count < 400:
                # Check for overview indicators
                text_lower = chunk.text.lower()
                if any(indicator in text_lower for indicator in [
                    "document:", "applies to:", "overview", "summary", 
                    "introduction", "about this", "purpose"
                ]):
                    level = "overview"
            
            # Atomic chunks: short, single-purpose content
            if chunk.token_count < 200:
                text_lower = chunk.text.lower()
                # Check for atomic indicators
                if any(pattern in text_lower for pattern in [
                    "is defined as", "means", "refers to",  # Definitions
                    "deadline", "due date", "before", "within",  # Deadlines
                    "if", "when", "must", "shall", "required",  # Rules
                    "form name:", "form number:", "application"  # Forms
                ]):
                    level = "atomic"
            
            # Faculty profiles: always section (complete profiles)
            if chunk.metadata.get("source_type") == "faculty_profile":
                level = "section"
            
            chunk.chunk_level = level
        
        return chunks
    
    def should_skip_chunk(self, chunk: Chunk) -> Tuple[bool, Optional[str]]:
        """
        Quality filter: determine if chunk should be skipped.
        
        Returns: (should_skip, reason)
        """
        text_lower = chunk.text.lower()
        
        # Skip acknowledgement / signature pages boilerplate
        boilerplate_phrases = [
            "i, the undersigned, hereby acknowledge",
            "please return this signed page",
            "for hr department use only",
            "strictly confidential — for internal use only — do not distribute",
            "employee signature",
            "date of acknowledgement",
        ]
        
        if any(phrase in text_lower for phrase in boilerplate_phrases):
            return True, "boilerplate_signature_page"
        
        # Skip chunks where section title is a single short word (PDF footer artifact)
        # BUT: Don't skip if it's a valid section like "Mission", "Vision", etc.
        section_title = chunk.metadata.get("section_title", "").strip()
        if section_title and len(section_title.split()) <= 2:
            # Only skip if it's very short AND looks like a footer (city names, etc.)
            footer_artifacts = ["india", "mumbai", "delhi", "bangalore", "pune", "hyderabad"]
            if section_title.lower() in footer_artifacts and len(section_title) < 15:
                return True, "artifact_section_title"
        
        # Atomic fact exception — short but contains high-value patterns
        atomic_indicators = [
            r"deadline", r"due date", r"must be submitted",
            r"is defined as", r"means ", r"shall not exceed",
            r"maximum", r"minimum", r"within \d+ days",
            r"not less than", r"not more than", r"effective from"
        ]
        
        is_atomic = any(
            re.search(pattern, chunk.text, re.IGNORECASE)
            for pattern in atomic_indicators
        )
        
        # Allow short chunks only if they contain atomic fact patterns OR are faculty profiles
        if chunk.token_count < 25:
            # Faculty profiles should never be skipped for being too short
            if chunk.metadata.get("source_type") == "faculty_profile":
                return False, None
            if is_atomic and chunk.token_count >= 15:
                return False, None  # Keep atomic facts
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
        """Clean chunk text before embedding while preserving table structure."""
        # Strip PDF headers/footers with exact patterns
        header_patterns = [
            r'NMIMS\s*[—–-]\s*Employee Resource Book.*?CONFIDENTIAL\s*\n?',
            r'NMIMS\s*[—–-]\s*Faculty Academic Guidelines.*?CONFIDENTIAL\s*\n?',
            r'NMIMS\s*[—–-]\s*Faculty Applications Compendium.*?CONFIDENTIAL\s*\n?',
            r'Narsee Monjee Institute of Management Studies\s*\|\s*HR Department.*?\n',
            r'Narsee Monjee Institute of Management Studies\s*\|\s*Office of the Dean.*?\n',
            r'www\.nmims\.edu\s*\n?',
            r'STRICTLY CONFIDENTIAL\s*[—–-]\s*FOR INTERNAL USE ONLY.*?\n',
            r'CONFIDENTIAL\s*\n',
            r'Page\s+\d+\s+of\s+\d+',
            r'Page\s*\|\s*\d+',           # "Page | 12" style footer
            r'(?m)^\s*Page\s+\d+\s*$',    # plain "Page 12" on its own line only
            r'©\s*\d{4}\s+NMIMS',
            r'Document Version:.*?\n',
            r'Last Updated:.*?\n',
        ]

        for pattern in header_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        # De-duplicate repeated table-header rows like "Parameter Details"
        # that reappear every page when a table spans multiple pages.
        text = re.sub(
            r'(?:^|\n)\s*Parameter\s+Details\s*(?=\n(?:.*\n){0,1}.*?Parameter\s+Details)',
            '\n',
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r'(\n\s*Parameter\s+Details\s*\n)(\s*Parameter\s+Details\s*\n)+',
            r'\1',
            text,
            flags=re.IGNORECASE,
        )
        
        # Preserve table-like content — don't collapse lines with consistent structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Don't merge lines that look like table rows (Parameter + Value pattern)
            # Pattern: Word/phrase followed by 3+ spaces then value
            if re.match(r'^[A-Za-z][A-Za-z\s\/\(\)]+\s{3,}.+', line):
                cleaned_lines.append(line)  # preserve as-is
            else:
                # Clean individual line
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove repeated newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode common unicode issues
        text = text.replace('\xa0', ' ')
        text = text.replace('\u200b', '')
        
        return text.strip()
