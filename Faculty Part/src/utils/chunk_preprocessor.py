"""
Chunk pre-processor for embedding safety and quality.

Handles normalization, validation, and splitting before embedding.
"""

import re
import unicodedata
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class PreprocessedChunk:
    """Result of chunk preprocessing."""
    text: str
    is_valid: bool
    discard_reason: str = None  # "empty", "too_short", None if valid
    was_split: bool = False
    split_index: int = None


class ChunkPreprocessor:
    """
    Pre-processes chunks before embedding.
    
    Handles:
    - Normalization (whitespace, encoding)
    - Special character replacement
    - Validity checking
    - Oversized chunk splitting
    """
    
    # Token to character ratio (approximate)
    TOKENS_PER_CHAR = 0.25  # 1 token ≈ 4 characters
    MIN_TOKENS = 5  # Reduced to capture short but important information (form names, keywords)
    MAX_TOKENS = 490  # Optimized for reranker safety (490 + 150 query = 640, safe margin)
    
    MIN_CHARS = int(MIN_TOKENS / TOKENS_PER_CHAR)  # ~20 chars
    MAX_CHARS = int(MAX_TOKENS / TOKENS_PER_CHAR)  # ~1960 chars
    
    # Character replacements
    CHAR_REPLACEMENTS = {
        '"': '"',  # Left double quote
        '"': '"',  # Right double quote
        ''': "'",  # Left single quote
        ''': "'",  # Right single quote
        '—': '-',  # Em dash
        '–': '-',  # En dash
        '…': '...',  # Ellipsis
        '₹': 'Rs.',  # Rupee symbol
        '§': 'Section',  # Section symbol
        '•': '-',  # Bullet
        '◦': '-',  # Circle bullet
        '▪': '-',  # Square bullet
        '‣': '-',  # Triangle bullet
        '\xa0': ' ',  # Non-breaking space
    }
    
    def preprocess(self, chunk_text: str) -> List[PreprocessedChunk]:
        """
        Pre-process a chunk and return list of valid chunks.
        
        May return multiple chunks if original was split.
        
        Args:
            chunk_text: Raw chunk text
        
        Returns:
            List of PreprocessedChunk objects
        """
        # Step 1: Strip and normalize
        normalized = self._strip_and_normalize(chunk_text)
        
        # Step 2: UTF-8 encoding safety
        safe_utf8 = self._ensure_utf8_safe(normalized)
        
        # Step 3: Special character normalization
        cleaned = self._normalize_special_chars(safe_utf8)
        
        # Step 4: Validity check and splitting
        results = self._validate_and_split(cleaned)
        
        return results
    
    def _strip_and_normalize(self, text: str) -> str:
        """
        Strip whitespace and collapse multiple spaces/newlines.
        Remove control characters (ASCII < 32 except newline).
        Protect tables from being collapsed.
        """
        # Protect tables first
        text = self._protect_tables(text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Remove control characters except newline
        text = ''.join(
            char for char in text
            if ord(char) >= 32 or char == '\n'
        )
        
        # Collapse multiple consecutive whitespace/newlines into single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_table_row(self, line: str) -> bool:
        """
        Detect if a line is part of a table.
        
        Table rows typically have:
        - Multiple pipe separators (|)
        - Multiple tab separators
        - Repeated whitespace blocks (PDF table extraction pattern)
        """
        return (
            line.count('|') >= 2 or
            line.count('\t') >= 2 or
            # Repeated whitespace blocks (PDF table extraction pattern)
            len(re.findall(r'\s{3,}', line)) >= 2
        )
    
    def _protect_tables(self, text: str) -> str:
        """
        Keep table rows together — don't split mid-table.
        
        Detects table blocks and preserves their structure by
        keeping consecutive table rows together.
        """
        lines = text.split('\n')
        result = []
        in_table = False
        table_buffer = []
        
        for line in lines:
            if self._is_table_row(line):
                in_table = True
                table_buffer.append(line)
            else:
                if in_table and table_buffer:
                    # End of table - add as single block
                    result.append('\n'.join(table_buffer))
                    table_buffer = []
                    in_table = False
                result.append(line)
        
        # Handle table at end of text
        if table_buffer:
            result.append('\n'.join(table_buffer))
        
        return '\n'.join(result)
    
    def _ensure_utf8_safe(self, text: str) -> str:
        """
        Encode to UTF-8 and remove any characters that fail encoding.
        """
        try:
            # Encode to UTF-8
            encoded = text.encode('utf-8')
            # Decode back - this will fail if there are invalid sequences
            decoded = encoded.decode('utf-8')
            return decoded
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Remove problematic characters
            # Use NFKD normalization to decompose characters
            normalized = unicodedata.normalize('NFKD', text)
            # Keep only ASCII-safe characters
            safe = normalized.encode('ascii', errors='ignore').decode('ascii')
            return safe
    
    def _normalize_special_chars(self, text: str) -> str:
        """Replace special characters with plain-text equivalents."""
        for special, replacement in self.CHAR_REPLACEMENTS.items():
            text = text.replace(special, replacement)
        return text
    
    def _validate_and_split(self, text: str) -> List[PreprocessedChunk]:
        """
        Validate chunk and split if necessary.
        
        Returns list of valid chunks or discard reasons.
        """
        # Check if empty
        if not text or len(text.strip()) == 0:
            return [PreprocessedChunk(
                text="",
                is_valid=False,
                discard_reason="empty"
            )]
        
        # Check if too short
        if len(text) < self.MIN_CHARS:
            return [PreprocessedChunk(
                text=text,
                is_valid=False,
                discard_reason="too_short"
            )]
        
        # Check if too long - need to split
        if len(text) > self.MAX_CHARS:
            return self._split_oversized_chunk(text)
        
        # Valid chunk
        return [PreprocessedChunk(
            text=text,
            is_valid=True,
            discard_reason=None
        )]
    
    def _split_oversized_chunk(self, text: str) -> List[PreprocessedChunk]:
        """
        Split oversized chunk at sentence boundaries with overlap.
        
        Each sub-chunk inherits metadata from parent and includes overlap
        from previous chunk for semantic continuity.
        """
        results = []
        split_index = 0
        
        # Split by sentence boundaries (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Overlap: keep last 2 sentences from previous chunk
        OVERLAP_SENTENCES = 2
        
        current_chunk = ""
        overlap_buffer = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.MAX_CHARS:
                current_chunk = test_chunk
                overlap_buffer.append(sentence)
                if len(overlap_buffer) > OVERLAP_SENTENCES:
                    overlap_buffer.pop(0)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    results.append(PreprocessedChunk(
                        text=current_chunk.strip(),
                        is_valid=True,
                        discard_reason=None,
                        was_split=True,
                        split_index=split_index
                    ))
                    split_index += 1
                
                # Start new chunk with overlap from previous
                if overlap_buffer and split_index > 0:
                    current_chunk = " ".join(overlap_buffer) + " " + sentence
                else:
                    current_chunk = sentence
                
                # Reset overlap buffer
                overlap_buffer = [sentence]
        
        # Add final chunk
        if current_chunk:
            results.append(PreprocessedChunk(
                text=current_chunk.strip(),
                is_valid=True,
                discard_reason=None,
                was_split=True,
                split_index=split_index
            ))
        
        return results if results else [PreprocessedChunk(
            text=text,
            is_valid=False,
            discard_reason="split_failed"
        )]
