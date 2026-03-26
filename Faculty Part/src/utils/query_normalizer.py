"""
Centralized query normalization utility.

Single source of truth for title stripping, whitespace normalization, and text cleaning.
Called at every point in the pipeline where query text is processed.
"""

import re
from typing import List


class QueryNormalizer:
    """
    Centralized query normalization.
    
    Ensures consistent text processing across:
    - Query embedding
    - BM25 search
    - Name matching
    - Entity extraction
    """
    
    # Titles to strip (order matters - longer phrases first)
    TITLES = [
        "associate professor",
        "assistant professor",
        "professor",
        "prof.",
        "prof",
        "doctor",
        "dr.",
        "dr",
        "mrs.",
        "mrs",
        "mr.",
        "mr",
        "ms.",
        "ms",
        "sir",
        "ma'am",
        "madam"
    ]
    
    @staticmethod
    def normalize_query(text: str, strip_titles: bool = True, lowercase: bool = True) -> str:
        """
        Normalize query text with consistent rules.
        
        Args:
            text: Raw query text
            strip_titles: Whether to remove titles (default: True)
            lowercase: Whether to convert to lowercase (default: True)
        
        Returns:
            Normalized text
        """
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Normalize whitespace (collapse multiple spaces)
        text = ' '.join(text.split())
        
        # Convert to lowercase if requested
        if lowercase:
            text = text.lower()
        
        # Strip titles if requested
        if strip_titles:
            for title in QueryNormalizer.TITLES:
                # Remove title with word boundaries
                text = re.sub(r'\b' + re.escape(title) + r'\b', '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace after title removal
        text = ' '.join(text.split())
        
        # Remove parentheses content (e.g., "(Shrivastava)")
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Clean up again
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def extract_name_parts(name: str) -> List[str]:
        """
        Extract name parts for matching.
        
        Args:
            name: Full name (titles already stripped)
        
        Returns:
            List of name parts (first, last, full)
        """
        parts = name.split()
        variations = []
        
        if len(parts) >= 2:
            variations.append(parts[0])  # First name
            variations.append(parts[-1])  # Last name
            if len(parts) > 2:
                variations.append(f"{parts[0]} {parts[-1]}")  # First + Last
        
        if name:
            variations.append(name)  # Full name
        
        return variations
