"""
BM25 tokenizer with proper word segmentation and domain-specific stopwords.
"""

import re
from typing import List
import logging

try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class BM25Tokenizer:
    """
    Tokenizer for BM25 with domain-specific stopwords.
    
    Uses NLTK word_tokenize if available, falls back to regex-based tokenization.
    Removes domain-specific stopwords that appear in almost every chunk.
    """
    
    # Domain-specific stopwords (institutional terms with no retrieval signal)
    DOMAIN_STOPWORDS = {
        "nmims", "university", "college", "document", "section",
        "page", "chapter", "article", "clause", "paragraph",
        "institute", "school", "department", "faculty", "staff"
    }
    
    # Standard English stopwords (subset - most common)
    ENGLISH_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "has", "he", "in", "is", "it", "its", "of", "on",
        "that", "the", "to", "was", "will", "with"
    }
    
    ALL_STOPWORDS = DOMAIN_STOPWORDS | ENGLISH_STOPWORDS
    
    def __init__(self):
        """Initialize tokenizer."""
        self.logger = logging.getLogger(__name__)
        
        if NLTK_AVAILABLE:
            try:
                # Download punkt tokenizer if not available
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self.logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            self.logger.info("Using NLTK word_tokenize for BM25")
        else:
            self.logger.warning("NLTK not available, using regex tokenizer for BM25")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of tokens (lowercase, stopwords removed)
        """
        # Lowercase
        text = text.lower()
        
        # Tokenize
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            # Fallback: regex-based tokenization
            tokens = re.findall(r'\b\w+\b', text)
        
        # Filter stopwords and short tokens
        tokens = [
            token for token in tokens
            if token not in self.ALL_STOPWORDS and len(token) > 2
        ]
        
        return tokens
