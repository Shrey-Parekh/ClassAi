"""
Intent classification for routing queries to appropriate chunk levels.
"""

from typing import List, Dict, Any
from config.chunking_config import IntentType, ChunkLevel, INTENT_TO_CHUNK_LEVELS


class IntentClassifier:
    """
    Classifies faculty queries into intent types to optimize retrieval.
    
    Intent types:
    - LOOKUP: Direct factual questions
    - PROCEDURE: How-to questions
    - ELIGIBILITY: Permission/qualification questions
    - FORM_HELP: Form filling assistance
    - GENERAL: Broad exploratory questions
    """
    
    def __init__(self, llm_client=None):
        """Initialize with optional LLM for complex classification."""
        self.llm_client = llm_client
        
        # Keyword patterns for rule-based classification
        self.intent_patterns = {
            IntentType.LOOKUP: [
                "what is", "what does", "define", "meaning of",
                "how many", "how much", "what are the limits"
            ],
            IntentType.PROCEDURE: [
                "how do i", "how to", "steps for", "process for",
                "apply for", "submit", "procedure"
            ],
            IntentType.ELIGIBILITY: [
                "can i", "am i allowed", "eligible for", "qualify for",
                "permitted to", "who can"
            ],
            IntentType.FORM_HELP: [
                "fill form", "help with form", "what to write",
                "form field", "application form"
            ],
        }
    
    def classify(self, query: str) -> IntentType:
        """
        Classify query intent using rule-based patterns.
        
        For production: enhance with LLM-based classification.
        """
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        # Default to GENERAL for unclear queries
        return IntentType.GENERAL
    
    def get_target_levels(self, intent: IntentType) -> List[ChunkLevel]:
        """
        Get prioritized chunk levels for a given intent.
        
        Returns levels in order of priority for retrieval.
        """
        return INTENT_TO_CHUNK_LEVELS.get(intent, [ChunkLevel.PROCEDURE])
    
    def get_metadata_filters(self, query: str, intent: IntentType) -> Dict[str, Any]:
        """
        Generate metadata filters based on query and intent.
        
        Examples:
        - Filter by applicability scope (teaching staff vs admin)
        - Filter by document date (exclude superseded documents)
        - Filter by content type
        """
        filters = {}
        
        # Extract applicability from query
        if "teaching" in query.lower() or "professor" in query.lower():
            filters["applies_to"] = "teaching_staff"
        elif "admin" in query.lower():
            filters["applies_to"] = "administrative_staff"
        
        # Exclude superseded documents by default
        filters["superseded_by"] = None
        
        return filters
