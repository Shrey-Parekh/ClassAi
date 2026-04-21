"""
Chunking configuration based on semantic completeness principles.
"""

from enum import Enum
from typing import Dict, Any


class ChunkLevel(Enum):
    """Three-level chunking hierarchy."""
    OVERVIEW = "level_1_overview"  # Document summary
    PROCEDURE = "level_2_procedure"  # Complete process/policy
    ATOMIC = "level_3_atomic"  # Single fact/rule


class ContentType(Enum):
    """Faculty document content types."""
    PROCEDURE = "procedure"
    RULE = "rule"
    FORM = "form"
    CIRCULAR = "circular"
    POLICY = "policy"
    DEFINITION = "definition"
    DEADLINE = "deadline"
    IMAGE = "image"
    DIAGRAM = "diagram"


class IntentType(Enum):
    """Query intent classification for retrieval routing."""
    LOOKUP = "lookup"  # "what is X"
    PROCEDURE = "procedure"  # "how do I X"
    ELIGIBILITY = "eligibility"  # "can I X"
    FORM_HELP = "form_help"  # "help me fill X"
    GENERAL = "general"  # Broad/unclear


# Chunking constraints
MAX_LEVEL2_TOKENS = 980  # Increased for procedures (keep complete processes together)
OVERLAP_TOKENS = 50  # Only for Level 2 chunks

# Retrieval configuration
TOP_K_INITIAL = 20  # Initial hybrid search results
TOP_K_RERANKED = 15  # After cross-encoder reranking (increased from 5)

# Intent-based chunk limits for LLM context
# These determine how many chunks to send to the LLM based on query intent
# With 16K context window, we can use ~8K for chunks (rest for prompt/response)
INTENT_CHUNK_LIMITS = {
    "lookup": 15,          # R15: raised from 8 — profile chunks are short
    "person_lookup": 15,   # Alias for lookup
    "department_list": 30, # Entire department listing
    "topic_search": 25,    # Broad topic coverage
    "procedure": 20,       # Policy + forms + guidelines + examples
    "eligibility": 15,     # Policy sections with conditions
    "salary_benefits": 15, # Salary/benefits sections
    "general": 20,         # Broad context for general queries
}

# Default chunk limit if intent not in map
DEFAULT_CHUNK_LIMIT = 20

# Intent-to-level routing
INTENT_TO_CHUNK_LEVELS: Dict[IntentType, list[ChunkLevel]] = {
    IntentType.LOOKUP: [ChunkLevel.ATOMIC, ChunkLevel.PROCEDURE],
    IntentType.PROCEDURE: [ChunkLevel.PROCEDURE, ChunkLevel.OVERVIEW],
    IntentType.ELIGIBILITY: [ChunkLevel.PROCEDURE, ChunkLevel.ATOMIC],
    IntentType.FORM_HELP: [ChunkLevel.PROCEDURE, ChunkLevel.ATOMIC],
    IntentType.GENERAL: [ChunkLevel.OVERVIEW, ChunkLevel.PROCEDURE],
}

# Never split these patterns
ATOMIC_PATTERNS = [
    "if.*then",  # Conditional rules
    "step [0-9]+",  # Procedure steps
    "form.*field",  # Form instructions
    "deadline.*date",  # Time-sensitive info
]
