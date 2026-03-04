"""
Enhanced query understanding with intent, domain, and entity detection.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import re


@dataclass
class QueryUnderstanding:
    """Structured query understanding result."""
    intent: str  # lookup, procedure, eligibility, general
    domain: str  # faculty_info, policies, procedures, general
    entities: List[str]  # Extracted entities (names, departments, etc.)
    is_current_only: bool  # Whether to filter for current/active documents
    metadata_filters: Dict[str, Any]  # Filters to apply
    expanded_query: str  # Query with added terms for better retrieval


class QueryAnalyzer:
    """
    Analyzes queries to extract intent, domain, and entities.
    
    This enables better retrieval through metadata pre-filtering.
    """
    
    def __init__(self):
        """Initialize query analyzer with patterns."""
        
        # Intent patterns
        self.intent_patterns = {
            "lookup": [
                r"\bwho\s+(is|are|teaches|works|researches)\b",
                r"\bwhat\s+(is|are|does)\b",
                r"\bshow\s+me\b",
                r"\bfind\b",
                r"\blist\b",
                r"\btell\s+me\s+about\b",
                r"\bsummari[sz]e\b",
                r"\bexplain\b",
                r"\bdescribe\b"
            ],
            "procedure": [
                r"\bhow\s+(do|to|can)\b",
                r"\bsteps\s+(for|to)\b",
                r"\bprocess\s+(for|to)\b",
                r"\bprocedure\b",
                r"\bapply\s+for\b"
            ],
            "eligibility": [
                r"\bcan\s+i\b",
                r"\bam\s+i\s+(allowed|eligible)\b",
                r"\bwho\s+can\b",
                r"\bqualify\s+for\b",
                r"\bpermitted\s+to\b"
            ]
        }
        
        # Domain patterns
        self.domain_patterns = {
            "faculty_info": [
                r"\bfaculty\b",
                r"\bprofessor\b",
                r"\bdr\.\s*\w+",
                r"\bresearch\b",
                r"\bpublication\b",
                r"\bteach(es|ing)?\b",
                r"\bdepartment\b"
            ],
            "policies": [
                r"\bpolicy\b",
                r"\bpolicies\b",
                r"\brule\b",
                r"\bregulation\b",
                r"\bguideline\b",
                r"\bcode\s+of\s+conduct\b"
            ],
            "procedures": [
                r"\bprocedure\b",
                r"\bprocess\b",
                r"\bform\b",
                r"\bapplication\b",
                r"\bsubmit\b",
                r"\bapproval\b"
            ]
        }
        
        # Common faculty/department names (can be expanded)
        self.entity_patterns = {
            "person": r"\bdr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
            "department": r"\b(computer\s+science|CS|mechanical|electrical|civil|AI|ML|data\s+science)\b",
            "topic": r"\b(AI|ML|machine\s+learning|blockchain|IoT|renewable\s+energy|solar|finance)\b"
        }
    
    def analyze(self, query: str) -> QueryUnderstanding:
        """
        Analyze query to extract intent, domain, and entities.
        
        Args:
            query: User query text
        
        Returns:
            QueryUnderstanding with extracted information
        """
        query_lower = query.lower()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Detect domain
        domain = self._detect_domain(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine if current documents only
        is_current_only = self._is_current_only(query_lower)
        
        # Build metadata filters
        metadata_filters = self._build_metadata_filters(
            domain, is_current_only, entities
        )
        
        # Expand query for better retrieval (especially for procedures)
        expanded_query = self._expand_query(query, intent)
        
        return QueryUnderstanding(
            intent=intent,
            domain=domain,
            entities=entities,
            is_current_only=is_current_only,
            metadata_filters=metadata_filters,
            expanded_query=expanded_query
        )
    
    def _expand_query(self, query: str, intent: str) -> str:
        """
        Expand query with related terms for better retrieval.
        
        Args:
            query: Original query
            intent: Detected intent
        
        Returns:
            Expanded query string
        """
        # For procedure queries, add common procedural terms
        if intent == "procedure":
            expansion_terms = ["steps", "process", "documents", "approval", "requirements", "application"]
            return f"{query} {' '.join(expansion_terms)}"
        
        # For eligibility queries, add eligibility-related terms
        elif intent == "eligibility":
            expansion_terms = ["criteria", "requirements", "eligible", "qualify"]
            return f"{query} {' '.join(expansion_terms)}"
        
        # For lookup queries, keep original
        return query
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent using pattern matching."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        return "general"
    
    def _detect_domain(self, query: str) -> str:
        """Detect query domain using pattern matching."""
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            domain_scores[domain] = score
        
        # Return domain with highest score, or general if no matches
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return "general"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates
    
    def _is_current_only(self, query: str) -> bool:
        """Determine if query should filter for current documents only."""
        current_keywords = [
            r"\bcurrent\b",
            r"\blatest\b",
            r"\brecent\b",
            r"\bnew\b",
            r"\b202[4-9]\b",  # Years 2024+
            r"\bthis\s+year\b"
        ]
        
        for keyword in current_keywords:
            if re.search(keyword, query, re.IGNORECASE):
                return True
        
        return False
    
    def _build_metadata_filters(
        self,
        domain: str,
        is_current_only: bool,
        entities: List[str]
    ) -> Dict[str, Any]:
        """Build metadata filters based on query understanding."""
        filters = {}
        
        # Domain-based filtering
        if domain != "general":
            filters["domain"] = domain
        
        # Current documents only
        if is_current_only:
            filters["is_current"] = True
            filters["superseded_by"] = None
        
        # Entity-based filtering (if applicable)
        # This can be expanded based on your metadata structure
        
        return filters
