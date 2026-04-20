"""
Enhanced query understanding with intent, domain, and entity detection.
"""

from typing import Dict, Any, List, Optional
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
        """Initialize query analyzer with comprehensive patterns."""
        
        # Intent patterns - expanded with synonyms and variations
        self.intent_patterns = {
            "lookup": [
                # Who/What questions
                r"\bwho\s+(is|are|was|were|teaches|works|researches|published|wrote)\b",
                r"\bwhat\s+(is|are|was|were|does|did)\b",
                r"\bwhich\s+(faculty|professor|prof|dr|person|people)\b",
                
                # Information requests
                r"\bshow\s+me\b",
                r"\bfind\b",
                r"\blist\b",
                r"\btell\s+me\s+(about|regarding)\b",
                r"\bgive\s+me\s+(information|details|info)\b",
                r"\bprovide\s+(information|details|info)\b",
                
                # Summary/explanation requests
                r"\bsummari[sz]e\b",
                r"\bexplain\b",
                r"\bdescribe\b",
                r"\bdetail\b",
                r"\bwhat\s+are\s+the\b",
                r"\bwhat\s+is\s+the\b",
                
                # Faculty-specific
                r"\bprofile\s+of\b",
                r"\bbackground\s+of\b",
                r"\bexpertise\s+of\b",
                r"\bqualifications?\s+of\b",
                r"\bresearch\s+(interests?|areas?|focus|work)\b",
                r"\bpublications?\s+(of|by)\b",
                r"\bpapers?\s+(of|by|published)\b",
                r"\bawards?\s+(of|received|won)\b"
            ],
            "procedure": [
                # How-to questions
                r"\bhow\s+(do|to|can|should|would)\b",
                r"\bsteps\s+(for|to|required)\b",
                r"\bprocess\s+(for|to|of)\b",
                r"\bprocedure\s+(for|to|of)\b",
                
                # Application/submission
                r"\bapply\s+for\b",
                r"\bsubmit\b",
                r"\bfile\s+(for|an?)\b",
                r"\brequest\s+(for|an?)\b",
                
                # Guidance requests
                r"\bguide\s+me\b",
                r"\bwalk\s+me\s+through\b",
                r"\binstruct(ions?)?\b",
                r"\bwhat\s+(is|are)\s+the\s+(steps|process|procedure)\b"
            ],
            "eligibility": [
                # Permission/ability questions
                r"\bcan\s+i\b",
                r"\bmay\s+i\b",
                r"\bam\s+i\s+(allowed|eligible|qualified|permitted)\b",
                r"\bwho\s+can\b",
                r"\bwho\s+is\s+(allowed|eligible|qualified|permitted)\b",
                
                # Qualification questions
                r"\bqualify\s+for\b",
                r"\beligib(le|ility)\b",
                r"\brequirements?\s+(for|to)\b",
                r"\bcriteria\s+(for|to)\b",
                r"\bdo\s+i\s+(need|require|qualify)\b"
            ]
        }
        
        # Domain patterns - comprehensive coverage
        self.domain_patterns = {
            "faculty_info": [
                # Faculty titles
                r"\bfaculty\b",
                r"\bprofessor\b",
                r"\bprof\b",
                r"\bdr\b",
                r"\bdoctor\b",
                r"\bteacher\b",
                r"\binstructor\b",
                r"\blecturer\b",
                
                # Academic activities
                r"\bresearch(er|ing|es)?\b",
                r"\bpublication(s)?\b",
                r"\bpublish(ed|ing|es)?\b",
                r"\bpaper(s)?\b",
                r"\bjournal(s)?\b",
                r"\bconference(s)?\b",
                r"\bteach(es|ing)?\b",
                r"\bcourse(s)?\b",
                
                # Academic attributes
                r"\bdepartment\b",
                r"\bexpertise\b",
                r"\bspeciali[sz]ation\b",
                r"\bqualification(s)?\b",
                r"\bexperience\b",
                r"\baward(s)?\b",
                r"\bachievement(s)?\b",
                r"\brecognition\b",
                
                # Research terms
                r"\bcitation(s)?\b",
                r"\bh-index\b",
                r"\bimpact\s+factor\b",
                r"\bscopus\b",
                r"\bweb\s+of\s+science\b"
            ],
            "policies": [
                # Policy documents
                r"\bpolicy\b",
                r"\bpolicies\b",
                r"\brule(s)?\b",
                r"\bregulation(s)?\b",
                r"\bguideline(s)?\b",
                r"\bcode\s+of\s+conduct\b",
                
                # Leave/HR policies
                r"\bleave\b",
                r"\bholiday(s)?\b",
                r"\bvacation\b",
                r"\babsence\b",
                r"\battendance\b",
                
                # Employment terms
                r"\bemployment\b",
                r"\bcontract\b",
                r"\bagreement\b",
                r"\bterms\s+(and|&)\s+conditions\b",
                r"\bsalary\b",
                r"\bcompensation\b",
                r"\bbenefits?\b"
            ],
            "procedures": [
                # Process-related
                r"\bprocedure(s)?\b",
                r"\bprocess(es)?\b",
                r"\bstep(s)?\b",
                r"\bworkflow\b",
                
                # Forms and applications
                r"\bform(s)?\b",
                r"\bapplication(s)?\b",
                r"\brequest(s)?\b",
                r"\bsubmission(s)?\b",
                r"\bsubmit\b",
                
                # Approval process
                r"\bapproval(s)?\b",
                r"\bapprove(d)?\b",
                r"\bauthori[sz]ation\b",
                r"\bclearance\b",
                
                # Grants and funding
                r"\bgrant(s)?\b",
                r"\bfunding\b",
                r"\bscholarship(s)?\b",
                r"\bfellowship(s)?\b"
            ]
        }
        
        # Enhanced entity patterns
        self.entity_patterns = {
            # Person names with various titles
            "person": r"\b(?:dr\.?|prof\.?|professor|mr\.?|ms\.?|mrs\.?|miss)\s+[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)*",
            
            # Departments and schools
            "department": r"\b(computer\s+science|CS|information\s+technology|IT|mechanical|electrical|electronics|civil|chemical|AI|ML|data\s+science|mathematics|physics|chemistry|biology|management|business|finance|economics|humanities)\b",
            
            # Research topics
            "topic": r"\b(AI|ML|machine\s+learning|deep\s+learning|neural\s+networks?|blockchain|IoT|internet\s+of\s+things|renewable\s+energy|solar|wind|thermal|finance|economics|marketing|supply\s+chain|operations?|robotics|automation|nanotechnology|biotechnology)\b",
            
            # Form / policy codes — e.g. "HR-LA-01", "Form CL-7", "FAC-04", "ERB 2024-25"
            # Matches optional "form"/"application" prefix, then a dash-separated
            # code with 1-4 letter segments and a trailing number.
            "form": r"\b(?:form|application)?\s*[A-Z]{1,4}(?:-[A-Z]{1,4}){0,2}-?\d{1,3}\b",
            
            # Document types
            "document": r"\b(policy|guideline|handbook|manual|compendium|agreement|contract)\b"
        }
        
        # Synonym mapping for query expansion
        self.synonyms = {
            "faculty": ["professor", "prof", "teacher", "instructor", "lecturer", "dr", "doctor"],
            "research": ["study", "investigation", "work", "project", "publication", "paper"],
            "publication": ["paper", "article", "journal", "conference", "book", "chapter"],
            "award": ["recognition", "achievement", "honor", "prize", "accolade"],
            "expertise": ["specialization", "focus", "area", "interest", "domain"],
            "procedure": ["process", "steps", "workflow", "method", "approach"],
            "apply": ["submit", "file", "request", "register"],
            "requirement": ["criteria", "prerequisite", "condition", "qualification"]
        }
    
    def analyze(self, query: str) -> QueryUnderstanding:
        """
        Analyze query to extract intent, domain, and entities.
        
        Handles typos, variations, and imprecise queries gracefully.
        
        Args:
            query: User query text
        
        Returns:
            QueryUnderstanding with extracted information
        """
        # Detect intent on ORIGINAL query BEFORE preprocessing (preserves signal words)
        intent = self._detect_intent(query.lower())
        
        # THEN preprocess query (clean, normalize)
        query_preprocessed = self._preprocess_query(query)
        
        # Normalize query (handle variations)
        query_normalized = self._normalize_query(query_preprocessed)
        query_lower = query_normalized.lower()
        
        # Check for document-scoped broad queries first
        doc_scope_result = self._detect_document_scope_query(query_lower)
        if doc_scope_result:
            return doc_scope_result
        
        # Detect domain with weighted scoring
        domain = self._detect_domain(query_lower)
        
        # Extract and normalize entities
        entities = self._extract_entities(query_normalized)
        
        # Determine if current documents only
        is_current_only = self._is_current_only(query_lower)
        
        # Build metadata filters
        metadata_filters = self._build_metadata_filters(
            domain, is_current_only, entities
        )
        
        # Expand query with synonyms and context
        expanded_query = self._expand_query(query_normalized, intent, entities)
        
        return QueryUnderstanding(
            intent=intent,
            domain=domain,
            entities=entities,
            is_current_only=is_current_only,
            metadata_filters=metadata_filters,
            expanded_query=expanded_query
        )
    
    def _detect_document_scope_query(self, query: str) -> Optional[QueryUnderstanding]:
        """
        Detect broad queries asking about entire documents.
        
        These queries need special handling:
        - Force metadata filter to specific document
        - Increase top_k to get more chunks
        - Change intent to document_overview
        
        Returns QueryUnderstanding if matched, None otherwise.
        """
        # Document-scoped broad query patterns
        document_patterns = [
            # Long-form names with intent words
            (r'(employee resource book|ERB)\s*(rules|policies|guidelines|content|information)',
             'NMIMS_Employee_Resource_Book_2024-25.pdf'),
            (r'(faculty academic guidelines|FAG)\s*(rules|policies|content|information)',
             'NMIMS_Faculty_Academic_Guidelines.pdf'),
            (r'(faculty applications compendium|FAC)\s*(rules|policies|content|information|forms)',
             'NMIMS_Faculty_Applications_Compendium.pdf'),
            (r'tell me (about|the)\s*(resource book|employee book|ERB)',
             'NMIMS_Employee_Resource_Book_2024-25.pdf'),
            (r'tell me (about|the)\s*(academic guidelines|FAG)',
             'NMIMS_Faculty_Academic_Guidelines.pdf'),
            (r'tell me (about|the)\s*(applications compendium|forms compendium|FAC)',
             'NMIMS_Faculty_Applications_Compendium.pdf'),
            (r'what (is|are) (in|the)\s*(resource book|employee book|ERB)',
             'NMIMS_Employee_Resource_Book_2024-25.pdf'),
            (r'what (is|are) (in|the)\s*(academic guidelines|FAG)',
             'NMIMS_Faculty_Academic_Guidelines.pdf'),
            (r'what (is|are) (in|the)\s*(applications compendium|FAC)',
             'NMIMS_Faculty_Applications_Compendium.pdf'),
            # Short-form abbreviations used in isolation (as whole words)
            (r'(?<![A-Za-z])ERB(?![A-Za-z])',
             'NMIMS_Employee_Resource_Book_2024-25.pdf'),
            (r'(?<![A-Za-z])FAG(?![A-Za-z])',
             'NMIMS_Faculty_Academic_Guidelines.pdf'),
            (r'(?<![A-Za-z])FAC(?![A-Za-z])',
             'NMIMS_Faculty_Applications_Compendium.pdf'),
        ]
        
        for pattern, doc_name in document_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryUnderstanding(
                    intent="document_overview",
                    domain="policies",
                    entities=[],
                    is_current_only=False,
                    metadata_filters={"document_name": doc_name},
                    expanded_query=query + " policy guidelines rules procedures sections"
                )
        
        return None
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query to clean and normalize before analysis.
        
        Removes filler words, handles special characters.
        Does NOT strip titles here - that's done in _strip_titles_for_embedding.
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove common filler words at start
        filler_patterns = [
            r'^who\s+is\s+',
            r'^what\s+is\s+',
            r'^tell\s+me\s+about\s+',
            r'^give\s+me\s+info\s+about\s+',
            r'^show\s+me\s+',
            r'^find\s+',
        ]
        
        original_query = query
        for pattern in filler_patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        # If we removed everything, keep original
        if not query.strip():
            query = original_query
        
        # Handle parentheses in names (keep them for now, will be handled in entity extraction)
        # Example: "Pragati Khare (Shrivastava)" stays as is
        
        # Remove multiple spaces
        query = ' '.join(query.split())
        
        return query.strip()
    
    def _strip_titles_for_embedding(self, text: str) -> str:
        """
        Strip all titles from text before embedding or SPLADE.
        
        This ensures clean name matching without title noise.
        Applied before: query embedding, name embedding, SPLADE expansion.
        NOT applied to display query shown to user.
        """
        text_lower = text.lower()
        
        # Titles to strip (order matters - longer phrases first)
        titles = [
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
        
        for title in titles:
            # Remove title with word boundaries
            text_lower = re.sub(r'\b' + re.escape(title) + r'\b', '', text_lower)
        
        # Clean up extra whitespace
        text_lower = ' '.join(text_lower.split())
        
        return text_lower.strip()
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query to handle common variations and typos.
        
        Makes the system more forgiving of imprecise queries.
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Fix common typos and variations
        replacements = {
            # Title variations
            r'\bprof\s+': 'professor ',
            r'\bprof\.\s*': 'professor ',
            r'\bdr\s+': 'dr. ',
            r'\bdoctor\s+': 'dr. ',
            
            # Common misspellings
            r'\bfaculty\s*member\b': 'faculty',
            r'\bteaching\s*staff\b': 'faculty',
            r'\bprofessors?\b': 'professor',
            
            # Query variations
            r'\btell\s+me\s+about\b': 'who is',
            r'\binfo\s+about\b': 'who is',
            r'\binformation\s+about\b': 'who is',
            r'\bdetails\s+about\b': 'who is',
            r'\bdetails\s+of\b': 'who is',
            
            # Research variations
            r'\bresearch\s+work\b': 'research',
            r'\bresearch\s+area\b': 'research interests',
            r'\bworking\s+on\b': 'researching',
            
            # Publication variations
            r'\bpublished\s+papers?\b': 'publications',
            r'\bpublished\s+articles?\b': 'publications',
            r'\bresearch\s+papers?\b': 'publications',
        }
        
        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def _expand_query(self, query: str, intent: str, entities: List[str]) -> str:
        """
        Expand query with related terms for better retrieval.
        
        Uses synonym mapping and context-aware expansion.
        IMPORTANT: This expanded query is used ONLY for sparse/keyword search,
        NOT for dense embedding.
        
        Titles are stripped before expansion for clean keyword matching.
        
        Args:
            query: Original query
            intent: Detected intent
            entities: Extracted entities (names, etc.)
        
        Returns:
            Expanded query string for keyword search (titles stripped)
        """
        # Strip titles before expansion
        query_clean = self._strip_titles_for_embedding(query)
        query_lower = query_clean.lower()
        expansion_terms = []
        
        # Detect vague/broad queries and expand aggressively
        vague_indicators = [
            "what should i know", "what do i need", "tell me about",
            "what are", "list", "overview", "summary", "all",
            "important", "key", "main", "essential"
        ]
        
        is_vague = any(indicator in query_lower for indicator in vague_indicators)
        
        # Form query rewriting: redirect to procedure context
        form_pattern = r'\b(?:form|application)\s+([A-Z]{1,3}-?\d{1,3})\b'
        form_match = re.search(form_pattern, query_clean, re.IGNORECASE)
        
        if form_match and any(word in query_lower for word in ["fill", "filling", "complete", "how to", "instructions", "fields"]):
            # User asking how to fill a form - rewrite to search for the procedure
            form_name = form_match.group(0)
            # Extract the procedure context from common patterns
            if "leave" in query_lower:
                expansion_terms.extend(["leave", "application", "process", "procedure", form_name, "submit", "approval"])
            elif "reimbursement" in query_lower or "claim" in query_lower:
                expansion_terms.extend(["reimbursement", "claim", "process", "procedure", form_name, "submit", "approval"])
            elif "travel" in query_lower:
                expansion_terms.extend(["travel", "process", "procedure", form_name, "submit", "approval"])
            elif "grant" in query_lower or "research" in query_lower:
                expansion_terms.extend(["grant", "research", "process", "procedure", form_name, "submit", "approval"])
            else:
                # Generic form query - search for procedure that uses this form
                expansion_terms.extend(["process", "procedure", "application", form_name, "submit", "approval", "steps"])
        
        # Intent-based expansion
        if intent == "procedure":
            expansion_terms.extend(["steps", "process", "procedure", "documents", "approval", "requirements", "application", "submit", "form"])
        
        elif intent == "eligibility":
            expansion_terms.extend(["criteria", "requirements", "eligible", "qualify", "conditions", "prerequisites"])
        
        elif intent == "lookup":
            # Faculty-specific lookup
            if any(word in query_lower for word in ["professor", "prof", "dr", "faculty", "who is", "who are"]):
                expansion_terms.extend(["faculty", "professor", "research", "publications", "profile", "expertise", "qualifications", "awards", "name"])
                
                # Add name variations for better matching
                if entities:
                    # Strip titles from entity before extracting variations
                    clean_entity = self._strip_titles_for_embedding(entities[0])
                    name_variations = self._extract_name_variations(clean_entity)
                    expansion_terms.extend(name_variations)
            
            # Research/publication lookup
            if any(word in query_lower for word in ["research", "publication", "paper", "journal", "published"]):
                expansion_terms.extend(["research", "publications", "papers", "journals", "articles", "conferences", "citations"])
            
            # Award/achievement lookup
            if any(word in query_lower for word in ["award", "achievement", "recognition", "honor"]):
                expansion_terms.extend(["awards", "achievements", "recognition", "honors", "prizes"])
        
        # Aggressive expansion for vague queries
        if is_vague:
            # Add domain-specific terms based on keywords
            if "form" in query_lower:
                expansion_terms.extend([
                    "form", "forms", "application", "applications", "template", "templates",
                    "document", "documents", "submission", "submit", "fill", "complete",
                    "leave", "reimbursement", "travel", "research", "grant", "approval",
                    "request", "requisition", "claim", "declaration"
                ])
            
            if "policy" in query_lower or "policies" in query_lower:
                expansion_terms.extend([
                    "policy", "policies", "rule", "rules", "regulation", "regulations",
                    "guideline", "guidelines", "procedure", "procedures", "requirement", "requirements"
                ])
            
            if "leave" in query_lower:
                expansion_terms.extend([
                    "leave", "leaves", "vacation", "absence", "casual", "medical", "earned",
                    "maternity", "paternity", "sabbatical", "duty", "compensatory"
                ])
            
            # General broad query expansion
            if len(expansion_terms) < 5:
                expansion_terms.extend([
                    "faculty", "staff", "employee", "teacher", "professor",
                    "policy", "procedure", "guideline", "rule", "requirement",
                    "form", "application", "document", "process", "step"
                ])
        
        # Synonym-based expansion
        for key, synonyms in self.synonyms.items():
            if key in query_lower:
                # Add 2-3 most relevant synonyms
                expansion_terms.extend(synonyms[:3])
        
        # Remove duplicates and terms already in query
        query_words = set(query_lower.split())
        expansion_terms = [term for term in set(expansion_terms) if term.lower() not in query_words]
        
        # Limit expansion based on query type
        if is_vague:
            max_terms = 20  # More terms for vague queries
        elif "faculty" in expansion_terms or "professor" in expansion_terms:
            max_terms = 10  # Faculty queries
        else:
            max_terms = 8  # Specific queries
        
        expansion_terms = expansion_terms[:max_terms]
        
        if expansion_terms:
            return f"{query_clean} {' '.join(expansion_terms)}"
        
        return query_clean
    
    def _extract_name_variations(self, name: str) -> List[str]:
        """
        Generate name variations for better matching.
        
        Handles cases like:
        - "Pragati Khare" → ["Pragati", "Khare", "Pragati Khare"]
        - "Dr. John Smith" → ["John", "Smith", "John Smith"]
        """
        variations = []
        
        # Remove titles
        name_clean = re.sub(r'\b(?:dr\.?|prof\.?|professor|mr\.?|ms\.?|mrs\.?|miss)\s+', '', name, flags=re.IGNORECASE).strip()
        
        # Remove parentheses content (e.g., "(Shrivastava)")
        name_clean = re.sub(r'\([^)]*\)', '', name_clean).strip()
        
        # Split into parts
        parts = name_clean.split()
        
        if len(parts) >= 2:
            # Add first name
            variations.append(parts[0])
            # Add last name
            variations.append(parts[-1])
            # Add full name without middle parts
            if len(parts) > 2:
                variations.append(f"{parts[0]} {parts[-1]}")
        
        # Add original cleaned name
        if name_clean:
            variations.append(name_clean)
        
        return variations
    
    def _detect_intent(self, query: str) -> str:
        """
        Detect query intent using pattern matching with scoring.

        Returns the intent with highest confidence score.
        """
        # Form-code queries (e.g. "HR-LA-01", "Form CL-7") are actually
        # procedure/how-to-fill questions — they should not fall through to
        # policy_lookup. Catch them before generic scoring so the prompt
        # router picks FORM_DETAILS_PROMPT / PROCEDURE_PROMPT.
        form_code_re = re.compile(r'\b[A-Z]{2,3}-[A-Z]{1,3}-\d{1,3}\b', re.IGNORECASE)
        form_code_present = bool(form_code_re.search(query))
        procedure_verbs = any(
            re.search(p, query, re.IGNORECASE) for p in [
                r"\bhow\s+to\b", r"\bfill\b", r"\bsubmit\b",
                r"\bapply\b", r"\bapplication\b", r"\bprocess\b",
                r"\bprocedure\b", r"\bsteps\b", r"\brequest\b",
            ]
        )
        form_noun = bool(re.search(r"\bform\b", query, re.IGNORECASE))
        # Route form-centric queries to the form_details prompt so the LLM
        # gets form-specific instructions (sections, fields, approval chain).
        # Explicit how-to language still routes to procedure.
        if form_code_present or form_noun:
            if procedure_verbs and not form_code_present:
                return "procedure"
            if form_code_present and procedure_verbs:
                return "procedure"
            return "form_details"

        # Definition-shaped queries — "what is a sabbatical", "define X",
        # "meaning of Y" — should get the dictionary-style DEFINITION_PROMPT.
        definition_patterns = [
            r"^\s*define\s+\S+",
            r"^\s*what\s+does\s+\S.+\s+mean",
            r"\bmeaning\s+of\b",
            r"\bdefinition\s+of\b",
        ]
        if any(re.search(p, query, re.IGNORECASE) for p in definition_patterns):
            return "definition"

        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            intent_scores[intent] = score

        # Return intent with highest score, or general if no matches
        if max(intent_scores.values()) > 0:
            top = max(intent_scores, key=intent_scores.get)

            # Refine generic "lookup" into person_lookup vs policy_lookup so
            # hybrid-search weights can differ for these two very different
            # query shapes.
            if top == "lookup":
                # A bare form code with no surrounding procedure verbs is
                # still a "what is this form" lookup — treat as procedure
                # so the user gets step-by-step guidance.
                if form_code_present:
                    return "procedure"

                person_signals = [
                    r"\b(dr|prof|professor|mr|ms|mrs|miss)\.?\b",
                    r"\bfaculty\b", r"\bteacher\b", r"\blecturer\b",
                    r"\bwho\s+(is|are|teaches|researches|wrote|published)\b",
                    r"\bpublications?\s+(of|by)\b",
                    r"\bprofile\s+of\b", r"\bexpertise\s+of\b",
                ]
                policy_signals = [
                    r"\bpolicy\b", r"\bclause\b", r"\bsection\b",
                    r"\bleave\b", r"\bsabbatical\b", r"\bgratuity\b",
                    r"\bprovident\s+fund\b", r"\bmedical\s+reimbursement\b",
                    r"\bagreement\b", r"\bcontract\b",
                    r"\bguideline(s)?\b", r"\beligibility\b",
                ]
                person_hits = sum(
                    1 for p in person_signals if re.search(p, query, re.IGNORECASE)
                )
                policy_hits = sum(
                    1 for p in policy_signals if re.search(p, query, re.IGNORECASE)
                )
                if policy_hits > person_hits:
                    return "policy_lookup"
                return "person_lookup"

            return top

        return "general"
    
    def _detect_domain(self, query: str) -> str:
        """
        Detect query domain using pattern matching with weighted scoring.
        
        Returns the domain with highest confidence score.
        """
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                # Weight by number of matches
                score += len(matches)
            domain_scores[domain] = score
        
        # Return domain with highest score, or general if no strong match
        max_score = max(domain_scores.values())
        if max_score > 0:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract named entities from query with normalization.
        
        Handles various name formats and extracts multiple entity types.
        """
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            
            # Normalize person names
            if entity_type == "person":
                for match in matches:
                    # Remove title prefixes for cleaner matching
                    normalized = re.sub(r'\b(?:dr\.?|prof\.?|professor|mr\.?|ms\.?|mrs\.?|miss)\s+', '', match, flags=re.IGNORECASE)
                    normalized = normalized.strip()
                    if normalized:
                        entities.append(normalized)
                        # Also add original with title
                        entities.append(match.strip())
            else:
                entities.extend([m.strip() for m in matches])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _is_current_only(self, query: str) -> bool:
        """
        Determine if query should filter for current documents only.

        Looks for temporal cues suggesting the user wants current/latest info.
        """
        current_indicators = [
            r"\bcurrent\b", r"\blatest\b", r"\bnow\b", r"\btoday\b",
            r"\bthis\s+(year|semester|term)\b", r"\brecent\b", r"\bnew\b",
            r"\bactive\b", r"\bongoing\b", r"\bpresent\b",
            r"\bwhat.?s\s+the\s+current\b",
        ]
        for pattern in current_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False

    def _build_metadata_filters(
        self,
        domain: str,
        is_current_only: bool,
        entities: List[str],
    ) -> Dict[str, Any]:
        """
        Build Qdrant metadata filters for pre-search pruning.

        Strategy is intentionally conservative — over-filtering hurts recall
        badly on small corpora, so we only attach filters for signals we're
        confident about (e.g. an explicit "current/latest" request). Any key
        we don't want to filter on is simply left off the returned dict;
        downstream code treats an empty dict as "no filter applied".

        Args:
            domain: Detected domain (faculty_info, policies, procedures, ...)
            is_current_only: True when the query asked for the latest version.
            entities: Normalized entity tokens pulled from the query.

        Returns:
            Dict suitable for ``VectorDBClient._build_filter``.
        """
        filters: Dict[str, Any] = {}

        # Only pre-filter on "current" when the user asked for it explicitly.
        # We don't infer it from ambient context because the corpus is small
        # and stale-document filtering cuts recall more than it helps.
        if is_current_only:
            filters["is_current"] = True

        # Domain filtering is reserved for future extensions once we have
        # enough document-level metadata to distinguish faculty-profile
        # chunks from policy chunks reliably. Keeping it off for now.
        _ = domain
        _ = entities

        return filters
