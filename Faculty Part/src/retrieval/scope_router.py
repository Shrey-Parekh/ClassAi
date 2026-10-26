"""
Scope-based routing for multi-collection RAG.

Routes queries to faculty and/or student collections based on user role and scope preference.
Uses Reciprocal Rank Fusion (RRF) to merge results from multiple collections.
"""

from typing import Dict, Any, List, Optional
import logging
import os

from .pipeline import RetrievalPipeline


class ScopeRouter:
    """
    Route queries to appropriate collection(s) based on role and scope.
    
    Role-based access:
    - student: Can ONLY query student collection
    - faculty: Can query student AND faculty collections
    - admin: Can query student AND faculty collections
    
    Scope options (for faculty/admin):
    - "student": Query student collection only
    - "faculty": Query faculty collection only
    - "both": Query both collections and merge results (default)
    """
    
    def __init__(
        self,
        vector_db_client,
        embedding_model,
        llm_client,
        faculty_collection_name: str = None,
        student_collection_name: str = None
    ):
        """
        Initialize scope router with two pipeline instances.
        
        Args:
            vector_db_client: Shared Qdrant client
            embedding_model: Shared embedding model (BAAI/bge-m3)
            llm_client: Shared LLM client
            faculty_collection_name: Faculty collection name (from env if None)
            student_collection_name: Student collection name (from env if None)
        """
        self.logger = logging.getLogger(__name__)
        
        # Get collection names from env if not provided
        if faculty_collection_name is None:
            faculty_collection_name = os.getenv("FACULTY_COLLECTION_NAME", "faculty_chunks")
        if student_collection_name is None:
            student_collection_name = os.getenv("STUDENT_COLLECTION_NAME", "academic_rag")
        
        self.faculty_collection_name = faculty_collection_name
        self.student_collection_name = student_collection_name
        
        self.logger.info(f"Initializing ScopeRouter:")
        self.logger.info(f"  Faculty collection: {faculty_collection_name}")
        self.logger.info(f"  Student collection: {student_collection_name}")
        
        # Create two pipeline instances with shared components
        self.faculty_pipeline = RetrievalPipeline(
            vector_db_client=vector_db_client,
            embedding_model=embedding_model,
            collection_name=faculty_collection_name,
            llm_client=llm_client
        )
        
        self.student_pipeline = RetrievalPipeline(
            vector_db_client=vector_db_client,
            embedding_model=embedding_model,
            collection_name=student_collection_name,
            llm_client=llm_client
        )
        
        self.logger.info("✓ ScopeRouter initialized with two pipelines")
    
    def retrieve(
        self,
        query: str,
        role: str,
        scope: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve chunks based on role and scope.
        
        Args:
            query: User query
            role: User role ("student", "faculty", "admin")
            scope: Scope preference ("student", "faculty", "both", or None)
            top_k: Number of final results to return
        
        Returns:
            Dict with chunks, intent, domain, entities, metadata
        """
        # Normalize role and scope
        role = role.lower() if role else "student"
        scope = scope.lower() if scope else "both"
        
        # Enforce role-based access control
        if role == "student":
            # Students can ONLY access student collection
            scope = "student"
            self.logger.info(f"[SCOPE] Role=student → forced scope=student")
        elif role in ("faculty", "admin"):
            # Faculty/admin can choose scope
            if scope not in ("student", "faculty", "both"):
                scope = "both"  # Default to both
            self.logger.info(f"[SCOPE] Role={role}, scope={scope}")
        else:
            # Unknown role → default to student (most restrictive)
            self.logger.warning(f"[SCOPE] Unknown role '{role}' → defaulting to student")
            role = "student"
            scope = "student"
        
        # Whole-document questions about course material ("all units", "every
        # question mapped to CO-2") need every matching chunk, which similarity
        # search cannot guarantee. Answer those from metadata instead.
        if scope in ("student", "both"):
            structured = self._structured_student_lookup(self._expand_abbreviations(query))
            if structured:
                return structured

        # Route based on scope
        if scope == "student":
            return self._retrieve_student_only(query, top_k)
        elif scope == "faculty":
            return self._retrieve_faculty_only(query, top_k)
        elif scope == "both":
            return self._retrieve_both_merged(query, top_k)
        else:
            # Fallback (should never reach here)
            self.logger.error(f"[SCOPE] Invalid scope '{scope}' → defaulting to student")
            return self._retrieve_student_only(query, top_k)
    
    def _retrieve_student_only(self, query: str, top_k: int) -> Dict[str, Any]:
        """Retrieve from student collection only."""
        self.logger.info(f"[RETRIEVE] Student collection only (top_k={top_k})")
        
        # Expand common abbreviations
        expanded_query = self._expand_abbreviations(query)
        if expanded_query != query:
            self.logger.info(f"[EXPAND] '{query}' -> '{expanded_query}'")
            query = expanded_query
        
        # For student queries, increase top_k for better coverage of course content
        # Student queries often ask for "all units", "all topics", etc.
        adjusted_top_k = max(top_k, 10)  # Minimum 10 chunks for student queries
        
        result = self.student_pipeline.retrieve(query, adjusted_top_k)
        
        # Tag all chunks with source_collection
        for chunk in result.get("chunks", []):
            chunk["metadata"]["source_collection"] = "student"
        
        result["metadata"]["retrieval_path"] = "student_only"
        result["metadata"]["collections_queried"] = ["student"]
        result["metadata"]["top_k_adjusted"] = adjusted_top_k
        
        return result
    
    def _retrieve_faculty_only(self, query: str, top_k: int) -> Dict[str, Any]:
        """Retrieve from faculty collection only."""
        self.logger.info(f"[RETRIEVE] Faculty collection only (top_k={top_k})")
        
        result = self.faculty_pipeline.retrieve(query, top_k)
        
        # Tag all chunks with source_collection
        for chunk in result.get("chunks", []):
            chunk["metadata"]["source_collection"] = "faculty"
        
        result["metadata"]["retrieval_path"] = "faculty_only"
        result["metadata"]["collections_queried"] = ["faculty"]
        
        return result
    
    def _retrieve_both_merged(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        Retrieve from both collections and merge using Reciprocal Rank Fusion.
        
        RRF formula: score(chunk) = sum(1 / (k + rank_i)) for all rankings
        where k=60 is a constant that reduces the impact of high ranks.
        """
        self.logger.info(f"[RETRIEVE] Both collections with RRF merge (top_k={top_k})")
        
        # Retrieve from both pipelines in parallel (could use threading for true parallelism)
        # For now, sequential is fine since both are fast
        
        # Retrieve more candidates for better merging
        retrieve_k = top_k * 2
        
        faculty_result = self.faculty_pipeline.retrieve(query, retrieve_k)
        student_result = self.student_pipeline.retrieve(query, retrieve_k)
        
        # Tag chunks with source collection
        for chunk in faculty_result.get("chunks", []):
            chunk["metadata"]["source_collection"] = "faculty"
        
        for chunk in student_result.get("chunks", []):
            chunk["metadata"]["source_collection"] = "student"
        
        # Merge using Reciprocal Rank Fusion (RRF)
        merged_chunks = self._reciprocal_rank_fusion(
            faculty_chunks=faculty_result.get("chunks", []),
            student_chunks=student_result.get("chunks", []),
            k=60
        )
        
        # Take top_k after merging
        merged_chunks = merged_chunks[:top_k]
        
        self.logger.info(
            f"[RRF] Merged {len(faculty_result.get('chunks', []))} faculty + "
            f"{len(student_result.get('chunks', []))} student → {len(merged_chunks)} final"
        )
        
        # Use faculty result as base, update with merged chunks
        result = faculty_result.copy()
        result["chunks"] = merged_chunks
        result["metadata"]["retrieval_path"] = "both_rrf_merged"
        result["metadata"]["collections_queried"] = ["faculty", "student"]
        result["metadata"]["rrf_k"] = 60
        result["metadata"]["faculty_count"] = len(faculty_result.get("chunks", []))
        result["metadata"]["student_count"] = len(student_result.get("chunks", []))
        
        return result
    
    def _reciprocal_rank_fusion(
        self,
        faculty_chunks: List[Dict[str, Any]],
        student_chunks: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) across all lists where chunk appears.
        
        Args:
            faculty_chunks: Ranked chunks from faculty collection
            student_chunks: Ranked chunks from student collection
            k: RRF constant (default: 60)
        
        Returns:
            Merged and re-ranked chunks
        """
        # Build RRF scores
        rrf_scores = {}
        
        # Process faculty chunks
        for rank, chunk in enumerate(faculty_chunks, start=1):
            chunk_id = chunk.get("chunk_id", f"faculty_{rank}")
            rrf_scores[chunk_id] = {
                "score": 1.0 / (k + rank),
                "chunk": chunk,
                "faculty_rank": rank,
                "student_rank": None
            }
        
        # Process student chunks
        for rank, chunk in enumerate(student_chunks, start=1):
            chunk_id = chunk.get("chunk_id", f"student_{rank}")
            
            if chunk_id in rrf_scores:
                # Chunk appears in both lists
                rrf_scores[chunk_id]["score"] += 1.0 / (k + rank)
                rrf_scores[chunk_id]["student_rank"] = rank
            else:
                # Chunk only in student list
                rrf_scores[chunk_id] = {
                    "score": 1.0 / (k + rank),
                    "chunk": chunk,
                    "faculty_rank": None,
                    "student_rank": rank
                }
        
        # Sort by RRF score (descending)
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Extract chunks and update their scores with RRF score
        merged = []
        for chunk_id, item in sorted_items:
            chunk = item["chunk"]
            chunk["rrf_score"] = item["score"]
            chunk["metadata"]["rrf_score"] = item["score"]
            if item["faculty_rank"]:
                chunk["metadata"]["faculty_rank"] = item["faculty_rank"]
            if item["student_rank"]:
                chunk["metadata"]["student_rank"] = item["student_rank"]
            merged.append(chunk)
        
        return merged

    
    def _expand_abbreviations(self, query: str) -> str:
        """
        Expand common course abbreviations to full names.
        
        Args:
            query: Original query
        
        Returns:
            Query with abbreviations expanded
        """
        import re
        
        # Common abbreviations for courses in the database
        abbreviations = {
            r'\bCS\b': 'Cyber Security',
            r'\bML\b': 'Machine Learning',
            r'\bHCI\b': 'Human Computer Interaction',
            r'\bCV\b': 'Computer Vision',
            r'\bDC\b': 'Distributed Computing',
            r'\bSQA\b': 'Software Quality Assurance',
        }
        
        expanded = query
        for abbr_pattern, full_name in abbreviations.items():
            # Only expand if it's a standalone word (not part of another word)
            if re.search(abbr_pattern, expanded, re.IGNORECASE):
                expanded = re.sub(abbr_pattern, full_name, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def _structured_student_lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """Metadata-based answers for exam-paper and full-syllabus questions, or None."""
        import re

        q = query.lower()
        course = self._detect_course(q)
        if not course:
            return None

        co_match = re.search(r"\bco-?\s?(\d+)\b", q)
        about_questions = re.search(r"\bquestions?\b|\bq\d", q)
        wants_all = re.search(r"\b(all|every|each|list|which|mapping|map|maps|mapped)\b", q)
        if about_questions and (co_match or wants_all):
            co = f"CO-{co_match.group(1)}" if co_match else None
            with_units = bool(re.search(r"\bunits?\b", q))
            self.logger.info(f"[STRUCTURED] Question paper lookup: course={course} co={co} units={with_units}")
            result = self._question_paper_chunks(course, co, with_units)
            if result:
                return result

        if self._detect_list_all_query(query):
            self.logger.info(f"[STRUCTURED] Syllabus unit listing: course={course}")
            return self._retrieve_by_metadata_filter(
                course_name=course, chunk_type='syllabus_unit', collection='student'
            )
        return None

    def _detect_course(self, query_lower: str) -> Optional[str]:
        """Course name mentioned in the query, normalised to the indexed name."""
        import re
        course_name_map = {
            'machine learning': 'Machine Learning',
            'ml': 'Machine Learning',
            'cyber security': 'Cyber Security',
            'distributed computing': 'Distributed Computing',
            'computer vision': 'Computer Vision',
            'human computer interaction': 'Human Computer Interaction',
            'hci': 'Human Computer Interaction',
            'biometrics': 'Biometrics',
            'microservices': 'Microservices and Architecture',
            'software quality': 'Software Quality Assurance',
            'interpersonal skills': 'Interpersonal Skills',
        }
        match = re.search(r'\b(' + '|'.join(course_name_map) + r')\b', query_lower)
        return course_name_map[match.group(1)] if match else None

    def _question_paper_chunks(self, course: str, co: Optional[str], with_units: bool) -> Optional[Dict[str, Any]]:
        """
        Every question from the course's paper (optionally only one CO), in paper
        order, as compact one-question chunks; plus the syllabus units when the
        question asks to map questions to units.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self.student_pipeline.vector_db.client
        must = [
            FieldCondition(key='metadata.subject', match=MatchValue(value=course)),
            FieldCondition(key='metadata.doc_type', match=MatchValue(value='question_paper')),
        ]
        if co:
            must.append(FieldCondition(key='metadata.course_outcome', match=MatchValue(value=co)))
        points, _ = client.scroll(self.student_collection_name, scroll_filter=Filter(must=must),
                                  limit=100, with_payload=True, with_vectors=False)
        if not points:
            return None

        def order(p):
            m = p.payload.get("metadata", {})
            return (int(m.get("question_number") or 99), str(m.get("question_part") or ""))

        chunks = []
        for p in sorted(points, key=order):
            m = p.payload.get("metadata", {})
            body = " ".join((p.payload.get("page_content") or "").split())[:400]
            header = f"{m.get('question_id')} [{m.get('course_outcome')}; {m.get('marks')} marks; {m.get('bloom_level')}]"
            chunks.append({"chunk_id": p.id, "text": f"{header}: {body}", "score": 1.0,
                           "metadata": {**m, "source_collection": "student"}})

        if with_units:
            units, _ = client.scroll(self.student_collection_name, limit=50, with_payload=True, with_vectors=False,
                                     scroll_filter=Filter(must=[
                                         FieldCondition(key='metadata.course_name', match=MatchValue(value=course)),
                                         FieldCondition(key='metadata.chunk_type', match=MatchValue(value='syllabus_unit')),
                                     ]))
            for u in sorted(units, key=lambda x: int(x.payload.get("metadata", {}).get("unit_number") or 99)):
                m = u.payload.get("metadata", {})
                body = " ".join((u.payload.get("page_content") or "").split())[:500]
                chunks.append({"chunk_id": u.id, "text": f"SYLLABUS Unit {m.get('unit_number')}: {body}",
                               "score": 1.0, "metadata": {**m, "source_collection": "student"}})

        self.logger.info(f"[STRUCTURED] {len(chunks)} chunks for {course} (co={co}, units={with_units})")
        return {
            "chunks": chunks,
            "intent": "listing",
            "domain": "academic",
            "entities": [course],
            "format_preference": None,
            "metadata": {
                "retrieval_path": "question_paper_filter",
                "collections_queried": ["student"],
                "filter_applied": {"subject": course, "course_outcome": co, "with_units": with_units},
            },
        }

    def _detect_list_all_query(self, query: str) -> Optional[Dict[str, str]]:
        """
        Detect if query is asking for a comprehensive list (e.g., "list all units").
        
        Returns:
            Dict with course_name if detected, None otherwise
        """
        import re
        
        query_lower = query.lower()
        
        # Patterns for "list all" queries
        list_patterns = [
            r'\b(list|show|tell|give|what are|display)\s+(all|the)\s+(units?|topics?|chapters?)',
            r'\ball\s+(units?|topics?|chapters?)\s+(in|of|for)',
            r'\b(units?|topics?|chapters?)\s+in\s+(.+?)\s+(course|subject)',
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, query_lower):
                # Extract course name
                course_patterns = [
                    r'(machine learning|ml|cyber security|distributed computing|computer vision|'
                    r'human computer interaction|hci|business information|biometrics|'
                    r'microservices|software quality|interpersonal skills)',
                ]
                
                for course_pattern in course_patterns:
                    match = re.search(course_pattern, query_lower)
                    if match:
                        course_name_raw = match.group(1)
                        # Normalize course name
                        course_name_map = {
                            'ml': 'Machine Learning',
                            'machine learning': 'Machine Learning',
                            'cyber security': 'Cyber Security',
                            'distributed computing': 'Distributed Computing',
                            'computer vision': 'Computer Vision',
                            'hci': 'Human Computer Interaction',
                            'human computer interaction': 'Human Computer Interaction',
                            'biometrics': 'Biometrics',
                            'microservices': 'Microservices and Architecture',
                            'software quality': 'Software Quality Assurance',
                            'interpersonal skills': 'Interpersonal Skills',
                        }
                        course_name = course_name_map.get(course_name_raw, course_name_raw.title())
                        
                        self.logger.info(f"[DETECT] List-all query detected for course: {course_name}")
                        return {'course_name': course_name}
        
        return None
    
    def _retrieve_by_metadata_filter(
        self,
        course_name: str,
        chunk_type: str,
        collection: str
    ) -> Dict[str, Any]:
        """
        Retrieve chunks using metadata filtering (for comprehensive listing).
        
        Args:
            course_name: Course name to filter by
            chunk_type: Chunk type (e.g., 'syllabus_unit')
            collection: 'student' or 'faculty'
        
        Returns:
            Dict with chunks and metadata
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        collection_name = (
            self.student_collection_name if collection == 'student'
            else self.faculty_collection_name
        )
        
        try:
            # Use scroll to get all matching chunks
            results, _ = self.student_pipeline.vector_db.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key='metadata.course_name',
                            match=MatchValue(value=course_name)
                        ),
                        FieldCondition(
                            key='metadata.chunk_type',
                            match=MatchValue(value=chunk_type)
                        )
                    ]
                ),
                limit=50,  # Max 50 units (should be enough for any course)
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                self.logger.warning(f"[FILTER] No results found for {course_name} / {chunk_type}")
                # Fall back to semantic search
                return self.student_pipeline.retrieve(f"all units in {course_name}", 20)
            
            # Sort by unit number
            sorted_results = sorted(
                results,
                key=lambda x: int(x.payload.get('metadata', {}).get('unit_number', 999))
            )
            
            # Convert to chunk format
            chunks = []
            for point in sorted_results:
                chunks.append({
                    "chunk_id": point.id,
                    "text": point.payload.get("content") or point.payload.get("page_content", ""),
                    "score": 1.0,  # All chunks equally relevant for listing
                    "metadata": point.payload.get("metadata", {}),
                })
            
            # Tag with source collection
            for chunk in chunks:
                chunk["metadata"]["source_collection"] = collection
            
            self.logger.info(f"[FILTER] Retrieved {len(chunks)} chunks via metadata filtering")
            
            return {
                "chunks": chunks,
                "intent": "listing",
                "domain": "academic",
                "entities": [course_name],
                "format_preference": None,
                "metadata": {
                    "retrieval_path": "metadata_filter",
                    "collections_queried": [collection],
                    "filter_applied": {
                        "course_name": course_name,
                        "chunk_type": chunk_type
                    },
                    "initial_results": len(chunks),
                    "final_results": len(chunks)
                }
            }
            
        except Exception as e:
            self.logger.error(f"[FILTER] Metadata filtering failed: {e}")
            # Fall back to semantic search
            return self.student_pipeline.retrieve(f"all units in {course_name}", 20)
