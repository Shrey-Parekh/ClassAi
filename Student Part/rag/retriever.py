"""
Simple and effective retrieval for NMIMS academic documents.
"""
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class AcademicRetriever:
    """Retriever optimized for academic content."""
    
    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = "academic_rag",
        k: int = 8
    ):
        import os
        
        # Use environment variable or parameter
        if qdrant_url is None:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.k = k
        
        # Initialize
        self.embeddings = OllamaEmbeddings(model="bge-m3")
        self.client = QdrantClient(url=qdrant_url)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Extract entities and intent from query."""
        query_lower = query.lower()
        
        analysis = {
            "query": query,
            "intent": "general",
            "entities": {}
        }
        
        # Detect intent
        if any(word in query_lower for word in ["unit", "topic", "syllabus", "course"]):
            analysis["intent"] = "syllabus"
        elif any(word in query_lower for word in ["question", "exam", "paper", "q", "marks"]):
            analysis["intent"] = "question_paper"
        
        # Extract entities
        # Unit numbers
        unit_matches = re.findall(r'unit\s*(\d+)', query, re.IGNORECASE)
        if unit_matches:
            analysis["entities"]["units"] = unit_matches
        
        # Question numbers
        q_matches = re.findall(r'q(?:uestion)?\s*(\d+)', query, re.IGNORECASE)
        if q_matches:
            analysis["entities"]["questions"] = q_matches
        
        # CO references
        co_matches = re.findall(r'co-?(\d+)', query, re.IGNORECASE)
        if co_matches:
            analysis["entities"]["cos"] = [f"CO-{co}" for co in co_matches]
        
        # SO references
        so_matches = re.findall(r'so-?(\d+)', query, re.IGNORECASE)
        if so_matches:
            analysis["entities"]["sos"] = [f"SO-{so}" for so in so_matches]
        
        # Course names
        courses = ["machine learning", "cyber security", "distributed computing"]
        for course in courses:
            if course in query_lower:
                analysis["entities"]["course"] = course
                break
        
        return analysis
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve relevant documents."""
        k = k or self.k
        
        # Analyze query
        analysis = self.analyze_query(query)
        
        # Detect "list all" queries and adjust strategy
        query_lower = query.lower()
        is_list_all_query = any(phrase in query_lower for phrase in 
                                ["all units", "list units", "all unit", "every unit", "complete list", "list all"])
        
        if is_list_all_query:
            # For "list all" queries, retrieve many more documents
            k = 30  # Get enough to cover all units
            search_k = 50  # Retrieve even more for filtering
        else:
            search_k = k * 2
        
        # Retrieve documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": search_k})
        docs = retriever.invoke(query)
        
        # For "list all units" queries, filter and deduplicate
        if is_list_all_query:
            # Keep only syllabus_unit chunks
            docs = [d for d in docs if d.metadata.get("chunk_type") == "syllabus_unit"]
            
            # If asking about specific course, filter by course
            if "course" in analysis.get("entities", {}):
                course = analysis["entities"]["course"]
                docs = [d for d in docs if course in d.metadata.get("course_name", "").lower()]
            
            # Deduplicate by unit number (keep first occurrence of each unit)
            seen_units = set()
            unique_docs = []
            for doc in docs:
                unit_key = (
                    doc.metadata.get("course_name", ""),
                    doc.metadata.get("unit_number", "")
                )
                if unit_key not in seen_units:
                    seen_units.add(unit_key)
                    unique_docs.append(doc)
            docs = unique_docs
        
        # Rerank
        if docs:
            docs = self.rerank(docs, analysis)
        
        return docs[:k]
    
    def rerank(self, docs: List[Document], analysis: Dict[str, Any]) -> List[Document]:
        """Rerank documents based on query analysis."""
        if not docs:
            return []
        
        query_lower = analysis["query"].lower()
        query_terms = set(query_lower.split())
        entities = analysis.get("entities", {})
        
        # Check if this is a "list all units" query
        is_list_all = any(phrase in query_lower for phrase in 
                         ["all units", "list units", "all unit", "every unit", "list all"])
        
        scored_docs = []
        
        for doc in docs:
            score = 0.0
            content_lower = doc.page_content.lower()
            metadata = doc.metadata
            
            # For "list all units" queries, heavily prioritize course match
            if is_list_all and "course" in entities:
                target_course = entities["course"]
                course_name = metadata.get("course_name", "").lower()
                
                if target_course in course_name:
                    score += 100  # Massive boost for correct course
                    # Ensure all units from this course rank high
                    if metadata.get("chunk_type") == "syllabus_unit":
                        score += 50  # Additional boost for being a unit
                else:
                    score -= 50  # Penalize wrong course
            
            # 1. Keyword matching
            matching_terms = sum(1 for term in query_terms if term in content_lower)
            score += matching_terms * 2
            
            # 2. Entity matching (high priority)
            # Unit matching
            if "units" in entities:
                for unit in entities["units"]:
                    if metadata.get("unit_number") == unit:
                        score += 30
                    elif f"unit {unit}" in content_lower:
                        score += 15
            
            # Question matching
            if "questions" in entities:
                for q_num in entities["questions"]:
                    if metadata.get("question_number") == int(q_num):
                        score += 30
            
            # CO matching
            if "cos" in entities:
                for co in entities["cos"]:
                    if co.lower() in content_lower:
                        score += 20
                    if co in metadata.get("cos", []):
                        score += 10
            
            # SO matching
            if "sos" in entities:
                for so in entities["sos"]:
                    if so.lower() in content_lower:
                        score += 20
                    if so in metadata.get("sos", []):
                        score += 10
            
            # Course matching (for non-list-all queries)
            if not is_list_all and "course" in entities:
                course = entities["course"]
                course_name = metadata.get("course_name", "").lower()
                if course in course_name:
                    score += 25  # High boost for exact course match
                elif course in content_lower:
                    score += 15
            
            # 3. Intent alignment
            intent = analysis.get("intent", "general")
            chunk_type = metadata.get("chunk_type", "")
            
            if intent == "syllabus" and "unit" in chunk_type:
                score += 10
            elif intent == "question_paper" and "question" in chunk_type:
                score += 10
            
            # 4. Content quality
            word_count = metadata.get("word_count", 0)
            if 50 <= word_count <= 1000:
                score += 3
            
            scored_docs.append((doc, score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs]


def get_retriever(
    qdrant_url: str = None,
    collection_name: str = "academic_rag",
    k: int = 8
) -> AcademicRetriever:
    """Factory function to create retriever."""
    return AcademicRetriever(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        k=k
    )


if __name__ == "__main__":
    # Test retriever
    import sys
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is covered in Unit 2 of Cyber Security?"
    
    print(f"Query: {query}\n")
    
    try:
        retriever = get_retriever(k=5)
        
        # Analyze
        analysis = retriever.analyze_query(query)
        print(f"Analysis:")
        print(f"  Intent: {analysis['intent']}")
        print(f"  Entities: {analysis['entities']}\n")
        
        # Retrieve
        docs = retriever.retrieve(query)
        
        print(f"Retrieved {len(docs)} documents:\n")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. Type: {doc.metadata.get('chunk_type', 'unknown')}")
            print(f"   Subject: {doc.metadata.get('subject', 'Unknown')[:60]}")
            print(f"   Preview: {doc.page_content[:100]}...")
            print()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running: ollama serve")
        print("2. Documents are indexed: python ingest/index.py")
