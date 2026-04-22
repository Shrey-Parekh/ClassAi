"""
Simple and effective RAG chain for NMIMS academic Q&A.
"""
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from rag.retriever import get_retriever


class AcademicRAG:
    """RAG chain for academic Q&A."""
    
    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = "academic_rag",
        llm_model: str = "gemma3:12b",
        k: int = 8
    ):
        self.k = k
        
        # Initialize retriever
        self.retriever = get_retriever(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            k=k
        )
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.1,
            num_ctx=8192
        )
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_template("""You are an expert academic assistant for NMIMS university students. Answer questions based STRICTLY on the provided context.

INSTRUCTIONS:
1. Answer using ONLY information from the context below
2. For syllabus questions:
   - When asked for "all units" or "list units", provide a COMPLETE list of ALL units found in the context
   - Include unit numbers, names, and key topics for each unit
   - Use bullet points or numbered lists for clarity
   - Include credits, evaluation schemes, course outcomes if mentioned

3. For question paper questions:
   - Show the exact question text
   - Include marks, CO, SO, and bloom level if mentioned
   - Explain what the question is asking

4. If information is not in the context, say: "This information is not available in the provided documents."

5. Be precise, complete, and academic in tone
6. When listing items, ensure you include ALL items from the context, not just a few examples

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")
        
        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents as context."""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            
            # Build header
            header_parts = [f"[Document {i}]"]
            
            if "subject" in meta:
                subject = meta["subject"][:80]
                header_parts.append(f"Subject: {subject}")
            
            if "chunk_type" in meta:
                header_parts.append(f"Type: {meta['chunk_type']}")
            
            if "course_name" in meta:
                header_parts.append(f"Course: {meta['course_name']}")
            
            if "unit_number" in meta:
                header_parts.append(f"Unit: {meta['unit_number']}")
            
            if "question_id" in meta:
                header_parts.append(f"Question: {meta['question_id']}")
            
            if "marks" in meta:
                header_parts.append(f"Marks: {meta['marks']}")
            
            header = " | ".join(header_parts)
            
            # Format document
            formatted.append(f"{header}\n\n{doc.page_content}\n")
        
        return "\n" + ("=" * 80 + "\n\n").join(formatted)
    
    def __call__(self, query: str) -> str:
        """Process query and generate answer."""
        try:
            # Retrieve documents
            docs = self.retriever.retrieve(query, k=self.k)
            
            if not docs:
                return "I couldn't find any relevant information in the indexed documents. Please make sure documents are properly indexed."
            
            # Format context
            context = self.format_context(docs)
            
            # Generate answer
            answer = self.chain.invoke({
                "context": context,
                "question": query
            })
            
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}\n\nPlease check:\n1. Ollama is running: ollama serve\n2. Model is available: ollama pull gemma3:12b"


def build_rag_chain(
    qdrant_url: str = None,
    collection_name: str = "academic_rag",
    llm_model: str = "gemma3:12b",
    k: int = 8
) -> AcademicRAG:
    """Factory function to build RAG chain."""
    return AcademicRAG(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        llm_model=llm_model,
        k=k
    )


if __name__ == "__main__":
    # Test chain
    import sys
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What topics are covered in Unit 2 of Cyber Security?"
    
    print(f"Query: {query}\n")
    print("Generating answer...\n")
    print("=" * 80)
    
    try:
        chain = build_rag_chain(k=5)
        answer = chain(query)
        print(answer)
        print("=" * 80)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Documents are indexed: python ingest/index.py")
        print("2. Ollama is running: ollama serve")
        print("3. Model is available: ollama pull gemma3:12b")
