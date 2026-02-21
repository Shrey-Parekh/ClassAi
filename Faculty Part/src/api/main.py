"""
FastAPI application for Faculty Part RAG system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ..retrieval.pipeline import RetrievalPipeline
from ..generation.answer_generator import AnswerGenerator
from ..utils.vector_db import VectorDBClient
from ..utils.embeddings import EmbeddingModel
from ..utils.llm import LLMClient


app = FastAPI(
    title="ClassAI Faculty Part",
    description="AI-powered faculty resource assistant with semantic RAG",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
from fastapi.staticfiles import StaticFiles
from pathlib import Path

frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


# Request/Response models
class QueryRequest(BaseModel):
    """Faculty query request."""
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    """Query response with answer and sources."""
    answer: str
    sources: List[Dict[str, str]]
    intent: str
    chunks_used: int
    metadata: Dict[str, Any]


# Initialize components (in production, use dependency injection)
retrieval_pipeline = None
answer_generator = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global retrieval_pipeline, answer_generator
    
    try:
        # Initialize components
        print("Initializing Faculty Part RAG system...")
        
        vector_db = VectorDBClient()
        embedding_model = EmbeddingModel()
        llm_client = LLMClient()
        
        # Create collection if it doesn't exist
        vector_db.create_collection(
            vector_size=embedding_model.get_dimension()
        )
        
        # Initialize retrieval pipeline
        retrieval_pipeline = RetrievalPipeline(
            vector_db_client=vector_db,
            embedding_model=embedding_model
        )
        
        # Build BM25 index
        print("Building BM25 index...")
        retrieval_pipeline.search_engine.build_bm25_index()
        
        # Initialize answer generator
        answer_generator = AnswerGenerator(llm_client)
        
        print("✓ Faculty Part API started successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("Make sure Qdrant is running and .env is configured")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "ClassAI Faculty Part",
        "status": "running",
        "version": "0.1.0"
    }


@app.post("/query", response_model=QueryResponse)
async def query_faculty_resources(request: QueryRequest):
    """
    Query faculty resources using semantic RAG.
    
    Pipeline:
    1. Intent classification
    2. Hybrid search (vector + BM25)
    3. Cross-encoder reranking
    4. LLM answer generation
    """
    if not retrieval_pipeline or not answer_generator:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        # Retrieve relevant chunks
        retrieval_result = retrieval_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k
        )
        
        # Generate answer
        answer_result = answer_generator.generate(
            query=request.query,
            retrieved_chunks=retrieval_result["chunks"],
            intent_type=retrieval_result["intent"]
        )
        
        return QueryResponse(
            answer=answer_result["answer"],
            sources=answer_result["sources"],
            intent=retrieval_result["intent"],
            chunks_used=answer_result["chunks_used"],
            metadata=retrieval_result["metadata"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "retrieval_pipeline": retrieval_pipeline is not None,
            "answer_generator": answer_generator is not None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
