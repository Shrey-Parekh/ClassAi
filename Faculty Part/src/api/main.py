"""
FastAPI application for Faculty Part RAG system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..retrieval.pipeline import RetrievalPipeline
from ..generation.answer_generator import AnswerGenerator
from ..utils.vector_db import VectorDBClient
from ..utils.query_embedder import QueryEmbedder
from ..utils.llm import LLMClient


app = FastAPI(
    title="ClassAI Faculty Part",
    description="AI-powered faculty resource assistant with semantic RAG",
    version="0.1.0"
)

# CORS middleware - configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
        query_embedder = QueryEmbedder(model_name="BAAI/bge-m3")
        llm_client = LLMClient()
        
        # Create collection if it doesn't exist
        try:
            vector_db.create_collection(
                vector_size=query_embedder.get_dimension()
            )
        except Exception as e:
            print(f"Collection already exists or creation failed: {e}")
        
        # Initialize retrieval pipeline
        retrieval_pipeline = RetrievalPipeline(
            vector_db_client=vector_db,
            embedding_model=query_embedder,
            llm_client=llm_client
        )
        
        # Build BM25 index for hybrid search
        print("Building BM25 index...")
        try:
            retrieval_pipeline.search_engine.build_bm25_index()
            print("✓ BM25 index built")
        except Exception as e:
            print(f"⚠ BM25 index build failed (will use dense search only): {e}")
        
        # Initialize answer generator
        answer_generator = AnswerGenerator(llm_client)
        
        print("✓ Faculty Part API started successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("Make sure Qdrant is running and .env is configured")
        import traceback
        traceback.print_exc()


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
    Query faculty resources using semantic RAG with structured JSON output.
    
    Pipeline:
    1. Intent classification
    2. Hybrid search (vector + BM25)
    3. Cross-encoder reranking
    4. Intent-based chunk limiting
    5. Structured JSON generation
    """
    if not retrieval_pipeline or not answer_generator:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check server logs."
        )
    
    # Validate input
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    if len(request.query) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Query too long (max 10000 characters)"
        )
    
    if request.top_k and (request.top_k < 1 or request.top_k > 50):
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 50"
        )
    
    try:
        # Retrieve relevant chunks
        retrieval_result = retrieval_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k or 15
        )
        
        # Check if any chunks were retrieved
        if not retrieval_result["chunks"]:
            return QueryResponse(
                answer=json.dumps({
                    "intent": "general",
                    "title": "No Results Found",
                    "subtitle": None,
                    "sections": [],
                    "footer": None,
                    "confidence": "none",
                    "fallback": "I couldn't find any relevant information for your query. Please try rephrasing or contact support."
                }),
                sources=[],
                intent=retrieval_result["intent"],
                chunks_used=0,
                metadata={
                    **retrieval_result["metadata"],
                    "domain": retrieval_result.get("domain", "general"),
                    "entities": retrieval_result.get("entities", [])
                }
            )
        
        # Generate structured JSON answer
        answer_result = answer_generator.generate(
            query=request.query,
            retrieved_chunks=retrieval_result["chunks"],
            intent_type=retrieval_result["intent"]
        )
        
        return QueryResponse(
            answer=json.dumps(answer_result["structured"]),
            sources=answer_result["sources"],
            intent=retrieval_result["intent"],
            chunks_used=answer_result["chunks_used"],
            metadata={
                **retrieval_result["metadata"],
                "domain": retrieval_result.get("domain", "general"),
                "entities": retrieval_result.get("entities", []),
                "structured": answer_result["structured"]
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error processing query: {str(e)}")
        print(traceback.format_exc())
        
        # Return user-friendly error
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again or contact support if the issue persists."
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


# Serve frontend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_path = Path(__file__).parent.parent.parent / "frontend"

@app.get("/chat")
async def serve_chat():
    """Serve the chat interface."""
    return FileResponse(str(frontend_path / "index.html"))

app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
