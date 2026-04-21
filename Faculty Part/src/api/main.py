"""
FastAPI application for Faculty Part RAG system.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncIterator
from pathlib import Path
from dotenv import load_dotenv
import json
import asyncio
import os
import bcrypt

# Load environment variables
load_dotenv()

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..retrieval.pipeline import RetrievalPipeline
from ..generation.answer_generator import AnswerGenerator
from ..utils.vector_db import VectorDBClient
from ..utils.query_embedder import QueryEmbedder
from ..utils.llm import LLMClient
from ..utils.cache_manager import CacheManager
from ..utils.conversation_manager import ConversationManager
from ..utils.rate_limiter import RateLimiter

# WARNING: This is a demo auth system for development only.
# Credentials loaded from .env file with bcrypt hashing.
# Replace with proper database-backed authentication before production.

def load_demo_users() -> Dict[str, Dict[str, str]]:
    """Load demo users from environment variables."""
    users = {}
    
    for key in ['DEMO_USER_STUDENT', 'DEMO_USER_FACULTY', 'DEMO_USER_ADMIN']:
        user_data = os.getenv(key)
        if user_data:
            try:
                email, password_hash, role, name = user_data.split(':')
                users[email.lower().strip()] = {
                    "password_hash": password_hash,
                    "role": role,
                    "name": name
                }
            except ValueError:
                print(f"Warning: Invalid format for {key}")
    
    return users

DEMO_USERS = load_demo_users()


app = FastAPI(
    title="ClassAI Faculty Part",
    description="AI-powered faculty resource assistant with semantic RAG",
    version="0.1.0"
)

# CORS middleware - configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://localhost:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Faculty query request."""
    query: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = None
    stream: Optional[bool] = False
    format_override: Optional[Dict[str, str]] = None  # {"verbosity": "...", "structure": "..."}


class SignInRequest(BaseModel):
    """Sign-in request — role is NOT accepted from client (A2)."""
    email: str
    password: str


class SignInResponse(BaseModel):
    """Sign-in response."""
    token: str
    role: str
    user: Dict[str, str]


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
cache_manager = None
conversation_manager = None
rate_limiter = None
llm_semaphore = None  # Limit concurrent LLM calls


@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global retrieval_pipeline, answer_generator, cache_manager, conversation_manager, rate_limiter, llm_semaphore
    
    try:
        # Initialize components
        print("Initializing Faculty Part RAG system...")
        
        # Initialize rate limiter (20 req/min per IP)
        rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
        
        # Initialize LLM semaphore (max 3 concurrent LLM calls)
        llm_semaphore = asyncio.Semaphore(3)
        
        # Initialize cache manager
        cache_manager = CacheManager(
            disk_cache_dir="./cache",
            enable_redis=False  # Set to True if Redis is available
        )
        
        # Initialize conversation manager
        conversation_manager = ConversationManager(
            storage_dir="./conversations",
            max_history=10
        )
        
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
        
        # Build BM25 index for hybrid search (with persistence)
        print("Loading BM25 index...")
        try:
            retrieval_pipeline.search_engine.build_bm25_index()
            print("✓ BM25 index ready")
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
    """Serve landing page."""
    return FileResponse(str(frontend_path / "index.html"))


@app.post("/api/auth/signin", response_model=SignInResponse)
async def signin(request: SignInRequest):
    """
    Sign in endpoint with role-based access and bcrypt password verification.
    
    WARNING: This is a demo implementation. For production:
    - Use database for user storage (SQLite minimum)
    - Generate proper JWT tokens with expiry
    - Implement refresh tokens
    - Add rate limiting
    """
    email = request.email.lower().strip()
    password = request.password
    
    # Check if user exists
    if email not in DEMO_USERS:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    
    user_data = DEMO_USERS[email]
    
    # Verify password with bcrypt
    try:
        password_hash = user_data["password_hash"].encode('utf-8')
        if not bcrypt.checkpw(password.encode('utf-8'), password_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
    except Exception as e:
        print(f"Password verification error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    
    # A2: role is NOT accepted from the client — the server authoritatively
    # returns the role stored with the user record.
    
    # Generate token (in production, use JWT with expiry)
    import secrets
    token = secrets.token_urlsafe(32)
    
    return SignInResponse(
        token=token,
        role=user_data["role"],
        user={
            "name": user_data["name"],
            "email": email
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query_faculty_resources(request: QueryRequest, req: Request):
    """
    Query faculty resources using semantic RAG with structured JSON output.
    
    Pipeline:
    1. Rate limiting check
    2. Check cache for recent identical queries
    3. Intent classification
    4. Hybrid search (vector + BM25)
    5. Cross-encoder reranking
    6. Intent-based chunk limiting
    7. Structured JSON generation (with LLM semaphore)
    8. Save to conversation history
    """
    # ── Sanitize and log query ──────────────────────────────────────
    import re
    raw_query = request.query or ""
    query = re.sub(r'\s+', ' ', raw_query).strip()
    
    print(f"\n{'='*50}")
    print(f"[QUERY] Raw: '{raw_query[:80]}'")
    print(f"[QUERY] Clean: '{query[:80]}'")
    print(f"[QUERY] Session: {request.session_id}")
    print(f"[QUERY] Stream: {request.stream}")
    print(f"{'='*50}")
    
    if not retrieval_pipeline or not answer_generator:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check server logs."
        )
    
    # Rate limiting check
    client_ip = req.client.host
    is_allowed, retry_after = await rate_limiter.is_allowed(client_ip)
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Validate input
    if not query or len(query) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    if len(query) < 2:
        raise HTTPException(
            status_code=400,
            detail="Query too short (minimum 2 characters)"
        )
    
    if len(query) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Query too long (max 10000 characters)"
        )
    
    if request.top_k and (request.top_k < 1 or request.top_k > 50):
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 50"
        )
    
    # Update request with sanitized query
    request.query = query
    
    # Handle streaming request
    if request.stream:
        return StreamingResponse(
            stream_query_response(request),
            media_type="text/event-stream"
        )
    
    try:
        # Check cache first
        cached_result = cache_manager.get_query_result(request.query, request.top_k or 15)
        if cached_result:
            print(f"Cache hit for query: {request.query[:50]}...")
            
            # Add to conversation history
            if request.session_id:
                conversation_manager.add_message(
                    request.session_id,
                    "user",
                    request.query
                )
                conversation_manager.add_message(
                    request.session_id,
                    "assistant",
                    cached_result["answer"],
                    metadata={
                        "sources": cached_result["sources"],
                        "intent": cached_result["intent"],
                        "cached": True
                    }
                )
            
            return QueryResponse(**cached_result)
        
        print(f"[PIPELINE] Starting retrieval for: {request.query[:50]}...")
        
        # Retrieve relevant chunks
        retrieval_result = await asyncio.to_thread(
            retrieval_pipeline.retrieve,
            query=request.query,
            top_k=request.top_k or 15
        )
        
        print(f"[PIPELINE] Retrieval complete. Found {len(retrieval_result['chunks'])} chunks")
        
        # Apply format override if provided
        format_preference = retrieval_result.get("format_preference")
        if request.format_override:
            from ..retrieval.query_understanding import FormatPreference
            
            # Validate and apply override
            verbosity = request.format_override.get("verbosity", "standard")
            structure = request.format_override.get("structure", "auto")
            
            # Validate values
            valid_verbosity = ["brief", "standard", "detailed"]
            valid_structure = ["auto", "paragraph", "bullets", "steps", "table"]
            
            if verbosity not in valid_verbosity:
                print(f"[WARNING] Invalid verbosity override: {verbosity}, ignoring")
                verbosity = "standard"
            
            if structure not in valid_structure:
                print(f"[WARNING] Invalid structure override: {structure}, ignoring")
                structure = "auto"
            
            format_preference = FormatPreference(
                verbosity=verbosity,  # type: ignore
                structure=structure,  # type: ignore
                verbosity_trigger="API override",
                structure_trigger="API override"
            )
            print(f"[FORMAT] Override applied: verbosity={verbosity}, structure={structure}")
        
        # Check if any chunks were retrieved
        if not retrieval_result["chunks"]:
            fallback_response = {
                "answer": json.dumps({
                    "intent": "general",
                    "title": "No Results Found",
                    "subtitle": None,
                    "sections": [],
                    "footer": None,
                    "confidence": "none",
                    "fallback": "I couldn't find any relevant information for your query. Please try rephrasing or contact support."
                }),
                "sources": [],
                "intent": retrieval_result["intent"],
                "chunks_used": 0,
                "metadata": {
                    **retrieval_result["metadata"],
                    "domain": retrieval_result.get("domain", "general"),
                    "entities": retrieval_result.get("entities", []),
                    "processing_steps": []
                }
            }
            return QueryResponse(**fallback_response)
        
        # Generate structured JSON answer with LLM semaphore
        async with llm_semaphore:
            answer_result = await asyncio.to_thread(
                answer_generator.generate,
                query=request.query,
                retrieved_chunks=retrieval_result["chunks"],
                intent_type=retrieval_result["intent"],
                format_preference=format_preference
            )
        
        # Check confidence and handle low-confidence responses
        confidence = answer_result.get("confidence", "high")
        if confidence == "low" or confidence == "none":
            suggestions = retrieval_result["metadata"].get("reformulation_suggestions", [])
            fallback_response = {
                "answer": json.dumps({
                    "intent": retrieval_result["intent"],
                    "title": "Unable to Find Reliable Information",
                    "subtitle": None,
                    "sections": [{
                        "heading": "Suggestions",
                        "type": "bullets",
                        "items": suggestions if suggestions else ["Try being more specific", "Include relevant keywords"]
                    }],
                    "footer": "If you need immediate assistance, please contact support directly.",
                    "confidence": "none",
                    "fallback": "I couldn't find reliable information to answer your question."
                }),
                "sources": [],
                "intent": retrieval_result["intent"],
                "chunks_used": 0,
                "metadata": {**retrieval_result["metadata"], "confidence_level": confidence}
            }
            return QueryResponse(**fallback_response)
        
        # Add processing steps to metadata
        processing_steps = [
            {"step": "Query Understanding", "status": "completed", "details": f"Intent: {retrieval_result['intent']}, Domain: {retrieval_result.get('domain', 'general')}"},
            {"step": "Document Retrieval", "status": "completed", "details": f"Found {len(retrieval_result['chunks'])} relevant chunks"},
            {"step": "Answer Generation", "status": "completed", "details": f"Generated {retrieval_result['intent']} response"}
        ]
        
        result = {
            "answer": json.dumps(answer_result["structured"]),
            "sources": answer_result["sources"],
            "intent": retrieval_result["intent"],
            "chunks_used": answer_result["chunks_used"],
            "metadata": {
                **retrieval_result["metadata"],
                "domain": retrieval_result.get("domain", "general"),
                "entities": retrieval_result.get("entities", []),
                "structured": answer_result["structured"],
                "processing_steps": processing_steps
            }
        }
        
        # Cache the result
        cache_manager.set_query_result(request.query, request.top_k or 15, result, ttl=3600)
        
        # Add to conversation history
        if request.session_id:
            conversation_manager.add_message(
                request.session_id,
                "user",
                request.query
            )
            conversation_manager.add_message(
                request.session_id,
                "assistant",
                result["answer"],
                metadata={
                    "sources": result["sources"],
                    "intent": result["intent"]
                }
            )
        
        return QueryResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("=" * 80)
        print("CRITICAL ERROR IN QUERY ENDPOINT")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        print("=" * 80)
        
        # Return user-friendly error
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again or contact support if the issue persists."
        )


@app.get("/health")
async def health_check():
    """Detailed health check."""
    cache_stats = cache_manager.get_stats() if cache_manager else {}
    
    return {
        "status": "healthy",
        "service": "ClassAI Faculty Part",
        "version": "0.1.0",
        "components": {
            "retrieval_pipeline": retrieval_pipeline is not None,
            "answer_generator": answer_generator is not None,
            "cache_manager": cache_manager is not None,
            "conversation_manager": conversation_manager is not None,
        },
        "cache_stats": cache_stats
    }


async def stream_query_response(request: QueryRequest) -> AsyncIterator[str]:
    """
    Stream query response as Server-Sent Events.
    
    Yields:
        SSE formatted events with progressive response
    """
    try:
        print(f"[SSE] Starting stream for: '{request.query[:60]}'")
        
        # Send initial event
        yield f"event: status\ndata: {json.dumps({'step': 'understanding', 'message': 'Analyzing query...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Check cache
        cached_result = cache_manager.get_query_result(request.query, request.top_k or 15)
        if cached_result:
            print(f"[SSE] Cache hit")
            yield f"event: status\ndata: {json.dumps({'step': 'cache_hit', 'message': 'Found cached result'})}\n\n"
            yield f"event: result\ndata: {json.dumps(cached_result)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        
        # Retrieve chunks
        print(f"[SSE] Starting retrieval")
        yield f"event: status\ndata: {json.dumps({'step': 'retrieval', 'message': 'Searching knowledge base...'})}\n\n"
        
        retrieval_result = await asyncio.to_thread(
            retrieval_pipeline.retrieve,
            query=request.query,
            top_k=request.top_k or 15
        )
        
        print(f"[SSE] Retrieval complete: {len(retrieval_result['chunks'])} chunks")
        
        # Apply format override if provided
        format_preference = retrieval_result.get("format_preference")
        if request.format_override:
            from ..retrieval.query_understanding import FormatPreference
            
            verbosity = request.format_override.get("verbosity", "standard")
            structure = request.format_override.get("structure", "auto")
            
            valid_verbosity = ["brief", "standard", "detailed"]
            valid_structure = ["auto", "paragraph", "bullets", "steps", "table"]
            
            if verbosity not in valid_verbosity:
                verbosity = "standard"
            if structure not in valid_structure:
                structure = "auto"
            
            format_preference = FormatPreference(
                verbosity=verbosity,  # type: ignore
                structure=structure,  # type: ignore
                verbosity_trigger="API override",
                structure_trigger="API override"
            )
        
        if not retrieval_result["chunks"]:
            print(f"[SSE] No chunks found")
            yield f"event: error\ndata: {json.dumps({'message': 'No relevant documents found'})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        
        # Generate answer
        print(f"[SSE] Starting generation")
        yield f"event: status\ndata: {json.dumps({'step': 'generation', 'message': 'Generating answer...'})}\n\n"
        
        answer_result = await asyncio.to_thread(
            answer_generator.generate,
            query=request.query,
            retrieved_chunks=retrieval_result["chunks"],
            intent_type=retrieval_result["intent"],
            format_preference=format_preference
        )
        
        print(f"[SSE] Generation complete")
        
        # Send result
        result = {
            "answer": json.dumps(answer_result["structured"]),
            "sources": answer_result["sources"],
            "intent": retrieval_result["intent"],
            "chunks_used": answer_result["chunks_used"],
            "metadata": retrieval_result["metadata"]
        }
        
        # Cache it
        cache_manager.set_query_result(request.query, request.top_k or 15, result, ttl=3600)
        
        # Add to conversation
        if request.session_id:
            conversation_manager.add_message(request.session_id, "user", request.query)
            conversation_manager.add_message(
                request.session_id,
                "assistant",
                result["answer"],
                metadata={"sources": result["sources"], "intent": result["intent"]}
            )
        
        yield f"event: result\ndata: {json.dumps(result)}\n\n"
        yield "event: done\ndata: {}\n\n"
        print(f"[SSE] Stream complete")
        
    except Exception as e:
        import traceback
        print("=" * 80)
        print("[SSE ERROR] Exception in stream handler")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        print("=" * 80)
        
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        yield "event: done\ndata: {}\n\n"


@app.post("/conversation/new")
async def create_conversation():
    """Create new conversation session."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="Conversation manager not initialized")
    
    session_id = conversation_manager.create_session()
    return {"session_id": session_id}


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str, limit: Optional[int] = None):
    """Get conversation history."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="Conversation manager not initialized")
    
    history = conversation_manager.get_history(session_id, limit)
    return {"session_id": session_id, "messages": history}


@app.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """Delete conversation session."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="Conversation manager not initialized")
    
    conversation_manager.clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/conversations")
async def list_conversations():
    """List all conversation sessions."""
    if not conversation_manager:
        raise HTTPException(status_code=503, detail="Conversation manager not initialized")
    
    sessions = conversation_manager.list_sessions()
    return {"sessions": sessions}


# Serve frontend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_path = Path(__file__).parent.parent.parent / "frontend"

@app.get("/app")
async def serve_app():
    """Serve the chat interface (deprecated - use /chat)."""
    return FileResponse(str(frontend_path / "chat.html"))


@app.get("/chat")
async def serve_chat():
    """Serve the chat interface."""
    return FileResponse(str(frontend_path / "chat.html"))


@app.get("/signin")
async def serve_signin():
    """Serve the sign-in page."""
    return FileResponse(str(frontend_path / "signin.html"))


app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
