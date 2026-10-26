"""
FastAPI application for Faculty Part RAG system.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncIterator
from pathlib import Path
from dotenv import load_dotenv
import json
import asyncio
import os
import re
import secrets
import traceback
import bcrypt

# Load environment variables
load_dotenv()

# Verify HTTPS against the Windows certificate store, so calls to Groq and
# Hugging Face work behind antivirus or proxy TLS inspection.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from ..retrieval.scope_router import ScopeRouter
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield


app = FastAPI(
    title="ClassAI Faculty Part",
    description="AI-powered faculty resource assistant with semantic RAG",
    version="0.1.0",
    lifespan=lifespan
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
    scope: Optional[str] = None  # "student", "faculty", "both" (for faculty/admin only)
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
FACULTY_DIR = Path(__file__).resolve().parents[2]  # "Faculty Part/", independent of CWD

scope_router = None
answer_generator = None
cache_manager = None
conversation_manager = None
rate_limiter = None
llm_client = None
vector_db = None
llm_semaphore = None  # Limit concurrent LLM calls
active_tokens = {}  # Store active session tokens with role info


async def startup_event():
    """
    Initialize RAG components on startup.

    Any failure here (Qdrant or Ollama down, model missing) propagates so the
    server refuses to start instead of running half-initialised.
    """
    global scope_router, answer_generator, cache_manager, conversation_manager
    global rate_limiter, llm_semaphore, llm_client, vector_db

    print("Initializing ClassAI Unified RAG system...")

    # 20 req/min per IP, at most 3 concurrent LLM calls
    rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
    llm_semaphore = asyncio.Semaphore(3)

    cache_manager = CacheManager(
        disk_cache_dir=str(FACULTY_DIR / "cache"),
        enable_redis=False
    )
    # Collections may have been re-indexed since the last run, so cached
    # answers could cite chunks that no longer exist.
    cache_manager.clear()

    conversation_manager = ConversationManager(
        storage_dir=str(FACULTY_DIR / "conversations"),
        max_history=10
    )

    vector_db = VectorDBClient()
    try:
        vector_db.client.get_collections()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Qdrant at {vector_db.url}. "
            f"Start it with `docker compose up -d` in Faculty Part. ({e})"
        ) from e

    llm_client = LLMClient()  # raises ConnectionError if Ollama is unreachable
    query_embedder = QueryEmbedder(model_name="BAAI/bge-m3")

    faculty_collection = os.getenv("FACULTY_COLLECTION_NAME", "faculty_chunks")
    student_collection = os.getenv("STUDENT_COLLECTION_NAME", "academic_rag")

    # No-op when the collection already exists
    for name in (faculty_collection, student_collection):
        vector_db.create_collection(collection_name=name, vector_size=query_embedder.get_dimension())

    scope_router = ScopeRouter(
        vector_db_client=vector_db,
        embedding_model=query_embedder,
        llm_client=llm_client,
        faculty_collection_name=faculty_collection,
        student_collection_name=student_collection
    )

    print("Loading BM25 indexes...")
    scope_router.faculty_pipeline.search_engine.build_bm25_index()
    scope_router.student_pipeline.search_engine.build_bm25_index()

    answer_generator = AnswerGenerator(llm_client)

    print("✓ ClassAI Unified API started successfully")
    print(f"  Collections: {faculty_collection}, {student_collection}")


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
    token = secrets.token_urlsafe(32)
    
    # Store token with role info for later validation
    active_tokens[token] = {
        "email": email,
        "role": user_data["role"],
        "name": user_data["name"]
    }
    
    return SignInResponse(
        token=token,
        role=user_data["role"],
        user={
            "name": user_data["name"],
            "email": email
        }
    )


@app.get("/api/auth/me")
async def whoami(req: Request):
    """Return the caller's role, or 401 if the token is unknown (e.g. after a server restart)."""
    auth_header = req.headers.get("Authorization", "")
    token = auth_header[len("Bearer "):].strip() if auth_header.startswith("Bearer ") else None
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")
    user = active_tokens[token]
    return {"role": user["role"], "name": user["name"], "email": user["email"]}


@app.post("/api/auth/signout")
async def signout(req: Request):
    """Revoke the caller's token."""
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        active_tokens.pop(auth_header[len("Bearer "):].strip(), None)
    return {"status": "signed_out"}


CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone question that can be understood without the conversation. Keep names, form codes, course names and any formatting request (table, bullets, brief, detailed). If it is already standalone, return it unchanged.

Conversation:
{history}

Follow-up question: {question}

Standalone question (one line, no explanation):"""


def _history_line(message: Dict[str, Any]) -> str:
    """One line per stored message; assistant turns are stored as structured JSON."""
    content = message.get("content", "")
    if message.get("role") == "assistant":
        try:
            answer = json.loads(content)
            content = " - ".join(p for p in (answer.get("title"), answer.get("subtitle")) if p)
        except (ValueError, AttributeError):
            pass
        return f"Assistant: {content[:300]}"
    return f"User: {content[:300]}"


async def _standalone_query(query: str, session_id: Optional[str]) -> str:
    """Resolve follow-ups ("what about unit 3?") against the session's history."""
    history = conversation_manager.get_history(session_id, limit=6) if session_id else []
    if not history:
        return query
    prompt = CONDENSE_PROMPT.format(
        history="\n".join(_history_line(m) for m in history), question=query
    )
    try:
        async with llm_semaphore:
            rewritten = await asyncio.to_thread(
                llm_client.generate, prompt, max_tokens=120, temperature=0.0
            )
    except Exception as e:
        print(f"[CONDENSE] Failed, using original query: {e}")
        return query
    lines = (rewritten or "").strip().splitlines()
    rewritten = lines[0].strip().strip('"') if lines else ""
    if not rewritten or len(rewritten) > 500:
        return query
    if rewritten != query:
        print(f"[CONDENSE] '{query[:60]}' -> '{rewritten[:80]}'")
    return rewritten


def _format_override(request: QueryRequest):
    """Validated FormatPreference from the request, or None."""
    if not request.format_override:
        return None
    from ..retrieval.query_understanding import FormatPreference
    verbosity = request.format_override.get("verbosity", "standard")
    structure = request.format_override.get("structure", "auto")
    if verbosity not in ("brief", "standard", "detailed"):
        verbosity = "standard"
    if structure not in ("auto", "paragraph", "bullets", "steps", "table"):
        structure = "auto"
    return FormatPreference(
        verbosity=verbosity,  # type: ignore
        structure=structure,  # type: ignore
        verbosity_trigger="API override",
        structure_trigger="API override"
    )


def _fallback(intent: str, title: str, message: str, metadata: Dict[str, Any],
              suggestions: Optional[List[str]] = None) -> Dict[str, Any]:
    """Result shaped like a normal answer that the frontend renders as a fallback."""
    sections = []
    if suggestions:
        sections.append({"heading": "Suggestions", "type": "bullets", "items": suggestions})
    return {
        "answer": json.dumps({
            "intent": intent, "title": title, "subtitle": None, "sections": sections,
            "footer": None, "confidence": "none", "fallback": message
        }),
        "sources": [],
        "intent": intent,
        "chunks_used": 0,
        "metadata": metadata
    }


def _status(step: str, message: str) -> tuple:
    return ("status", {"step": step, "message": message})


async def _answer_events(request: QueryRequest, role: str, scope: str):
    """
    The full query pipeline, shared by the JSON and streaming endpoints.

    Yields ("status", {...}) progress events, then exactly one ("result", {...}).
    """
    top_k = request.top_k or 20

    yield _status("understanding", "Analyzing query...")
    query = await _standalone_query(request.query, request.session_id)

    # Role and scope change which collections are searched, so they are part
    # of the key; otherwise switching scope would replay the previous answer.
    cache_key = json.dumps([role, scope, request.format_override, query], sort_keys=True)
    result = cache_manager.get_query_result(cache_key, top_k)

    if result:
        yield _status("cache_hit", "Found cached result")
    else:
        yield _status("retrieval", "Searching knowledge base...")
        retrieval = await asyncio.to_thread(
            scope_router.retrieve, query=query, role=role, scope=scope, top_k=top_k
        )
        print(f"[PIPELINE] Retrieved {len(retrieval['chunks'])} chunks")
        metadata = {
            **retrieval["metadata"],
            "domain": retrieval.get("domain", "general"),
            "entities": retrieval.get("entities", []),
            "standalone_query": query,
        }

        if not retrieval["chunks"]:
            result = _fallback(
                retrieval["intent"], "No Results Found",
                "I couldn't find any relevant information for your query. Please try rephrasing.",
                metadata
            )
        else:
            yield _status("generation", "Generating answer...")
            async with llm_semaphore:
                answer = await asyncio.to_thread(
                    answer_generator.generate,
                    query=query,
                    retrieved_chunks=retrieval["chunks"],
                    intent_type=retrieval["intent"],
                    format_preference=_format_override(request) or retrieval.get("format_preference")
                )

            confidence = answer.get("confidence", "high")
            if confidence == "none":
                result = _fallback(
                    retrieval["intent"], "Unable to Find Reliable Information",
                    "I couldn't find reliable information to answer your question.",
                    {**metadata, "confidence_level": confidence},
                    suggestions=retrieval["metadata"].get("reformulation_suggestions")
                    or ["Try being more specific", "Include relevant keywords"]
                )
            else:
                # "low" answers are shown with the frontend's verify-with-source warning
                structured = {**answer["structured"], "confidence": confidence}
                result = {
                    "answer": json.dumps(structured),
                    "sources": answer["sources"],
                    "intent": retrieval["intent"],
                    "chunks_used": answer["chunks_used"],
                    "metadata": {**metadata, "structured": structured, "confidence_level": confidence}
                }
                cache_manager.set_query_result(cache_key, top_k, result, ttl=3600)

    if request.session_id:
        conversation_manager.add_message(request.session_id, "user", request.query)
        conversation_manager.add_message(
            request.session_id, "assistant", result["answer"],
            metadata={"sources": result["sources"], "intent": result["intent"]}
        )

    yield ("result", result)


@app.post("/query", response_model=QueryResponse)
async def query_faculty_resources(request: QueryRequest, req: Request):
    """
    Query resources using semantic RAG with role-based scope routing.

    The role comes from the server-side token store, never from the client;
    requests without a valid token are treated as student.
    """
    auth_header = req.headers.get("Authorization", "")
    token = auth_header[len("Bearer "):].strip() if auth_header.startswith("Bearer ") else None
    if token and token not in active_tokens:
        # A stale token (tokens live in memory, so any restart drops them) must
        # not silently fall back to student access.
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")
    role = active_tokens[token]["role"] if token else "student"

    query = re.sub(r'\s+', ' ', request.query or "").strip()
    scope = request.scope or "both"
    print(f"\n[QUERY] '{query[:80]}' role={role} scope={scope} "
          f"session={request.session_id} stream={request.stream}")

    is_allowed, retry_after = await rate_limiter.is_allowed(req.client.host)
    if not is_allowed:
        raise HTTPException(status_code=429, detail="Too many requests",
                            headers={"Retry-After": str(retry_after)})

    if len(query) < 2:
        raise HTTPException(status_code=400, detail="Query too short (minimum 2 characters)")
    if len(query) > 10000:
        raise HTTPException(status_code=400, detail="Query too long (max 10000 characters)")
    if request.top_k and (request.top_k < 1 or request.top_k > 50):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    request.query = query

    if request.stream:
        return StreamingResponse(stream_query_response(request, role, scope),
                                 media_type="text/event-stream")

    try:
        async for kind, payload in _answer_events(request, role, scope):
            if kind == "result":
                return QueryResponse(**payload)
    except Exception:
        print("[QUERY ERROR]\n" + traceback.format_exc())
    raise HTTPException(
        status_code=500,
        detail="An error occurred while processing your query. Check the server log."
    )


@app.get("/health")
async def health_check():
    """Live check of Qdrant and the LLM backend; 503 if either is unreachable."""
    checks = {}
    try:
        await asyncio.to_thread(vector_db.client.get_collections)
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
    try:
        await asyncio.to_thread(llm_client.ping)
        checks["llm"] = "ok"
    except Exception as e:
        checks["llm"] = f"error: {e}"

    healthy = all(v == "ok" for v in checks.values())
    body = {
        "status": "healthy" if healthy else "degraded",
        "service": "ClassAI Unified",
        "checks": checks,
        "collections": {
            "faculty": os.getenv("FACULTY_COLLECTION_NAME", "faculty_chunks"),
            "student": os.getenv("STUDENT_COLLECTION_NAME", "academic_rag")
        },
        "cache_stats": cache_manager.get_stats(),
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


async def stream_query_response(request: QueryRequest, role: str, scope: str) -> AsyncIterator[str]:
    """Stream the shared pipeline as Server-Sent Events."""
    try:
        async for kind, payload in _answer_events(request, role, scope):
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
    except Exception:
        print("[SSE ERROR]\n" + traceback.format_exc())
        message = "Something went wrong while answering. Check the server log."
        yield f"event: error\ndata: {json.dumps({'message': message})}\n\n"
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
