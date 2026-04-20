"""
Conversation history management with persistence.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import uuid


# --- Follow-up coreference rewriting ---------------------------------------
#
# A "follow-up" is a query that depends on the previous turn for its meaning.
# We detect them with two cheap, deterministic heuristics so a stand-alone
# retrieval call doesn't lose context:
#
#   1. Anchor cue list — opening phrases that almost always mean "continue
#      the previous topic" ("what about", "and", "also", "tell me more"…).
#   2. Pronoun-only / very-short messages — a 1-3 word question that contains
#      a bare pronoun ("it", "that", "those") cannot retrieve well on its
#      own and must be expanded with the prior topic.
#
# When either heuristic fires, we extract topic anchors from the previous
# user message AND the previous assistant message (form codes like
# HR-LA-01, capitalized noun phrases, quoted strings) and prepend the
# strongest anchor to the rewritten query. We deliberately avoid calling the
# LLM here: this runs on every query and the latency budget is tight.

_FOLLOWUP_OPENERS = (
    "what about", "how about", "and ", "also ", "what if",
    "tell me more", "more on", "more about",
    "expand on", "elaborate", "go deeper", "explain that",
    "why", "how come", "and what", "and how",
    "what's that", "what is that",
    "is it", "are they", "does it", "do they", "can it", "can they",
)

# Bare pronouns whose presence — without a clear antecedent in the same
# sentence — usually signals a follow-up.
_PRONOUNS = re.compile(
    r"\b(it|its|that|this|these|those|them|they|their|theirs|he|she|him|her|hers|his)\b",
    re.IGNORECASE,
)

# Anchors we mine out of prior turns.  Order = priority.
_ANCHOR_PATTERNS = [
    # Form / policy codes — most specific, always worth carrying forward.
    (re.compile(r"\b([A-Z]{2,3}-[A-Z]{1,3}-\d{1,3})\b"), "code"),
    # Section letters in their canonical form.
    (re.compile(r"\bSECTION\s+[A-Z]\b"), "section"),
    # Quoted phrases — explicit topic from the user.
    (re.compile(r'["“]([^"”]{3,80})["”]'), "quoted"),
    # Capitalised multi-word noun phrases (Faculty Academic Guidelines, etc).
    (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})\b"), "noun"),
]

_STOP_NOUNS = {
    "I", "You", "We", "They", "He", "She", "It", "The", "A", "An", "Is", "Are",
    "Was", "Were", "Do", "Does", "Did", "Have", "Has", "Had", "Will", "Would",
    "Can", "Could", "Should", "May", "Might", "Yes", "No",
}


class ConversationManager:
    """
    Manage conversation history with persistence.
    
    Features:
    - Session-based conversation tracking
    - Persistent storage (JSON files)
    - Context window management
    - Conversation summarization
    """
    
    def __init__(
        self,
        storage_dir: str = "./conversations",
        max_history: int = 10
    ):
        """
        Initialize conversation manager.
        
        Args:
            storage_dir: Directory for conversation storage
            max_history: Max messages to keep in context
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache of active sessions
        self.active_sessions: Dict[str, List[Dict]] = {}
    
    def create_session(self) -> str:
        """Create new conversation session."""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = []
        self.logger.info(f"Created session: {session_id}")
        return session_id
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add message to conversation.
        
        Args:
            session_id: Session ID
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (sources, intent, etc.)
        """
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.active_sessions[session_id].append(message)
        
        # Trim if exceeds max history
        if len(self.active_sessions[session_id]) > self.max_history * 2:
            self.active_sessions[session_id] = self.active_sessions[session_id][-(self.max_history * 2):]
        
        # Persist to disk
        self._save_session(session_id)
    
    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history.
        
        Args:
            session_id: Session ID
            limit: Max messages to return (None = all)
        
        Returns:
            List of messages
        """
        if session_id not in self.active_sessions:
            # Try loading from disk
            self._load_session(session_id)
        
        history = self.active_sessions.get(session_id, [])
        
        if limit:
            return history[-limit:]
        return history
    
    def get_context(
        self,
        session_id: str,
        max_messages: int = 5
    ) -> str:
        """
        Get formatted context for LLM.
        
        Args:
            session_id: Session ID
            max_messages: Max recent messages to include
        
        Returns:
            Formatted context string
        """
        history = self.get_history(session_id, limit=max_messages)
        
        if not history:
            return ""
        
        context_parts = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)

    def clear_session(self, session_id: str) -> None:
        """Clear conversation session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Delete from disk
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        
        self.logger.info(f"Cleared session: {session_id}")
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all conversation sessions."""
        sessions = []
        
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                sessions.append({
                    "session_id": session_file.stem,
                    "message_count": len(data.get("messages", [])),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                })
            except Exception as e:
                self.logger.debug(f"Failed to load session {session_file}: {e}")
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def _save_session(self, session_id: str) -> None:
        """Save session to disk."""
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            
            data = {
                "session_id": session_id,
                "messages": self.active_sessions[session_id],
                "created_at": self.active_sessions[session_id][0]["timestamp"] if self.active_sessions[session_id] else datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save session {session_id}: {e}")
    
    def _load_session(self, session_id: str) -> None:
        """Load session from disk."""
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            
            if session_file.exists():
                with open(session_file, 'r') as f:
                    data = json.load(f)
                
                self.active_sessions[session_id] = data.get("messages", [])
                self.logger.info(f"Loaded session: {session_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            self.active_sessions[session_id] = []

    # ---- Follow-up / coreference rewriting ---------------------------------

    def _is_followup(self, query: str) -> bool:
        """Heuristic: does this query depend on the previous turn?

        Fires when the query starts with a canonical follow-up opener, is
        very short AND contains a bare pronoun, or is short AND has no
        substantive noun of its own.
        """
        q = query.strip().lower()
        if not q:
            return False

        for opener in _FOLLOWUP_OPENERS:
            if q.startswith(opener):
                return True

        # Short queries with pronouns almost always need context.
        word_count = len(q.split())
        if word_count <= 6 and _PRONOUNS.search(q):
            return True

        # Very short queries ("section B?", "the form?") with no proper noun
        # are almost always follow-ups too.
        if word_count <= 3 and not re.search(r"\b[A-Z]{2,}", query):
            return True

        return False

    @staticmethod
    def _extract_anchors(text: str) -> List[Tuple[str, str]]:
        """Pull ordered (anchor_text, anchor_type) pairs out of a message.

        Scans _ANCHOR_PATTERNS in priority order and returns unique hits.
        Type labels match the patterns ("code", "section", "quoted", "noun")
        so callers can prefer higher-specificity anchors.
        """
        found: List[Tuple[str, str]] = []
        seen: set = set()
        for pat, label in _ANCHOR_PATTERNS:
            for m in pat.finditer(text or ""):
                val = m.group(1) if m.groups() else m.group(0)
                val = val.strip()
                if not val or val in _STOP_NOUNS or val.lower() in seen:
                    continue
                seen.add(val.lower())
                found.append((val, label))
        return found

    def _best_anchor(self, session_id: str) -> Optional[str]:
        """Walk the session history backwards to find the most specific
        anchor (form code > section > quoted > noun phrase).
        """
        history = self.get_history(session_id)
        if not history:
            return None

        # Walk from most-recent to oldest, scanning both user queries and
        # assistant answers. Return the first highest-priority hit we find.
        priority = {"code": 0, "section": 1, "quoted": 2, "noun": 3}
        best: Optional[Tuple[int, str]] = None
        for msg in reversed(history):
            anchors = self._extract_anchors(msg.get("content", ""))
            for val, label in anchors:
                rank = priority.get(label, 99)
                if best is None or rank < best[0]:
                    best = (rank, val)
                    if rank == 0:
                        return val  # form code — can't beat this
        return best[1] if best else None

    def rewrite_followup_query(
        self,
        session_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """Rewrite a user query for retrieval, resolving references to
        prior turns when the query is a follow-up.

        Returns a dict with:
          - ``rewritten``: the query to send to the retriever (same as
            ``query`` when no rewriting was applied).
          - ``is_followup``: whether the follow-up heuristic matched.
          - ``anchor``: the topic anchor that was merged in (None otherwise).
          - ``reason``: short diagnostic string for logging.

        The rewrite is deliberately conservative: when we can't find a
        good anchor we return the original query unchanged rather than
        risk injecting noise.
        """
        if not query or not query.strip():
            return {
                "rewritten": query,
                "is_followup": False,
                "anchor": None,
                "reason": "empty_query",
            }

        if not self._is_followup(query):
            return {
                "rewritten": query,
                "is_followup": False,
                "anchor": None,
                "reason": "not_followup",
            }

        anchor = self._best_anchor(session_id)
        if not anchor:
            return {
                "rewritten": query,
                "is_followup": True,
                "anchor": None,
                "reason": "no_anchor_found",
            }

        # If the anchor already appears (case-insensitive) in the query we
        # don't need to rewrite.
        if anchor.lower() in query.lower():
            return {
                "rewritten": query,
                "is_followup": True,
                "anchor": anchor,
                "reason": "anchor_already_present",
            }

        # Substitute a leading bare pronoun with the anchor where possible.
        # Otherwise append "regarding <anchor>" so the retriever still gets
        # the topic signal without distorting the user's intent.
        pronoun_sub = re.sub(
            r"^(it|that|this|they|those|these|them)\b",
            anchor,
            query.strip(),
            count=1,
            flags=re.IGNORECASE,
        )
        if pronoun_sub != query.strip():
            rewritten = pronoun_sub
            reason = "pronoun_substitution"
        else:
            rewritten = f"{query.strip()} (regarding {anchor})"
            reason = "anchor_appended"

        return {
            "rewritten": rewritten,
            "is_followup": True,
            "anchor": anchor,
            "reason": reason,
        }
