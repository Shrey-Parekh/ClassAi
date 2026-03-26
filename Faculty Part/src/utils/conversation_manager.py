"""
Conversation history management with persistence.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import uuid


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
