"""
Structured per-query logging for the RAG pipeline.

Each query produces a single JSON line containing enough context to rerun
the retrieval/generation offline: the original and rewritten queries, the
chunk IDs + scores at each stage, the generated answer, grounding stats,
and latency breakdowns.

The file is kept intentionally flat (one JSON object per line, no nesting
across lines) so we can stream it into pandas / DuckDB without a parser.

Usage:
    from src.utils.query_logger import QueryLogger

    logger = QueryLogger()                             # default path
    entry = logger.start(query="how do I apply for HR-LA-01?", session_id="abc")
    entry.rewritten_query = "HR-LA-01 application procedure"
    entry.retrieved = [{"chunk_id": ..., "score": ...}, ...]
    entry.reranked  = [...]
    entry.answer    = {...}                            # the structured reply
    entry.grounding = {"ratio": 0.83, ...}
    entry.latency_ms = {"retrieval": 120, "rerank": 40, "generation": 900}
    logger.finalize(entry)                             # writes one JSONL line
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_LOG_PATH = Path("data/logs/retrieval.jsonl")


@dataclass
class QueryLogEntry:
    """Single row in the retrieval log.

    Only ``query_id``, ``timestamp``, and ``query`` are required up front;
    every other field is filled in as the pipeline progresses. Sticking to
    plain types (dict/list/str/int/float) keeps the JSONL portable across
    Python versions and pandas / DuckDB without any custom decoder.
    """
    query_id: str
    timestamp: str
    query: str
    session_id: Optional[str] = None
    intent: Optional[str] = None
    rewritten_query: Optional[str] = None
    followup_anchor: Optional[str] = None
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    reranked: List[Dict[str, Any]] = field(default_factory=list)
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[Dict[str, Any]] = None
    grounding: Optional[Dict[str, Any]] = None
    confidence: Optional[str] = None
    chunks_used: Optional[int] = None
    latency_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # Non-persisted: start time for latency accounting.
    _started_at: float = field(default=0.0, repr=False)


class QueryLogger:
    """Append-only JSONL logger with safe concurrent writes.

    We use a module-level lock so multiple threads (FastAPI request workers)
    don't interleave partial lines. The write volume is tiny compared to
    retrieval cost, so the lock is not a bottleneck.
    """

    _lock = threading.Lock()

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def start(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> QueryLogEntry:
        """Begin a new log entry and stamp the start time for latency."""
        return QueryLogEntry(
            query_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            query=query,
            session_id=session_id,
            _started_at=time.perf_counter(),
        )

    def mark(self, entry: QueryLogEntry, stage: str) -> None:
        """Record time elapsed since ``start`` (or previous mark) as a stage.

        Example: ``logger.mark(entry, "retrieval")`` stores the ms since the
        last marked time (or since start) under ``latency_ms["retrieval"]``.
        """
        now = time.perf_counter()
        previous = sum(entry.latency_ms.values()) / 1000.0
        elapsed_total = now - entry._started_at
        stage_ms = max(0.0, (elapsed_total - previous) * 1000.0)
        entry.latency_ms[stage] = round(stage_ms, 2)

    def attach_retrieval(
        self,
        entry: QueryLogEntry,
        chunks: List[Dict[str, Any]],
        stage: str = "retrieved",
    ) -> None:
        """Summarise a list of chunks into compact log rows.

        Only the fields useful for offline debugging go in: chunk_id, score,
        document_name, section_title, and first 120 chars of text. Embedding
        vectors and long payloads are deliberately excluded to keep the log
        small and grep-friendly.
        """
        compact: List[Dict[str, Any]] = []
        for c in chunks:
            md = c.get("metadata", {}) or {}
            compact.append({
                "chunk_id": c.get("chunk_id") or md.get("chunk_id"),
                "score": round(float(c.get("score", 0.0) or 0.0), 4),
                "document_name": md.get("document_name"),
                "section_title": md.get("section_title"),
                "chunk_type": md.get("chunk_type"),
                "text_preview": (c.get("text", "") or "")[:120],
            })
        if stage == "retrieved":
            entry.retrieved = compact
        elif stage == "reranked":
            entry.reranked = compact
        else:
            entry.extras[stage] = compact

    def finalize(self, entry: QueryLogEntry) -> None:
        """Write the entry to disk as a single JSON line."""
        # Total latency so far, for convenience.
        if entry._started_at:
            total = (time.perf_counter() - entry._started_at) * 1000.0
            entry.latency_ms.setdefault("total", round(total, 2))

        record = asdict(entry)
        # Strip the private book-keeping field before serialising.
        record.pop("_started_at", None)

        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001
            self.logger.error("query_logger: JSON encode failed: %s", e)
            return

        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:  # noqa: BLE001
                self.logger.error("query_logger: write failed: %s", e)


# Module-level convenience: many callers just want a singleton writing to
# the default path. We lazily construct it so importing this module has no
# filesystem side effects in tests.
_default_logger: Optional[QueryLogger] = None


def get_default_logger() -> QueryLogger:
    """Return a process-wide QueryLogger, honouring QUERY_LOG_PATH env var."""
    global _default_logger
    if _default_logger is None:
        path = os.environ.get("QUERY_LOG_PATH")
        _default_logger = QueryLogger(Path(path)) if path else QueryLogger()
    return _default_logger
