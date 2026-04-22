from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


class VectorStore:
    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        vectorizer: TfidfVectorizer | None = None,
        matrix=None,
        created_at: str | None = None,
    ):
        self.chunks = chunks or []
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.created_at = created_at

    def add(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "VectorStore":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=250_000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        return cls(
            chunks=chunks,
            vectorizer=vectorizer,
            matrix=matrix,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [(self.chunks[index], float(scores[index])) for index in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(self.chunks),
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not path.exists():
            return cls()
        payload = joblib.load(path)
        chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        return cls(
            chunks=chunks,
            vectorizer=payload.get("vectorizer"),
            matrix=payload.get("matrix"),
            created_at=payload.get("created_at"),
        )
