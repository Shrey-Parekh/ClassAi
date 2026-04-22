from __future__ import annotations

import argparse
from dataclasses import dataclass
import re

from .analysis import build_unit_mapping_answer, is_unit_mapping_query
from .config import get_settings
from .tavily_client import search_web
from .vector_store import Chunk, VectorStore


@dataclass
class RetrievedSource:
    source_name: str
    document_type: str
    page_number: int
    chunk_index: int
    score: float
    text: str


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedSource]
    web_sources: list[dict[str, str]]


def answer_question(question: str, use_web: bool = False, top_k: int | None = None) -> RagAnswer:
    settings = get_settings()
    store = VectorStore.load(settings.index_path)
    if not store.chunks:
        raise RuntimeError(
            f"No index found at {settings.index_path}. Add PDFs and run: python -m src.ingest"
        )

    if is_unit_mapping_query(question):
        answer = build_unit_mapping_answer(store.chunks)
        question_paper_chunks = [
            chunk for chunk in store.chunks if chunk.metadata.get("document_type") == "question_paper"
        ][: settings.top_k]
        sources = [to_source(chunk, 0.0) for chunk in question_paper_chunks]
        return RagAnswer(answer=answer, sources=sources, web_sources=[])

    retrieved = store.search(question, top_k or settings.top_k)
    sources = [to_source(chunk, score) for chunk, score in retrieved]
    web_sources = search_web(question, settings.tavily_api_key) if use_web else []
    answer = build_local_answer(question, sources, web_sources)
    return RagAnswer(answer=answer, sources=sources, web_sources=web_sources)


def to_source(chunk: Chunk, score: float) -> RetrievedSource:
    metadata = chunk.metadata
    return RetrievedSource(
        source_name=str(metadata.get("source_name", "")),
        document_type=str(metadata.get("document_type", "")),
        page_number=int(metadata.get("page_number", 0)),
        chunk_index=int(metadata.get("chunk_index", 0)),
        score=score,
        text=chunk.text,
    )


def build_local_answer(
    question: str,
    sources: list[RetrievedSource],
    web_sources: list[dict[str, str]],
) -> str:
    if not sources:
        return "I could not find matching syllabus or question-paper passages in the local index."

    query_terms = important_terms(question)
    lines = ["Most relevant local matches:"]
    for index, source in enumerate(sources[:5], start=1):
        snippet = best_snippet(source.text, query_terms)
        lines.append(
            f"[{index}] {source.source_name}, page {source.page_number}: {snippet}"
        )

    if web_sources:
        lines.append("")
        lines.append("Optional web matches:")
        for index, source in enumerate(web_sources[:3], start=1):
            content = source.get("content", "").strip()
            title = source.get("title", "Web source").strip()
            lines.append(f"[W{index}] {title}: {content[:280]}")

    return "\n".join(lines)


def important_terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "which",
        "what",
        "are",
        "most",
        "often",
        "show",
        "give",
        "about",
    }
    return {word for word in words if word not in stop}


def best_snippet(text: str, terms: set[str], max_chars: int = 520) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = []
    for sentence in sentences:
        clean = " ".join(sentence.split())
        if not clean:
            continue
        score = sum(1 for term in terms if term in clean.lower())
        scored.append((score, clean))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    snippet = scored[0][1] if scored else " ".join(text.split())
    if len(snippet) > max_chars:
        return snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the academic RAG assistant.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--web", action="store_true", help="Include Tavily web context")
    args = parser.parse_args()
    result = answer_question(args.question, use_web=args.web)
    print(result.answer)
    print("\nSources:")
    for index, source in enumerate(result.sources, start=1):
        print(
            f"[{index}] {source.source_name}, page {source.page_number}, "
            f"score {source.score:.3f}"
        )


if __name__ == "__main__":
    main()
