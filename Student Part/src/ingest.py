from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import re

from .config import ensure_project_dirs, get_settings
from .catalog import selected_question_paper_pdfs, selected_syllabus_pdfs
from .analysis import is_instruction_text
from .local_ocr import LocalOcr
from .pdf_processing import PageText, extract_pdf_pages, list_pdfs
from .vector_store import Chunk, VectorStore


PAGE_CACHE_FILE = "page_cache.json"
CACHE_VERSION = "local_pdf_text_rapidocr_v1"


def build_index(
    force: bool = False,
    semester_id: str | None = None,
    subject_ids: list[str] | None = None,
) -> dict[str, int]:
    settings = get_settings()
    ensure_project_dirs(settings)
    ocr = LocalOcr()
    page_cache = PageCache.load(settings.index_dir / PAGE_CACHE_FILE)

    syllabus_pdfs = selected_syllabus_pdfs(settings, semester_id)
    all_question_papers = selected_question_paper_pdfs(
        settings,
        semester_id,
        subject_ids,
    )
    question_papers = filter_question_papers_by_year(
        all_question_papers,
        settings.min_question_paper_year,
    )
    skipped_old_pdfs = len(all_question_papers) - len(question_papers)

    pdf_jobs = [
        *(("syllabus", path) for path in syllabus_pdfs),
        *(("question_paper", path) for path in question_papers),
    ]

    all_pages: list[PageText] = []
    cached_pdfs = 0
    processed_pdfs = 0
    for document_type, path in pdf_jobs:
        cached_pages = page_cache.get(path, document_type)
        if cached_pages and not force:
            cached_pdfs += 1
            print(f"Using cached text for {document_type}: {path}")
            all_pages.extend(cached_pages)
            continue

        processed_pdfs += 1
        print(f"Reading {document_type}: {path}")
        pages = extract_pdf_pages(path, document_type, settings, ocr)
        page_cache.set(path, document_type, pages)
        all_pages.extend(pages)

    chunk_payloads = chunk_pages(all_pages, settings.chunk_size, settings.chunk_overlap)
    chunks: list[Chunk] = []
    for payload in chunk_payloads:
        chunks.append(
            Chunk(
                id=payload["id"],
                text=payload["text"],
                metadata=payload["metadata"],
            )
        )
    store = VectorStore.from_chunks(chunks) if chunks else VectorStore()

    all_current_pdfs = {
        str(path)
        for path in [
            *list_pdfs(settings.syllabus_dir),
            *list_pdfs(settings.question_papers_dir),
        ]
    }
    page_cache.prune(all_current_pdfs)
    page_cache.save(settings.index_dir / PAGE_CACHE_FILE)
    store.save(settings.index_path)
    stats = {
        "pdfs": len(pdf_jobs),
        "syllabus_pdfs": len(syllabus_pdfs),
        "question_papers": len(question_papers),
        "skipped_old_question_papers": skipped_old_pdfs,
        "min_question_paper_year": settings.min_question_paper_year,
        "semester_id": semester_id,
        "selected_subjects": subject_ids or [],
        "processed_pdfs": processed_pdfs,
        "cached_pdfs": cached_pdfs,
        "pages": len(all_pages),
        "chunks": len(store.chunks),
        "index_format": "local_tfidf_joblib",
    }
    print(f"Saved index to {settings.index_path}")
    print(stats)
    return stats


def filter_question_papers_by_year(paths: list[Path], min_year: int) -> list[Path]:
    eligible: list[Path] = []
    for path in paths:
        year = question_paper_year(path)
        if year is not None and year >= min_year:
            eligible.append(path)
    return eligible


def question_paper_year(path: Path) -> int | None:
    years = extract_years(str(path))
    return max(years) if years else None


def extract_years(text: str) -> list[int]:
    years: set[int] = set()

    for match in re.finditer(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text):
        years.add(int(match.group(1)))

    for match in re.finditer(r"(?<!\d)((?:19|20)\d{2})\s*[-_/]\s*(\d{2})(?!\d)", text):
        start_year = int(match.group(1))
        suffix = int(match.group(2))
        century = start_year // 100 * 100
        end_year = century + suffix
        if end_year < start_year:
            end_year += 100
        years.add(start_year)
        years.add(end_year)

    for match in re.finditer(r"(?<!\d)(2[0-9])\s*[-_/]\s*(2[0-9])(?!\d)", text):
        years.add(2000 + int(match.group(1)))
        years.add(2000 + int(match.group(2)))

    return sorted(year for year in years if 1900 <= year <= 2099)


class PageCache:
    def __init__(self, documents: dict[str, dict] | None = None):
        self.documents = documents or {}

    @classmethod
    def load(cls, path: Path) -> "PageCache":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload.get("documents", {}))

    def get(self, path: Path, document_type: str) -> list[PageText] | None:
        key = str(path)
        cached = self.documents.get(key)
        if not cached:
            return None
        if cached.get("cache_version") != CACHE_VERSION:
            return None
        if cached.get("signature") != file_signature(path):
            return None
        if cached.get("document_type") != document_type:
            return None
        return [PageText(**page) for page in cached.get("pages", [])]

    def set(self, path: Path, document_type: str, pages: list[PageText]) -> None:
        self.documents[str(path)] = {
            "cache_version": CACHE_VERSION,
            "signature": file_signature(path),
            "document_type": document_type,
            "pages": [asdict(page) for page in pages],
        }

    def prune(self, live_paths: set[str]) -> None:
        self.documents = {
            path: payload
            for path, payload in self.documents.items()
            if path in live_paths
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": self.documents}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def chunk_pages(
    pages: list[PageText],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        if not page.text.strip():
            continue
        if page.document_type == "question_paper" and is_instruction_text(page.text):
            continue
        page_chunks = split_text(page.text, chunk_size, chunk_overlap)
        for chunk_index, text in enumerate(page_chunks, start=1):
            metadata = asdict(page)
            metadata.pop("text", None)
            metadata["chunk_index"] = chunk_index
            chunk_id = stable_chunk_id(metadata, text)
            chunks.append({"id": chunk_id, "text": text, "metadata": metadata})
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = normalize_text(text)
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            boundary = max(clean.rfind("\n", start, end), clean.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(0, end - chunk_overlap)
    return [chunk for chunk in chunks if chunk]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_chunk_id(metadata: dict, text: str) -> str:
    raw = "|".join(
        [
            str(metadata.get("source_path", "")),
            str(metadata.get("page_number", "")),
            str(metadata.get("chunk_index", "")),
            text[:120],
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the academic RAG index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached page text and reprocess every PDF.",
    )
    parser.add_argument("--semester", help="Semester number to index, e.g. 5")
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject id/path under the selected semester. Can be used multiple times.",
    )
    args = parser.parse_args()
    build_index(force=args.force, semester_id=args.semester, subject_ids=args.subject)


if __name__ == "__main__":
    main()
