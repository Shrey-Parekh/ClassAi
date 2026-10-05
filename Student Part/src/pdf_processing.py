from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

from .config import Settings
from .local_ocr import LocalOcr


@dataclass
class PageText:
    source_path: str
    source_name: str
    document_type: str
    page_number: int
    text: str
    extraction_method: str


def list_pdfs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*.pdf") if item.is_file())


def extract_pdf_pages(
    pdf_path: Path,
    document_type: str,
    settings: Settings,
    ocr: LocalOcr | None = None,
) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            if document_type == "question_paper":
                if ocr is None:
                    raise RuntimeError("Question-paper ingestion requires a local OCR engine.")
                image = render_page(page, settings.render_dpi)
                text = ocr.image_to_text(image)
                pages.append(_page_text(pdf_path, document_type, page_index, text, "local_ocr"))
                continue

            direct_text = page.get_text("text").strip()
            if direct_text:
                pages.append(_page_text(pdf_path, document_type, page_index, direct_text, "pdf_text"))
                continue

            if ocr is None:
                pages.append(_page_text(pdf_path, document_type, page_index, "", "empty"))
                continue

            image = render_page(page, settings.render_dpi)
            text = ocr.image_to_text(image)
            pages.append(_page_text(pdf_path, document_type, page_index, text, "local_ocr_fallback"))
    return pages


def _page_text(
    pdf_path: Path,
    document_type: str,
    page_number: int,
    text: str,
    extraction_method: str,
) -> PageText:
    return PageText(
        source_path=str(pdf_path),
        source_name=pdf_path.name,
        document_type=document_type,
        page_number=page_number,
        text=text.strip(),
        extraction_method=extraction_method,
    )


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
