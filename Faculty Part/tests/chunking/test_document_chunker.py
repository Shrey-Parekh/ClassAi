"""
Regression tests for DocumentChunker regex and chunking logic.
"""
import sys
from pathlib import Path

import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chunking.document_chunker import DocumentChunker


@pytest.fixture
def chunker():
    return DocumentChunker()


# ── detect_source_type ────────────────────────────────────────────────────────

def test_detect_faculty_csv(chunker):
    assert chunker.detect_source_type(Path("data/raw/faculty_list.csv")) == "faculty_profile"

def test_detect_leave_policy(chunker):
    result = chunker.detect_source_type(Path("data/raw/leave_policy.pdf"))
    assert result == "hr_policy"

def test_detect_compendium_forms(chunker):
    result = chunker.detect_source_type(Path("data/raw/compendium_forms.pdf"))
    assert result == "form_document"

def test_detect_agreement(chunker):
    result = chunker.detect_source_type(Path("data/raw/agreement.pdf"))
    assert result == "legal_document"


# ── _chunk_form ───────────────────────────────────────────────────────────────

NMIMS_HEADER = "NMIMS — Narsee Monjee Institute of Management Studies"

FORM_1 = f"""{NMIMS_HEADER}
LEAVE APPLICATION FORM
Form Code: HR-LA-01

SECTION A: Applicant Details
Name: ___________

SECTION B: Leave Details
From: ___________
"""

FORM_2 = f"""{NMIMS_HEADER}
CONFERENCE ATTENDANCE FORM
Form Code: HR-CO-01

SECTION A: Event Details
Conference: ___________
"""

TWO_FORM_TEXT = FORM_1 + "\n" + FORM_2


def test_chunk_form_two_forms(chunker, tmp_path):
    f = tmp_path / "NMIMS_Faculty_Applications_Compendium.pdf"
    f.touch()
    chunks = chunker._chunk_form(TWO_FORM_TEXT, f, {"doc_id": "test"})
    form_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "form_template"]
    assert len(form_chunks) == 2, f"Expected 2 form chunks, got {len(form_chunks)}"


def test_chunk_form_repeated_header_stays_one(chunker, tmp_path):
    """Repeated NMIMS page header inside a single form must NOT split it."""
    single_form = f"""{NMIMS_HEADER}
LEAVE APPLICATION FORM
Form Code: HR-LA-01

SECTION A: Applicant Details
Name: ___________

{NMIMS_HEADER}
(continued)

SECTION B: Leave Details
From: ___________
"""
    f = tmp_path / "NMIMS_Faculty_Applications_Compendium.pdf"
    f.touch()
    chunks = chunker._chunk_form(single_form, f, {"doc_id": "test"})
    form_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "form_template"]
    assert len(form_chunks) == 1, f"Expected 1 form chunk, got {len(form_chunks)}"


# ── clean_chunk_text ──────────────────────────────────────────────────────────

def test_clean_removes_page_marker(chunker):
    text = "Some content\nPage | 12\nMore content"
    result = chunker.clean_chunk_text(text)
    assert "Page | 12" not in result

def test_clean_deduplicates_parameter_details(chunker):
    text = "Parameter Details\nParameter Details\nSome value"
    result = chunker.clean_chunk_text(text)
    assert result.count("Parameter Details") <= 1
