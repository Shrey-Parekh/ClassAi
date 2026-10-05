"""
Tests for intent routing fixes and FORM_DETAILS_PROMPT cleanup.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.query_understanding import QueryAnalyzer
from src.generation.prompt_templates import FORM_DETAILS_PROMPT


@pytest.fixture(scope="module")
def analyzer():
    return QueryAnalyzer()


def test_list_forms_routes_to_document_overview(analyzer):
    result = analyzer.analyze("Give me the form code for all the form applications")
    assert result.intent == "document_overview", f"Got: {result.intent}"


def test_specific_form_code_routes_to_form_details(analyzer):
    result = analyzer.analyze("What is HR-LA-01?")
    assert result.intent == "form_details", f"Got: {result.intent}"


def test_list_all_forms_routes_to_document_overview(analyzer):
    result = analyzer.analyze("List all forms in the compendium")
    assert result.intent == "document_overview", f"Got: {result.intent}"


def test_how_to_fill_form_routes_to_procedure(analyzer):
    result = analyzer.analyze("How do I fill HR-LA-01?")
    assert result.intent == "procedure", f"Got: {result.intent}"


def test_this_form_routes_to_form_details(analyzer):
    result = analyzer.analyze("Tell me about this form")
    assert result.intent == "form_details", f"Got: {result.intent}"


def test_form_details_prompt_no_required_fields_priming():
    assert 'Include a "Purpose"' not in FORM_DETAILS_PROMPT
    assert 'Required Fields (bullets), Approval Chain (steps)' not in FORM_DETAILS_PROMPT


def test_form_codes_list_routes_to_overview():
    from src.retrieval.query_understanding import QueryAnalyzer
    qa = QueryAnalyzer()
    assert qa.analyze("Give me the form code for all the form applications").intent == "document_overview"
    assert qa.analyze("list all form codes").intent == "document_overview"


def test_overview_prompt_has_enumeration_rules():
    from src.generation.prompt_templates import INTENT_PROMPTS
    p = INTENT_PROMPTS["document_overview"]
    assert "Enumerate every distinct item" in p
    assert "bullets" in p


def test_grounding_empty_answer_returns_none_ratio():
    from src.generation.answer_generator import AnswerGenerator
    from types import SimpleNamespace
    import logging
    gen = AnswerGenerator.__new__(AnswerGenerator)
    gen.logger = logging.getLogger("t")
    empty = SimpleNamespace(sections=[], summary=None, footer=None)
    r = gen._check_grounding(empty, [{"text": "some chunk"}])
    assert r["ratio"] is None
    assert r["reason"] == "no_answer_sentences"
