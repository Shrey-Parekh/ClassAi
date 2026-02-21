"""
Tests for semantic chunking.
"""

import pytest
from src.chunking.semantic_chunker import SemanticChunker, Chunk
from config.chunking_config import ChunkLevel, ContentType


def test_procedure_not_split():
    """Test that procedures with steps stay together."""
    chunker = SemanticChunker()
    
    content = """
    Leave Application Procedure:
    Step 1: Fill the leave application form
    Step 2: Get approval from department head
    Step 3: Submit to HR department
    Step 4: Wait for confirmation email
    Step 5: Mark leave in attendance system
    """
    
    doc_metadata = {
        "doc_id": "test_doc",
        "title": "Test Document",
    }
    
    chunks = chunker.chunk_document(content, doc_metadata)
    
    # Find procedure chunks
    procedure_chunks = [c for c in chunks if c.level == ChunkLevel.PROCEDURE]
    
    # All steps should be in one chunk
    assert any("Step 1" in c.content and "Step 5" in c.content 
               for c in procedure_chunks)


def test_conditional_rule_not_split():
    """Test that if-then rules stay together."""
    chunker = SemanticChunker()
    
    content = """
    If you are a teaching staff member with more than 5 years of service,
    then you are eligible for sabbatical leave of up to 1 year.
    """
    
    doc_metadata = {"doc_id": "test_doc"}
    
    chunks = chunker.chunk_document(content, doc_metadata)
    
    # Find the rule chunk
    rule_chunks = [c for c in chunks if "If you are" in c.content]
    
    # Condition and consequence should be together
    assert any("If you are" in c.content and "then you are eligible" in c.content
               for c in rule_chunks)


def test_three_levels_created():
    """Test that all three chunk levels are created."""
    chunker = SemanticChunker()
    
    content = """
    Faculty Leave Policy
    
    Casual leave entitlement is 12 days per year.
    
    Leave Application Process:
    Step 1: Submit application
    Step 2: Get approval
    """
    
    doc_metadata = {"doc_id": "test_doc"}
    
    chunks = chunker.chunk_document(content, doc_metadata)
    
    levels = {c.level for c in chunks}
    
    # Should have all three levels
    assert ChunkLevel.OVERVIEW in levels
    assert ChunkLevel.PROCEDURE in levels
    assert ChunkLevel.ATOMIC in levels
