"""
Pydantic models for structured JSON responses.
"""

from typing import List, Optional, Union, Literal
from pydantic import BaseModel


class BulletSection(BaseModel):
    """Bullet point list section."""
    heading: Optional[str] = None
    type: Literal["bullets"]
    items: List[str]


class ParagraphSection(BaseModel):
    """Paragraph text section."""
    heading: Optional[str] = None
    type: Literal["paragraph"]
    content: str


class TableSection(BaseModel):
    """Table with headers and rows."""
    heading: Optional[str] = None
    type: Literal["table"]
    headers: List[str]
    rows: List[List[str]]


class StepsSection(BaseModel):
    """Numbered steps section."""
    heading: Optional[str] = None
    type: Literal["steps"]
    items: List[str]


class AlertSection(BaseModel):
    """Alert/warning box section."""
    heading: Optional[str] = None
    type: Literal["alert"]
    content: str
    severity: Literal["info", "warning", "important"]


# Union of all section types
Section = Union[
    BulletSection,
    ParagraphSection,
    TableSection,
    StepsSection,
    AlertSection
]


class StructuredResponse(BaseModel):
    """Complete structured response from LLM."""
    intent: str
    title: str
    subtitle: Optional[str] = None
    sections: List[Section]
    footer: Optional[str] = None
    confidence: Literal["high", "medium", "low", "none"]
    fallback: Optional[str] = None
