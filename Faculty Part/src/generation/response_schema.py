"""
Pydantic schema for structured JSON responses.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ParagraphSection(BaseModel):
    """Paragraph section with heading and content."""
    heading: Optional[str] = None
    type: Literal["paragraph"] = "paragraph"
    content: str


class BulletsSection(BaseModel):
    """Bullet list section."""
    heading: Optional[str] = None
    type: Literal["bullets"] = "bullets"
    items: List[str]


class StepsSection(BaseModel):
    """Numbered steps section."""
    heading: Optional[str] = None
    type: Literal["steps"] = "steps"
    items: List[str]


class AlertSection(BaseModel):
    """Alert/callout section."""
    heading: Optional[str] = None
    type: Literal["alert"] = "alert"
    content: str
    severity: Literal["info", "warning", "important"] = "info"


# Union type for all section types
Section = ParagraphSection | BulletsSection | StepsSection | AlertSection


class StructuredResponse(BaseModel):
    """
    Complete structured response schema.
    
    This is the JSON format that the LLM must return.
    """
    intent: str = Field(..., description="Query intent type")
    title: str = Field(..., description="Response title")
    subtitle: Optional[str] = Field(None, description="Optional subtitle")
    sections: List[Section] = Field(default_factory=list, description="Content sections")
    footer: Optional[str] = Field(None, description="Optional footer note")
    confidence: Literal["high", "medium", "low", "none"] = Field("high", description="Confidence level")
    fallback: Optional[str] = Field(None, description="Fallback message if answer not found")
