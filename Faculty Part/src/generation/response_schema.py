"""
Pydantic schema for structured JSON responses.
"""

from typing import List, Optional, Literal, Union, Annotated, Any
from pydantic import BaseModel, Field


class ParagraphSection(BaseModel):
    """Paragraph section with heading and content."""
    heading: Optional[str] = None
    type: Literal["paragraph"] = "paragraph"
    content: str = ""


class BulletsSection(BaseModel):
    """Bullet list section."""
    heading: Optional[str] = None
    type: Literal["bullets"] = "bullets"
    items: List[str] = []


class StepsSection(BaseModel):
    """Numbered steps section."""
    heading: Optional[str] = None
    type: Literal["steps"] = "steps"
    items: List[str] = []


class AlertSection(BaseModel):
    """Alert/callout section."""
    heading: Optional[str] = None
    type: Literal["alert"] = "alert"
    content: str = ""
    # G23: added "danger" severity
    severity: Literal["info", "warning", "important", "danger"] = "info"


class TableSection(BaseModel):
    """Table section with headers and rows."""
    heading: Optional[str] = None
    type: Literal["table"] = "table"
    # G22: added caption
    caption: Optional[str] = None
    headers: List[str] = []
    rows: List[List[Optional[str]]] = []

    def model_post_init(self, __context: Any) -> None:
        # Coerce None cells to empty string
        self.rows = [
            [cell if cell is not None else "" for cell in row]
            for row in self.rows
        ]


# G20: Discriminated union — faster validation, precise error messages
Section = Annotated[
    Union[ParagraphSection, BulletsSection, StepsSection, TableSection, AlertSection],
    Field(discriminator="type"),
]


class StructuredResponse(BaseModel):
    """Complete structured response schema returned by the LLM."""
    intent: str = Field(..., description="Query intent type")
    title: str = Field(..., description="Response title")
    subtitle: Optional[str] = Field(None, description="Optional subtitle")
    sections: List[Section] = Field(default_factory=list, description="Content sections")
    footer: Optional[str] = Field(None, description="Optional footer note")
    confidence: Literal["high", "medium", "low", "none"] = Field(
        "high", description="Confidence level"
    )
    fallback: Optional[str] = Field(None, description="Fallback message if answer not found")
    # G24: policy revision notes and effective date
    caveats: List[str] = Field(default_factory=list, description="Policy revision notes")
    effective_date: Optional[str] = Field(None, description="Effective date from chunk metadata")
    # G25: suggested follow-up queries
    suggested_followups: List[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )
