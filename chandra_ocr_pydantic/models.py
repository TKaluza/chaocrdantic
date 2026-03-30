"""
Pydantic models for structured OCR output.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LayoutBlock(BaseModel):
    """A single detected layout block within a page."""

    label: str = Field(description="Block type label (e.g. Text, Table, Section-Header)")
    bbox: Optional[List[int]] = Field(
        default=None,
        description="Bounding box [x0, y0, x1, y1] in image pixels, or None if not available",
    )
    content: str = Field(description="HTML content of the block")


class OCRPageResult(BaseModel):
    """OCR result for a single page."""

    page_number: int = Field(description="0-based page index")
    raw_html: str = Field(description="Raw HTML string returned by the model")
    markdown: str = Field(description="Markdown converted from the raw HTML")
    html: str = Field(description="Cleaned HTML (headers/footers stripped)")
    layout_blocks: List[LayoutBlock] = Field(
        default_factory=list,
        description="Parsed layout blocks with bounding boxes and labels",
    )
    error: bool = Field(default=False, description="True if inference failed for this page")
    error_message: Optional[str] = Field(default=None, description="Error message if error=True")


class OCRResult(BaseModel):
    """Aggregated OCR result for an entire document."""

    file_path: str = Field(description="Absolute path of the input file")
    num_pages: int = Field(description="Total number of pages processed")
    pages: List[OCRPageResult] = Field(description="Per-page results in order")

    @property
    def markdown(self) -> str:
        """Concatenated Markdown for all pages."""
        return "\n\n".join(p.markdown for p in self.pages if p.markdown)

    @property
    def html(self) -> str:
        """Concatenated HTML for all pages."""
        return "\n".join(p.html for p in self.pages if p.html)

    @property
    def has_errors(self) -> bool:
        """True if any page had an inference error."""
        return any(p.error for p in self.pages)
