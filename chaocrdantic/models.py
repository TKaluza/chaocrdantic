"""
Pydantic models for structured OCR output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from PIL import Image


class PageDimensions(BaseModel):
    """Rendered page dimensions used for OCR and output metadata."""

    dpi: Optional[int] = Field(default=None, description="Raster DPI used to render the page")
    width: int = Field(description="Rendered page width in pixels")
    height: int = Field(description="Rendered page height in pixels")


class LayoutBlock(BaseModel):
    """A single detected layout block within a page."""

    label: str = Field(description="Block type label (e.g. Text, Table, Section-Header)")
    bbox: Optional[List[int]] = Field(
        default=None,
        description="Bounding box [x0, y0, x1, y1] in image pixels, or None if not available",
    )
    content: str = Field(description="HTML content of the block")


class ExtractedImage(BaseModel):
    """Metadata about an extracted figure/image crop."""

    name: str = Field(description="File name used when saving the extracted image")
    label: str = Field(description="Original OCR block label")
    bbox: List[int] = Field(description="Pixel-space crop bbox")
    alt: str = Field(default="", description="Image alt text from OCR HTML")
    title: str = Field(default="", description="Image title from OCR HTML")


class OCRPageResult(BaseModel):
    """OCR result for a single page."""

    page_number: int = Field(description="0-based page index in the source document")
    raw_html: str = Field(description="Raw HTML string returned by the model")
    markdown: str = Field(description="Page-level Markdown converted from the raw HTML")
    html: str = Field(description="Cleaned HTML (headers/footers stripped)")
    dimensions: PageDimensions = Field(description="Rendered page dimensions")
    token_count: int = Field(default=0, description="Completion tokens used for this page")
    layout_blocks: List[LayoutBlock] = Field(
        default_factory=list,
        description="Parsed layout blocks with bounding boxes and labels",
    )
    extracted_images: List[ExtractedImage] = Field(
        default_factory=list,
        description="Metadata for image crops extracted from figure/image blocks",
    )
    error: bool = Field(default=False, description="True if inference failed for this page")
    error_message: Optional[str] = Field(default=None, description="Error message if error=True")

    _pil_images: dict[str, "Image.Image"] = PrivateAttr(default_factory=dict)

    def set_extracted_images(self, images: dict[str, "Image.Image"]) -> None:
        self._pil_images = images

    def iter_extracted_images(self):
        return self._pil_images.items()


class OCRResult(BaseModel):
    """Aggregated OCR result for an entire document."""

    file_path: str = Field(description="Absolute path of the input file")
    ocr_engine: str = Field(description="OCR pipeline identifier")
    ocr_model: str = Field(description="Model name used for OCR")
    num_pages: int = Field(description="Total number of pages processed")
    pages: List[OCRPageResult] = Field(description="Per-page results in order")

    @property
    def asset_dir_name(self) -> str:
        return f"{Path(self.file_path).stem}_assets"

    @property
    def markdown(self) -> str:
        """Merged document Markdown in the high-fidelity renderer format."""
        return self.render_markdown()

    @property
    def html(self) -> str:
        """Concatenated cleaned HTML for all pages with page break markers."""
        return self.render_html()

    @property
    def has_errors(self) -> bool:
        """True if any page had an inference error."""
        return any(p.error for p in self.pages)

    def render_markdown(self, include_images: bool = True) -> str:
        from chaocrdantic.document_renderer import render_document_markdown

        return render_document_markdown(self, include_images=include_images)

    def render_html(self, include_page_breaks: bool = True) -> str:
        page_break = "\n\n<div style=\"break-before: page;\"></div>\n\n"
        html_pages = [page.html for page in self.pages if page.html]
        if include_page_breaks:
            return page_break.join(html_pages)
        return "\n".join(html_pages)

    def save_extracted_images(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        assets_dir = output_path / self.asset_dir_name
        assets_dir.mkdir(parents=True, exist_ok=True)
        for page in self.pages:
            for name, pil_image in page.iter_extracted_images():
                pil_image.save(assets_dir / name)
        return assets_dir
