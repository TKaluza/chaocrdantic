"""
pydantic-ai based OCR agent for chandra-llamaserver.

The ChandraOCRAgent wraps a pydantic_ai.Agent configured to use an
OpenAI-compatible provider (default: http://127.0.0.1:12434/v1 with model
chandra-ocr-2).  It processes multi-page PDFs by:

1. Converting each page to a scaled, base64-encoded PNG.
2. Sending each page image together with the OCR prompt to the agent.
3. Parsing the returned HTML into structured OCRPageResult objects.
4. Combining per-page results into a single OCRResult.
"""

from __future__ import annotations

import os
from typing import List, Optional

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from chandra_ocr_pydantic.config import ChandraOCRSettings, default_settings
from chandra_ocr_pydantic.image_utils import (
    load_file_pages,
    prepare_image_for_inference,
)
from chandra_ocr_pydantic.models import LayoutBlock, OCRPageResult, OCRResult
from chandra_ocr_pydantic.prompts import OCR_LAYOUT_PROMPT, OCR_PROMPT


def _parse_layout_blocks(raw_html: str) -> List[LayoutBlock]:
    """
    Parse the raw HTML returned by the model into a list of LayoutBlock objects.

    Each top-level <div> with data-label and data-bbox attributes is treated as
    one layout block.  Mirrors chandra.output.parse_layout behaviour.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(raw_html, "html.parser")
    top_divs = soup.find_all("div", recursive=False)
    blocks: List[LayoutBlock] = []

    for div in top_divs:
        label = div.get("data-label") or "block"
        if label == "Blank-Page":
            continue

        bbox_raw = div.get("data-bbox")
        bbox: Optional[List[int]] = None
        if bbox_raw:
            try:
                parts = bbox_raw.split()
                bbox = [int(float(p)) for p in parts]
                if len(bbox) != 4:
                    bbox = None
            except (ValueError, TypeError):
                bbox = None

        # Strip nested data-bbox from content (not needed in structured output)
        content_soup = BeautifulSoup(str(div.decode_contents()), "html.parser")
        for tag in content_soup.find_all(attrs={"data-bbox": True}):
            del tag["data-bbox"]
        content = str(content_soup)

        blocks.append(LayoutBlock(label=label, bbox=bbox, content=content))

    return blocks


def _html_to_markdown(raw_html: str) -> str:
    """
    Convert the raw model HTML into Markdown.

    Falls back to the raw HTML string if markdownify is not available.
    """
    try:
        from chandra.output import parse_markdown
        return parse_markdown(raw_html)
    except ImportError:
        pass

    # Minimal fallback using markdownify directly
    try:
        from markdownify import markdownify
        return markdownify(raw_html)
    except ImportError:
        return raw_html


def _html_strip(raw_html: str) -> str:
    """
    Strip page-level boilerplate (headers/footers, blank pages) from raw HTML.

    Falls back to the raw HTML string if beautifulsoup4 is not available.
    """
    try:
        from chandra.output import parse_html
        return parse_html(raw_html, include_headers_footers=False, include_images=True)
    except ImportError:
        return raw_html


class ChandraOCRAgent:
    """
    pydantic-ai based OCR agent for Chandra.

    Configures a pydantic_ai.Agent with an OpenAI-compatible provider and
    exposes a simple ``run_file()`` / ``run_pages()`` API for processing
    PDFs and images.

    Args:
        settings: ChandraOCRSettings instance. Defaults to the module-level
            ``default_settings`` (reads env vars with CHANDRA_PYDANTIC_ prefix).
        use_layout: If True (default), use the layout-aware OCR prompt that
            returns bounding boxes and block labels. Set to False for simpler
            plain-HTML output.
    """

    def __init__(
        self,
        settings: ChandraOCRSettings = default_settings,
        use_layout: bool = True,
    ) -> None:
        self.settings = settings
        self.use_layout = use_layout

        model = OpenAIChatModel(
            settings.MODEL_NAME,
            provider=OpenAIProvider(
                base_url=settings.BASE_URL,
                api_key=settings.API_KEY,
            ),
        )

        # The agent returns plain strings (the raw HTML from the model).
        # Structured parsing is done in Python after inference.
        self._agent: Agent[None, str] = Agent(
            model=model,
            output_type=str,
            system_prompt=(
                "You are an expert OCR system. "
                "Convert document images to accurate, structured HTML."
            ),
        )

        self._prompt = OCR_LAYOUT_PROMPT if use_layout else OCR_PROMPT

    # ------------------------------------------------------------------
    # Low-level: process a pre-loaded list of PIL images
    # ------------------------------------------------------------------

    def run_pages(
        self,
        pages: list,
        file_path: str = "<unknown>",
    ) -> OCRResult:
        """
        Run OCR on a list of pre-loaded PIL Images.

        Args:
            pages: List of PIL.Image.Image objects (one per page).
            file_path: Source file path (used for metadata only).

        Returns:
            OCRResult with per-page results.
        """
        page_results: List[OCRPageResult] = []

        for page_idx, image in enumerate(pages):
            try:
                image_b64 = prepare_image_for_inference(image)

                # Build the multimodal user message as a list of content parts.
                # pydantic_ai accepts a list of dicts matching the OpenAI
                # content-block format when passed as the prompt.
                user_content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": self._prompt,
                    },
                ]

                result = self._agent.run_sync(user_content)
                raw_html = result.output

                page_results.append(
                    OCRPageResult(
                        page_number=page_idx,
                        raw_html=raw_html,
                        markdown=_html_to_markdown(raw_html),
                        html=_html_strip(raw_html),
                        layout_blocks=_parse_layout_blocks(raw_html) if self.use_layout else [],
                        error=False,
                    )
                )

            except Exception as exc:
                page_results.append(
                    OCRPageResult(
                        page_number=page_idx,
                        raw_html="",
                        markdown="",
                        html="",
                        layout_blocks=[],
                        error=True,
                        error_message=str(exc),
                    )
                )

        return OCRResult(
            file_path=os.path.abspath(file_path),
            num_pages=len(pages),
            pages=page_results,
        )

    # ------------------------------------------------------------------
    # High-level: process a file path
    # ------------------------------------------------------------------

    def run_file(
        self,
        file_path: str,
        page_range: Optional[List[int]] = None,
    ) -> OCRResult:
        """
        Run OCR on a PDF or image file.

        Args:
            file_path: Path to the PDF or image to process.
            page_range: Optional list of 0-based page indices to process.
                Pass None to process all pages.

        Returns:
            OCRResult with per-page results.
        """
        pages = load_file_pages(
            file_path,
            page_range=page_range,
            settings=self.settings,
        )
        return self.run_pages(pages, file_path=file_path)
