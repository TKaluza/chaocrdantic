"""Convenience functions for one-call usage of the chaocrdantic OCR library."""

from __future__ import annotations

from typing import List, Optional

from PIL import Image as _PilImage

from chaocrdantic.agent import ChaocrdanticAgent
from chaocrdantic.config import default_settings
from chaocrdantic.models import OCRResult

# Module-level agent singleton — created lazily on first use
_default_agent: Optional[ChaocrdanticAgent] = None


def _get_agent() -> ChaocrdanticAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = ChaocrdanticAgent(settings=default_settings, use_layout=True)
    return _default_agent


def ocr_file(
    file_path: str,
    page_range: Optional[List[int]] = None,
) -> OCRResult:
    """
    OCR a PDF or image file using the default agent configuration.

    The first call creates a module-level agent pointing at
    http://127.0.0.1:12434/v1 with model chandra-ocr-2-vllm. That model name
    comes from the upstream Chandra project and is retained here for
    compatibility. Override these values by setting CHAOCRDANTIC_BASE_URL and
    CHAOCRDANTIC_MODEL_NAME environment variables before the first call.

    Args:
        file_path: Path to the PDF or image file.
        page_range: Optional list of 0-based page indices to process.
            Pass None (default) to process all pages.

    Returns:
        OCRResult containing per-page results plus convenience properties
        ``markdown`` and ``html`` for the full document.

    Example::

        from chaocrdantic import ocr_file

        result = ocr_file("report.pdf")
        print(result.markdown)
        for page in result.pages:
            print(f"Page {page.page_number}: {len(page.layout_blocks)} blocks")
    """
    agent = _get_agent()
    return agent.run_file(file_path, page_range=page_range)


async def ocr_file_async(
    file_path: str,
    page_range: Optional[List[int]] = None,
) -> OCRResult:
    """Async OCR for a PDF or image file using the default agent configuration."""
    agent = _get_agent()
    return await agent.run_file_async(file_path, page_range=page_range)


def ocr_image(image: "_PilImage.Image") -> OCRResult:
    """
    OCR a single PIL Image using the default agent configuration.

    Args:
        image: A PIL.Image.Image object (any mode; will be converted to RGB).

    Returns:
        OCRResult with a single page entry at index 0.

    Example::

        from PIL import Image
        from chaocrdantic import ocr_image

        img = Image.open("scan.png")
        result = ocr_image(img)
        print(result.pages[0].markdown)
    """
    agent = _get_agent()
    rgb = image.convert("RGB")
    return agent.run_pages([rgb], file_path="<PIL.Image>")


async def ocr_image_async(image: "_PilImage.Image") -> OCRResult:
    """Async OCR for a single PIL image using the default agent configuration."""
    agent = _get_agent()
    rgb = image.convert("RGB")
    return await agent.run_pages_async([rgb], file_path="<PIL.Image>")
