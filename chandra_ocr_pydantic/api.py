"""
Convenience functions for one-call usage of the pydantic-ai OCR library.

These are thin wrappers around ChandraOCRAgent that use the module-level
default settings.  They are suitable for simple scripts and notebooks.

For more control (custom settings, reusing the agent across multiple files,
accessing the underlying pydantic_ai.Agent), instantiate ChandraOCRAgent
directly.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image as _PilImage

from chandra_ocr_pydantic.agent import ChandraOCRAgent
from chandra_ocr_pydantic.config import default_settings
from chandra_ocr_pydantic.models import OCRResult

# Module-level agent singleton — created lazily on first use
_default_agent: Optional[ChandraOCRAgent] = None


def _get_agent() -> ChandraOCRAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = ChandraOCRAgent(settings=default_settings, use_layout=True)
    return _default_agent


def ocr_file(
    file_path: str,
    page_range: Optional[List[int]] = None,
) -> OCRResult:
    """
    OCR a PDF or image file using the default agent configuration.

    The first call creates a module-level agent pointing at
    http://127.0.0.1:12434/v1 with model chandra-ocr-2.  Override these
    values by setting CHANDRA_PYDANTIC_BASE_URL and CHANDRA_PYDANTIC_MODEL_NAME
    environment variables before the first call.

    Args:
        file_path: Path to the PDF or image file.
        page_range: Optional list of 0-based page indices to process.
            Pass None (default) to process all pages.

    Returns:
        OCRResult containing per-page results plus convenience properties
        ``markdown`` and ``html`` for the full document.

    Example::

        from chandra_ocr_pydantic import ocr_file

        result = ocr_file("report.pdf")
        print(result.markdown)
        for page in result.pages:
            print(f"Page {page.page_number}: {len(page.layout_blocks)} blocks")
    """
    agent = _get_agent()
    return agent.run_file(file_path, page_range=page_range)


def ocr_image(image: "_PilImage.Image") -> OCRResult:
    """
    OCR a single PIL Image using the default agent configuration.

    Args:
        image: A PIL.Image.Image object (any mode; will be converted to RGB).

    Returns:
        OCRResult with a single page entry at index 0.

    Example::

        from PIL import Image
        from chandra_ocr_pydantic import ocr_image

        img = Image.open("scan.png")
        result = ocr_image(img)
        print(result.pages[0].markdown)
    """
    agent = _get_agent()
    rgb = image.convert("RGB")
    return agent.run_pages([rgb], file_path="<PIL.Image>")
