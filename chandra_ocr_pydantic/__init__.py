"""
chandra_ocr_pydantic — pydantic-ai based OCR library for chandra-llamaserver.

Provides an Agent-based interface for running OCR on PDFs and images using an
OpenAI-compatible provider (e.g. a locally-running chandra-ocr-2 model server).

Quick start::

    from chandra_ocr_pydantic import ocr_file, ChandraOCRAgent

    # Simple one-call API
    result = ocr_file("document.pdf")
    for page in result.pages:
        print(page.markdown)

    # Or use the agent directly
    agent = ChandraOCRAgent()
    result = agent.run_file("document.pdf")
"""

from chandra_ocr_pydantic.agent import ChandraOCRAgent
from chandra_ocr_pydantic.models import OCRPageResult, OCRResult
from chandra_ocr_pydantic.api import ocr_file, ocr_image

__all__ = [
    "ChandraOCRAgent",
    "OCRPageResult",
    "OCRResult",
    "ocr_file",
    "ocr_image",
]
