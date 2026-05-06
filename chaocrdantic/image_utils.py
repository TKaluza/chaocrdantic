"""Image utilities for the chaocrdantic OCR library."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import filetype
from PIL import Image

from chaocrdantic.config import ChaocrdanticSettings, default_settings
from chaocrdantic.util import scale_to_fit


@dataclass
class RenderedPage:
    page_number: int
    image: Image.Image
    dpi: Optional[int] = None


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string (no data-URL prefix)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _as_file_input(filepath: str | Path | bytes) -> str | Path | io.BytesIO:
    if isinstance(filepath, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(filepath))
    return filepath


def _guess_filetype(filepath: str | Path | bytes):
    if isinstance(filepath, (bytes, bytearray, memoryview)):
        return filetype.guess(bytes(filepath))
    return filetype.guess(str(filepath))


def load_pdf_pages(
    filepath: str | Path | bytes,
    page_range: Optional[List[int]] = None,
    settings: ChaocrdanticSettings = default_settings,
) -> List[RenderedPage]:
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError as exc:
        raise ImportError(
            "pypdfium2 is required for PDF rendering. "
            "Install it with: pip install pypdfium2"
        ) from exc

    doc = pdfium.PdfDocument(_as_file_input(filepath))
    try:
        doc.init_forms()

        pages: List[RenderedPage] = []
        for page_idx in range(len(doc)):
            if page_range is not None and page_idx not in page_range:
                continue

            page_obj = doc[page_idx]
            min_dim = min(page_obj.get_width(), page_obj.get_height())
            scale_dpi = (settings.MIN_PDF_IMAGE_DIM / min_dim) * 72
            scale_dpi = max(scale_dpi, settings.IMAGE_DPI)

            page_obj = doc[page_idx]
            rc = pdfium_c.FPDFPage_Flatten(page_obj, pdfium_c.FLAT_NORMALDISPLAY)
            if rc == pdfium_c.FLATTEN_FAIL:
                print(f"Warning: failed to flatten annotations on page {page_idx}")

            page_obj = doc[page_idx]
            pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
            pages.append(
                RenderedPage(
                    page_number=page_idx,
                    image=pil_image,
                    dpi=int(scale_dpi),
                )
            )

        return pages
    finally:
        doc.close()


def load_image_file(
    filepath: str | Path | bytes,
    settings: ChaocrdanticSettings = default_settings,
) -> RenderedPage:
    source = _as_file_input(filepath)
    with Image.open(source) as pil_image:
        dpi_value = pil_image.info.get("dpi")
        image = pil_image.convert("RGB")

    min_dim = min(image.width, image.height)
    if min_dim < settings.MIN_IMAGE_DIM:
        scale = settings.MIN_IMAGE_DIM / min_dim
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    dpi = None
    try:
        if dpi_value:
            dpi = int(dpi_value[0])
    except Exception:
        dpi = None

    return RenderedPage(page_number=0, image=image, dpi=dpi)


def load_file_pages(
    filepath: str | Path | bytes,
    page_range: Optional[List[int]] = None,
    settings: ChaocrdanticSettings = default_settings,
) -> List[RenderedPage]:
    detected = _guess_filetype(filepath)
    if detected and detected.extension == "pdf":
        return load_pdf_pages(filepath, page_range=page_range, settings=settings)
    return [load_image_file(filepath, settings=settings)]


def prepare_image_for_inference(
    image: Image.Image,
    max_size: tuple[int, int] = (3072, 2048),
    min_size: tuple[int, int] = (1792, 28),
) -> str:
    scaled = scale_to_fit(image, max_size=max_size, min_size=min_size)
    return image_to_base64(scaled)
