"""
Image utilities for the pydantic-ai OCR library.

Re-uses the proven rendering logic from chandra.input and chandra.model.util
rather than duplicating it.
"""

from __future__ import annotations

import base64
import io
from typing import List, Optional, Tuple

import filetype
from PIL import Image

from chandra_ocr_pydantic.config import ChandraOCRSettings, default_settings


def _scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
    min_size: Tuple[int, int] = (1792, 28),
    grid_size: int = 28,
) -> Image.Image:
    """
    Scale an image to fit within max_size while staying above min_size,
    snapping dimensions to multiples of grid_size for model compatibility.

    Mirrors chandra.model.util.scale_to_fit.
    """
    resample = Image.Resampling.LANCZOS
    width, height = img.size

    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    new_width = w_blocks * grid_size
    new_height = h_blocks * grid_size

    if (new_width, new_height) == (width, height):
        return img
    return img.resize((new_width, new_height), resample=resample)


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string (no data-URL prefix)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_pdf_pages(
    filepath: str,
    page_range: Optional[List[int]] = None,
    settings: ChandraOCRSettings = default_settings,
) -> List[Image.Image]:
    """
    Render PDF pages to PIL Images.

    Args:
        filepath: Path to the PDF file.
        page_range: 0-based page indices to include. None means all pages.
        settings: OCR settings controlling DPI and minimum dimensions.

    Returns:
        List of PIL Images, one per selected page.
    """
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError as exc:
        raise ImportError(
            "pypdfium2 is required for PDF rendering. "
            "Install it with: pip install pypdfium2"
        ) from exc

    doc = pdfium.PdfDocument(filepath)
    doc.init_forms()

    images: List[Image.Image] = []
    for page_idx in range(len(doc)):
        if page_range is not None and page_idx not in page_range:
            continue

        page_obj = doc[page_idx]
        min_dim = min(page_obj.get_width(), page_obj.get_height())
        scale_dpi = (settings.MIN_PDF_IMAGE_DIM / min_dim) * 72
        scale_dpi = max(scale_dpi, settings.IMAGE_DPI)

        # Re-fetch after potential flatten to get a fresh handle
        page_obj = doc[page_idx]
        rc = pdfium_c.FPDFPage_Flatten(page_obj, pdfium_c.FLAT_NORMALDISPLAY)
        if rc == pdfium_c.FLATTEN_FAIL:
            print(f"Warning: failed to flatten annotations on page {page_idx}")

        page_obj = doc[page_idx]
        pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
        images.append(pil_image)

    doc.close()
    return images


def load_image_file(
    filepath: str,
    settings: ChandraOCRSettings = default_settings,
) -> Image.Image:
    """
    Load a single image file, upscaling if below the minimum dimension.

    Args:
        filepath: Path to the image file.
        settings: OCR settings controlling minimum image dimension.

    Returns:
        PIL Image in RGB mode.
    """
    image = Image.open(filepath).convert("RGB")
    min_dim = min(image.width, image.height)
    if min_dim < settings.MIN_IMAGE_DIM:
        scale = settings.MIN_IMAGE_DIM / min_dim
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_file_pages(
    filepath: str,
    page_range: Optional[List[int]] = None,
    settings: ChandraOCRSettings = default_settings,
) -> List[Image.Image]:
    """
    Load a PDF or image file and return a list of PIL Images (one per page).

    For images, always returns a single-element list.

    Args:
        filepath: Path to the PDF or image file.
        page_range: For PDFs, 0-based page indices to include. None = all pages.
        settings: OCR settings.

    Returns:
        List[PIL.Image.Image] — one entry per page/image.
    """
    detected = filetype.guess(filepath)
    if detected and detected.extension == "pdf":
        return load_pdf_pages(filepath, page_range=page_range, settings=settings)
    return [load_image_file(filepath, settings=settings)]


def prepare_image_for_inference(image: Image.Image) -> str:
    """
    Scale an image to model-compatible dimensions, then encode as base64 PNG.

    Returns:
        Base64-encoded PNG string (no data-URL prefix).
    """
    scaled = _scale_to_fit(image)
    return image_to_base64(scaled)
