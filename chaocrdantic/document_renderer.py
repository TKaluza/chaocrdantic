"""
Document-level Markdown renderer that assembles page outputs into the gold-target format.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from chaocrdantic.output import (
    convert_fragment_to_markdown,
    figure_html,
    html_table,
)


PAGE_BREAK = '<div style="break-before: page;"></div>'


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _front_matter(result) -> str:
    lines = ["---"]
    lines.append(f"doc_id: {_yaml_quote(Path(result.file_path).stem)}")
    lines.append(f"ocr_engine: {_yaml_quote(result.ocr_engine)}")
    lines.append(f"ocr_model: {_yaml_quote(result.ocr_model)}")
    lines.append("pages:")
    for page in result.pages:
        dims = page.dimensions
        dpi = dims.dpi if dims.dpi is not None else 0
        lines.append(f"  - page: {page.page_number + 1}")
        lines.append(
            f"    dimensions: {{dpi: {dpi}, width: {dims.width}, height: {dims.height}}}"
        )
    lines.append("---")
    return "\n".join(lines)


def _block_markdown(block) -> str:
    return convert_fragment_to_markdown(block.content).strip()


def _img_text_from_content(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    img = soup.find("img")
    if not img:
        return "", ""
    return img.get("alt", ""), img.get("title", "")


def _render_page(page, asset_dir_name: str, include_images: bool) -> str:
    if page.error:
        message = (page.error_message or "Unknown OCR error.").strip()
        return f"> OCR error on page {page.page_number + 1}: {message}"

    pieces: list[str] = []
    pending_caption: str | None = None
    figure_index = 0

    image_by_name = {img.name: img for img in page.extracted_images}
    image_iter = iter(page.extracted_images)

    for block in page.layout_blocks:
        label = block.label

        if label in ["Page-Header", "Page-Footer", "Blank-Page"]:
            continue

        if label == "Caption":
            caption_text = _block_markdown(block)
            if caption_text:
                pending_caption = caption_text
            continue

        if label == "Table":
            if pending_caption:
                pieces.append(pending_caption)
                pending_caption = None
            pieces.append(html_table(block.content))
            continue

        if label in ["Image", "Figure"]:
            if include_images and page.extracted_images:
                figure_index += 1
                try:
                    extracted = next(image_iter)
                except StopIteration:
                    extracted = None

                alt, title = _img_text_from_content(block.content)
                caption = title or ""
                trailing_caption = None
                if pending_caption:
                    if re.match(r"^fig\.", pending_caption, flags=re.IGNORECASE):
                        trailing_caption = pending_caption
                    elif not caption:
                        caption = pending_caption
                    else:
                        trailing_caption = pending_caption
                    pending_caption = None

                if extracted is not None:
                    src = f"{asset_dir_name}/{extracted.name}"
                else:
                    src = ""

                pieces.append(
                    figure_html(
                        figure_id=f"fig-p{page.page_number + 1}-{figure_index - 1}",
                        src=src,
                        alt=alt,
                        title=title,
                        caption=caption or title,
                    )
                )
                if trailing_caption:
                    pieces.append(trailing_caption)
            continue

        block_md = _block_markdown(block)
        if not block_md:
            continue
        if pending_caption:
            pieces.append(pending_caption)
            pending_caption = None
        pieces.append(block_md)

    if pending_caption:
        pieces.append(pending_caption)

    text = "\n\n".join(piece.strip() for piece in pieces if piece and piece.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def render_document_markdown(result, include_images: bool = True) -> str:
    asset_dir_name = result.asset_dir_name
    parts = [_front_matter(result)]

    rendered_pages = [
        _render_page(page, asset_dir_name=asset_dir_name, include_images=include_images)
        for page in result.pages
    ]
    if rendered_pages:
        parts.append(rendered_pages[0])
        for page_content in rendered_pages[1:]:
            parts.append(PAGE_BREAK)
            parts.append(page_content)

    return "\n\n".join(part for part in parts if part and part.strip()).strip() + "\n"
