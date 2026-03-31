"""
HTML parsing and Markdown conversion utilities for chaocrdantic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from html import escape as html_escape
from pathlib import Path
from typing import Dict, Iterable, Optional

import six
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter, re_whitespace
from PIL import Image


PAGE_BREAK_HTML_RE = re.compile(
    r'<div\b[^>]*style=["\'][^"\']*break-before\s*:\s*page;?[^"\']*["\'][^>]*>\s*</div>',
    flags=re.IGNORECASE,
)


@lru_cache
def _hash_html(html: str) -> str:
    return hashlib.md5(html.encode("utf-8")).hexdigest()


def strip_page_break_markup(html: str) -> str:
    return PAGE_BREAK_HTML_RE.sub("", html)


def get_image_name(
    html: str,
    div_idx: int,
    page_number: Optional[int] = None,
) -> str:
    if page_number is None:
        html_hash = _hash_html(html)
        return f"{html_hash}_{div_idx}_img.webp"
    return f"page-{page_number + 1:02d}-img-{div_idx}.webp"


def _image_tag_metadata(chunk_content: str) -> tuple[str, str]:
    soup = BeautifulSoup(chunk_content, "html.parser")
    img = soup.find("img")
    if not img:
        return "", ""
    return img.get("alt", ""), img.get("title", "")


def extract_images(
    html: str,
    chunks: list[dict],
    image: Image.Image,
    page_number: Optional[int] = None,
) -> tuple[dict[str, Image.Image], list[dict]]:
    images: dict[str, Image.Image] = {}
    metadata: list[dict] = []
    div_idx = 0

    for chunk in chunks:
        div_idx += 1
        if chunk["label"] not in ["Image", "Figure"]:
            continue

        bbox = chunk["bbox"]
        if not bbox or len(bbox) != 4:
            continue

        try:
            block_image = image.crop(bbox)
        except ValueError:
            continue

        img_name = get_image_name(html, div_idx, page_number=page_number)
        alt, title = _image_tag_metadata(chunk["content"])
        images[img_name] = block_image
        metadata.append(
            {
                "name": img_name,
                "bbox": list(bbox),
                "label": chunk["label"],
                "alt": alt,
                "title": title,
            }
        )

    return images, metadata


def parse_html(
    html: str,
    include_headers_footers: bool = False,
    include_images: bool = True,
    page_number: Optional[int] = None,
) -> str:
    html = strip_page_break_markup(html)
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    out_html = ""
    div_idx = 0

    for div in top_level_divs:
        div_idx += 1
        label = div.get("data-label")

        if label == "Blank-Page":
            continue
        if label and not include_headers_footers and label in ["Page-Header", "Page-Footer"]:
            continue
        if label and not include_images and label in ["Image", "Figure"]:
            continue

        if label in ["Image", "Figure"]:
            img = div.find("img")
            img_src = get_image_name(html, div_idx, page_number=page_number)
            if img:
                img["src"] = img_src
            else:
                img = BeautifulSoup(f"<img src='{img_src}'/>", "html.parser")
                div.append(img)

        if label not in ["Image", "Figure"]:
            for img_tag in div.find_all("img"):
                if not img_tag.get("src"):
                    img_tag.decompose()

        if label in ["Text"] and not re.search("<.+>", str(div.decode_contents()).strip()):
            text_content = str(div.decode_contents()).strip()
            text_content = f"<p>{text_content}</p>"
            div.clear()
            div.append(BeautifulSoup(text_content, "html.parser"))

        content = str(div.decode_contents())
        out_html += content

    return out_html


class Markdownify(MarkdownConverter):
    def __init__(
        self,
        inline_math_delimiters: tuple[str, str],
        block_math_delimiters: tuple[str, str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.inline_math_delimiters = inline_math_delimiters
        self.block_math_delimiters = block_math_delimiters

    def convert_math(self, el, text, parent_tags):
        block = el.has_attr("display") and el["display"] == "block"
        if block:
            return (
                "\n"
                + self.block_math_delimiters[0]
                + text.strip()
                + self.block_math_delimiters[1]
                + "\n"
            )
        return (
            " "
            + self.inline_math_delimiters[0]
            + text.strip()
            + self.inline_math_delimiters[1]
            + " "
        )

    def convert_table(self, el, text, parent_tags):
        return "\n\n" + str(el) + "\n\n"

    def convert_a(self, el, text, parent_tags):
        text = self.escape(text)
        text = re.sub(r"([\[\]()])", r"\\\1", text)
        return super().convert_a(el, text, parent_tags)

    def escape(self, text, parent_tags=None):
        text = super().escape(text, parent_tags)
        if self.options["escape_dollars"]:
            text = text.replace("$", r"\$")
        return text

    def process_text(self, el, parent_tags=None):
        text = six.text_type(el) or ""
        if not el.find_parent("pre"):
            text = re_whitespace.sub(" ", text)
        if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
            text = self.escape(text)
        if el.parent.name == "li" and (
            not el.next_sibling or getattr(el.next_sibling, "name", None) in ["ul", "ol"]
        ):
            text = text.rstrip()
        return text


def make_markdown_converter() -> Markdownify:
    return Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )


def parse_markdown(
    html: str,
    include_headers_footers: bool = False,
    include_images: bool = True,
    page_number: Optional[int] = None,
) -> str:
    html = parse_html(
        html,
        include_headers_footers=include_headers_footers,
        include_images=include_images,
        page_number=page_number,
    )
    try:
        markdown = make_markdown_converter().convert(html)
    except Exception:
        markdown = ""
    return markdown.strip()


def convert_fragment_to_markdown(html: str) -> str:
    html = strip_page_break_markup(html)
    try:
        return make_markdown_converter().convert(html).strip()
    except Exception:
        return html.strip()


@dataclass
class LayoutBlock:
    bbox: list[int]
    label: str
    content: str


def parse_layout(html: str, image: Image.Image, bbox_scale: int = 1000) -> list[LayoutBlock]:
    html = strip_page_break_markup(html)
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    width, height = image.size
    width_scaler = width / bbox_scale
    height_scaler = height / bbox_scale
    layout_blocks: list[LayoutBlock] = []

    for div in top_level_divs:
        label = div.get("data-label")
        if label == "Blank-Page":
            continue

        bbox_raw = div.get("data-bbox")
        try:
            bbox = list(map(int, str(bbox_raw).split(" ")))
            assert len(bbox) == 4
        except Exception:
            bbox = [0, 0, 1, 1]

        bbox = [
            max(0, int(bbox[0] * width_scaler)),
            max(0, int(bbox[1] * height_scaler)),
            min(int(bbox[2] * width_scaler), width),
            min(int(bbox[3] * height_scaler), height),
        ]

        content = str(div.decode_contents())
        content_soup = BeautifulSoup(content, "html.parser")
        for tag in content_soup.find_all(attrs={"data-bbox": True}):
            del tag["data-bbox"]
        layout_blocks.append(
            LayoutBlock(
                bbox=bbox,
                label=label or "block",
                content=str(content_soup),
            )
        )

    return layout_blocks


def parse_chunks(html: str, image: Image.Image, bbox_scale: int = 1000) -> list[dict]:
    return [asdict(block) for block in parse_layout(html, image, bbox_scale=bbox_scale)]


def plain_text(html: str) -> str:
    text = convert_fragment_to_markdown(html)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_table(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    return str(table) if table else html.strip()


def figure_html(
    figure_id: str,
    src: str,
    alt: str,
    title: str,
    caption: str,
) -> str:
    alt_attr = html_escape(alt, quote=True)
    title_attr = html_escape(title, quote=True)
    caption_html = html_escape(caption) if caption else ""
    out = [f'<figure id="{figure_id}">']
    out.append(f'<img src="{src}" alt="{alt_attr}" title="{title_attr}">')
    if caption_html:
        out.append(f"<figcaption>{caption_html}</figcaption>")
    out.append("</figure>")
    return "\n".join(out)
