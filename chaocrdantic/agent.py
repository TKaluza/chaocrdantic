"""pydantic-ai based OCR agent for chaocrdantic."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from chaocrdantic.config import ChaocrdanticSettings, default_settings
from chaocrdantic.image_utils import RenderedPage, load_file_pages, prepare_image_for_inference
from chaocrdantic.layout_analysis import analyze_layouts
from chaocrdantic.models import (
    ExtractedImage,
    LayoutBlock,
    LayoutSummary,
    OCRPageResult,
    OCRResult,
    PageDimensions,
)
from chaocrdantic.output import (
    convert_fragment_to_markdown,
    extract_images,
    parse_chunks,
    parse_html,
    parse_markdown,
)
from chaocrdantic.prompts import OCR_LAYOUT_PROMPT, OCR_PROMPT
from chaocrdantic.util import detect_repeat_token


logger = logging.getLogger(__name__)


class ChaocrdanticAgent:
    """Standalone OCR agent implemented on top of pydantic-ai."""

    def __init__(
        self,
        settings: ChaocrdanticSettings = default_settings,
        use_layout: bool = True,
    ) -> None:
        self.settings = settings
        self.use_layout = use_layout
        self.include_images = settings.INCLUDE_IMAGES
        self.include_headers_footers = settings.INCLUDE_HEADERS_FOOTERS

        model = OpenAIChatModel(
            settings.MODEL_NAME,
            provider=OpenAIProvider(
                base_url=settings.BASE_URL,
                api_key=settings.API_KEY,
            ),
        )
        self._agent: Agent[None, str] = Agent(
            model=model,
            output_type=str,
            system_prompt=(
                "You are an expert OCR system. "
                "Convert document images to accurate, structured HTML."
            ),
        )
        self._prompt = OCR_LAYOUT_PROMPT if use_layout else OCR_PROMPT

    def _normalize_pages(self, pages: Iterable[RenderedPage | object]) -> list[RenderedPage]:
        normalized: list[RenderedPage] = []
        for idx, page in enumerate(pages):
            if isinstance(page, RenderedPage):
                normalized.append(page)
            else:
                normalized.append(RenderedPage(page_number=idx, image=page, dpi=None))
        return normalized

    def _model_settings(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ):
        return ModelSettings(
            max_tokens=self.settings.MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens,
            temperature=self.settings.TEMPERATURE if temperature is None else temperature,
            top_p=self.settings.TOP_P if top_p is None else top_p,
            timeout=self.settings.REQUEST_TIMEOUT,
        )

    async def _request_page(
        self,
        page: RenderedPage,
        temperature: float,
        top_p: float,
        *,
        max_tokens: int,
    ):
        return await self._request_image(
            page.image,
            self._prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    async def _request_image(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        image_b64 = prepare_image_for_inference(
            image,
            max_size=(
                self.settings.INFERENCE_MAX_IMAGE_WIDTH,
                self.settings.INFERENCE_MAX_IMAGE_HEIGHT,
            ),
            min_size=(
                self.settings.INFERENCE_MIN_IMAGE_WIDTH,
                self.settings.INFERENCE_MIN_IMAGE_HEIGHT,
            ),
        )
        user_content = [
            BinaryContent(
                data=base64.b64decode(image_b64),
                media_type="image/png",
                vendor_metadata={"detail": "auto"},
            ),
            prompt,
        ]
        result = await self._agent.run(
            user_content,
            model_settings=self._model_settings(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
        )
        raw_html = result.output.strip()
        token_count = getattr(result.usage(), "output_tokens", 0)
        return raw_html, token_count

    async def _ocr_text_only(self, image: Image.Image) -> str:
        raw_html, _ = await self._request_image(
            image,
            OCR_PROMPT,
            temperature=self.settings.TEMPERATURE,
            top_p=self.settings.TOP_P,
            max_tokens=self.settings.MAX_OUTPUT_TOKENS,
        )
        return convert_fragment_to_markdown(raw_html)

    def _should_retry(self, raw_html: str, error: Exception | None, retries: int) -> bool:
        if retries >= self.settings.MAX_RETRIES:
            return False
        if self._is_context_overflow_error(error):
            return False
        if error is not None:
            return True
        if not raw_html.strip():
            return True
        if detect_repeat_token(raw_html) or (len(raw_html) > 50 and detect_repeat_token(raw_html, cut_from_end=50)):
            return True
        return False

    @staticmethod
    def _is_context_overflow_error(error: Exception | None) -> bool:
        if error is None:
            return False
        message = str(error).lower()
        return (
            "context size has been exceeded" in message
            or "maximum context length" in message
            or "context length exceeded" in message
        )

    def _build_page_result(
        self,
        page: RenderedPage,
        raw_html: str,
        token_count: int,
    ) -> OCRPageResult:
        if self.use_layout:
            chunks = parse_chunks(raw_html, page.image)
            pil_images, image_metadata = extract_images(
                raw_html,
                chunks,
                page.image,
                page_number=page.page_number,
            )
            layout_blocks = [LayoutBlock(**chunk) for chunk in chunks]
            extracted_images = [ExtractedImage(**meta) for meta in image_metadata]
            html = parse_html(
                raw_html,
                include_headers_footers=self.include_headers_footers,
                include_images=self.include_images,
                page_number=page.page_number,
            )
            markdown = parse_markdown(
                raw_html,
                include_headers_footers=self.include_headers_footers,
                include_images=self.include_images,
                page_number=page.page_number,
            )
        else:
            layout_blocks = []
            extracted_images = []
            pil_images = {}
            html = raw_html.strip()
            markdown = convert_fragment_to_markdown(raw_html)

        page_result = OCRPageResult(
            page_number=page.page_number,
            raw_html=raw_html,
            markdown=markdown,
            html=html,
            dimensions=PageDimensions(
                dpi=page.dpi,
                width=page.image.width,
                height=page.image.height,
            ),
            token_count=token_count,
            layout_blocks=layout_blocks,
            extracted_images=extracted_images,
            error=False,
        )
        page_result.set_extracted_images(pil_images)
        return page_result

    async def _process_page(self, page: RenderedPage, semaphore: asyncio.Semaphore) -> OCRPageResult:
        async with semaphore:
            retries = 0
            raw_html = ""
            token_count = 0
            error: Exception | None = None

            while True:
                try:
                    temperature = min(self.settings.TEMPERATURE + 0.2 * retries, 0.8)
                    top_p = 0.95 if retries else self.settings.TOP_P
                    raw_html, token_count = await self._request_page(
                        page,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=self.settings.MAX_OUTPUT_TOKENS,
                    )
                    error = None
                except Exception as exc:
                    error = exc
                    if self._is_context_overflow_error(error):
                        logger.warning(
                            "Page %s exceeded the model context window at max_tokens=%s and "
                            "inference image size %sx%s. Increase the server context size or "
                            "use smaller inference image settings.",
                            page.page_number + 1,
                            self.settings.MAX_OUTPUT_TOKENS,
                            self.settings.INFERENCE_MAX_IMAGE_WIDTH,
                            self.settings.INFERENCE_MAX_IMAGE_HEIGHT,
                        )

                if not self._should_retry(raw_html, error, retries):
                    break

                retries += 1
                await asyncio.sleep(min(2 * retries, 10))

            if error is not None:
                return OCRPageResult(
                    page_number=page.page_number,
                    raw_html="",
                    markdown="",
                    html="",
                    dimensions=PageDimensions(
                        dpi=page.dpi,
                        width=page.image.width,
                        height=page.image.height,
                    ),
                    token_count=0,
                    layout_blocks=[],
                    extracted_images=[],
                    error=True,
                    error_message=str(error),
                )

            if (not raw_html.strip()) or detect_repeat_token(raw_html) or (
                len(raw_html) > 50 and detect_repeat_token(raw_html, cut_from_end=50)
            ):
                return OCRPageResult(
                    page_number=page.page_number,
                    raw_html=raw_html,
                    markdown="",
                    html="",
                    dimensions=PageDimensions(
                        dpi=page.dpi,
                        width=page.image.width,
                        height=page.image.height,
                    ),
                    token_count=token_count,
                    layout_blocks=[],
                    extracted_images=[],
                    error=True,
                    error_message="Model returned empty or degenerate output after retries.",
                )

            return self._build_page_result(page=page, raw_html=raw_html, token_count=token_count)

    def run_pages(
        self,
        pages: list,
        file_path: str = "<unknown>",
    ) -> OCRResult:
        return asyncio.run(self.run_pages_async(pages, file_path=file_path))

    async def run_pages_async(
        self,
        pages: list,
        file_path: str = "<unknown>",
    ) -> OCRResult:
        rendered_pages = self._normalize_pages(pages)
        if self.settings.CLEAN_MARKDOWN and len(rendered_pages) >= self.settings.CLEAN_MARKDOWN_MIN_PAGES:
            try:
                return await self._run_clean_markdown_async(rendered_pages, file_path=file_path)
            except Exception:
                logger.warning(
                    "--clean-markdown layout analysis failed; falling back to standard pipeline.",
                    exc_info=True,
                )
        elif self.settings.CLEAN_MARKDOWN:
            logger.warning(
                "--clean-markdown requested but only %d page(s); falling back to standard pipeline.",
                len(rendered_pages),
            )

        return await self._run_standard_pages_async(rendered_pages, file_path=file_path)

    async def _run_standard_pages_async(
        self,
        rendered_pages: list[RenderedPage],
        *,
        file_path: str,
    ) -> OCRResult:
        semaphore = asyncio.Semaphore(max(1, self.settings.MAX_WORKERS))
        tasks = [self._process_page(page, semaphore) for page in rendered_pages]
        page_results = await asyncio.gather(*tasks)
        page_results.sort(key=lambda page: page.page_number)

        return OCRResult(
            file_path=os.path.abspath(file_path),
            ocr_engine="chaocrdantic",
            ocr_model=self.settings.MODEL_NAME,
            num_pages=len(rendered_pages),
            pages=page_results,
        )

    async def _run_clean_markdown_async(
        self,
        pages: list[RenderedPage],
        *,
        file_path: str,
    ) -> OCRResult:
        analysis = analyze_layouts(
            pages,
            layout_threshold=self.settings.CLEAN_MARKDOWN_LAYOUT_THRESHOLD,
        )
        page_by_number = {page.page_number: page for page in pages}
        layout_by_page = {page.page_number: page for page in analysis.pages}
        semaphore = asyncio.Semaphore(max(1, self.settings.MAX_WORKERS))

        async def crop_text(box: tuple[int, int, int, int] | None, page: RenderedPage) -> str | None:
            if box is None:
                return None
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                return None
            async with semaphore:
                return (await self._ocr_text_only(page.image.crop(box))).strip() or None

        async def ocr_layout(layout) -> tuple[int, str | None, str | None]:
            representative = page_by_number[layout.representative_page]
            header, footer = await asyncio.gather(
                crop_text(layout.header_box, representative),
                crop_text(layout.footer_box, representative),
            )
            return layout.layout_id, header, footer

        layout_text_results = await asyncio.gather(*(ocr_layout(layout) for layout in analysis.layouts))
        text_by_layout = {
            layout_id: (header_text, footer_text)
            for layout_id, header_text, footer_text in layout_text_results
        }

        async def process_main(page: RenderedPage) -> OCRPageResult:
            page_layout = layout_by_page[page.page_number]
            crop = page.image.crop(page_layout.main_box)
            cropped_page = RenderedPage(page_number=page.page_number, image=crop, dpi=page.dpi)
            result = await self._process_page(cropped_page, semaphore)
            header_text, footer_text = text_by_layout.get(page_layout.layout_id, (None, None))
            result.layout_id = page_layout.layout_id
            result.header_text = header_text
            result.footer_text = footer_text
            result.dimensions = PageDimensions(
                dpi=page.dpi,
                width=page.image.width,
                height=page.image.height,
            )
            return result

        page_results = await asyncio.gather(*(process_main(page) for page in pages))
        page_results.sort(key=lambda page: page.page_number)

        layout_summaries = [
            LayoutSummary(
                layout_id=layout.layout_id,
                page_numbers=layout.page_numbers,
                header_text=text_by_layout.get(layout.layout_id, (None, None))[0],
                footer_text=text_by_layout.get(layout.layout_id, (None, None))[1],
                is_outlier=layout.is_outlier,
                representative_page=layout.representative_page,
            )
            for layout in analysis.layouts
        ]

        return OCRResult(
            file_path=os.path.abspath(file_path),
            ocr_engine="chaocrdantic",
            ocr_model=self.settings.MODEL_NAME,
            num_pages=len(pages),
            pages=page_results,
            layouts=layout_summaries,
        )

    def run_file(
        self,
        file_path: str | Path | bytes,
        page_range: Optional[list[int]] = None,
    ) -> OCRResult:
        return asyncio.run(self.run_file_async(file_path, page_range=page_range))

    async def run_file_async(
        self,
        file_path: str | Path | bytes,
        page_range: Optional[list[int]] = None,
    ) -> OCRResult:
        pages = load_file_pages(
            file_path,
            page_range=page_range,
            settings=self.settings,
        )
        result_file_path = (
            os.path.abspath("in-memory.pdf")
            if isinstance(file_path, (bytes, bytearray, memoryview))
            else file_path
        )
        return await self.run_pages_async(pages, file_path=result_file_path)


ChandraOCRAgent = ChaocrdanticAgent
