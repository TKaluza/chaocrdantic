import asyncio
import base64
import io
import logging
from pathlib import Path

from PIL import Image

from chaocrdantic import agent as agent_module
from chaocrdantic.agent import ChaocrdanticAgent
from chaocrdantic import api
from chaocrdantic.config import ChaocrdanticSettings, default_settings, settings
from chaocrdantic.image_utils import RenderedPage, load_file_pages, prepare_image_for_inference
from chaocrdantic.layout_analysis import LayoutAnalysisResult, LayoutCluster, PageLayout


def test_model_settings_pass_max_tokens():
    agent = ChaocrdanticAgent(
        settings=ChaocrdanticSettings(MAX_OUTPUT_TOKENS=777, TEMPERATURE=0.2, TOP_P=0.3),
        use_layout=False,
    )
    settings = agent._model_settings()
    assert settings["max_tokens"] == 777
    assert settings["temperature"] == 0.2
    assert settings["top_p"] == 0.3


def test_process_page_retries_on_exception():
    attempts = {"count": 0}
    agent = ChaocrdanticAgent(
        settings=ChaocrdanticSettings(MAX_RETRIES=2, MAX_WORKERS=1),
        use_layout=False,
    )
    page = RenderedPage(page_number=0, image=Image.new("RGB", (100, 100), "white"), dpi=200)

    async def fake_request(page, temperature, top_p, *, max_tokens):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("boom")
        assert max_tokens == agent.settings.MAX_OUTPUT_TOKENS
        return "<p>ok</p>", 11

    agent._request_page = fake_request
    result = asyncio.run(agent._process_page(page, asyncio.Semaphore(1)))

    assert attempts["count"] == 2
    assert result.error is False
    assert result.markdown


def test_process_page_retries_on_repeat_token_output():
    attempts = {"count": 0}
    agent = ChaocrdanticAgent(
        settings=ChaocrdanticSettings(MAX_RETRIES=2, MAX_WORKERS=1),
        use_layout=False,
    )
    page = RenderedPage(page_number=0, image=Image.new("RGB", (100, 100), "white"), dpi=200)

    async def fake_request(page, temperature, top_p, *, max_tokens):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return "abcabcabcabcabcabcabcabcabcabc", 10
        assert max_tokens == agent.settings.MAX_OUTPUT_TOKENS
        return "<p>stable output</p>", 12

    agent._request_page = fake_request
    result = asyncio.run(agent._process_page(page, asyncio.Semaphore(1)))

    assert attempts["count"] == 2
    assert result.error is False
    assert "stable output" in result.markdown


def test_process_page_does_not_retry_on_context_overflow():
    attempts = {"count": 0}
    agent = ChaocrdanticAgent(
        settings=ChaocrdanticSettings(MAX_RETRIES=2, MAX_WORKERS=1, MAX_OUTPUT_TOKENS=8000),
        use_layout=False,
    )
    page = RenderedPage(page_number=0, image=Image.new("RGB", (100, 100), "white"), dpi=200)

    async def fake_request(page, temperature, top_p, *, max_tokens):
        attempts["count"] += 1
        raise RuntimeError("Context size has been exceeded.")

    agent._request_page = fake_request
    result = asyncio.run(agent._process_page(page, asyncio.Semaphore(1)))

    assert result.error is True
    assert attempts["count"] == 1
    assert "Context size has been exceeded." in (result.error_message or "")


def test_prepare_image_for_inference_respects_custom_max_size():
    image = Image.new("RGB", (4000, 3000), "white")

    encoded = prepare_image_for_inference(
        image,
        max_size=(1024, 1024),
        min_size=(28, 28),
    )
    scaled = Image.open(io.BytesIO(base64.b64decode(encoded)))

    assert scaled.width * scaled.height <= 1024 * 1024


def test_agent_uses_plain_openai_chat_model():
    assert ChaocrdanticAgent()._agent.model.__class__.__name__ == "OpenAIChatModel"


def test_settings_aliases_point_to_same_singleton():
    assert settings is default_settings


def test_settings_can_load_prefilled_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "CHAOCRDANTIC_MODEL_NAME=test-model",
                "CHAOCRDANTIC_BASE_URL=http://localhost:9999/v1",
                "CHAOCRDANTIC_MAX_WORKERS=3",
            ]
        ),
        encoding="utf-8",
    )

    loaded = ChaocrdanticSettings(_env_file=env_file)

    assert loaded.MODEL_NAME == "test-model"
    assert loaded.BASE_URL == "http://localhost:9999/v1"
    assert loaded.MAX_WORKERS == 3


def test_load_file_pages_accepts_pdf_bytes():
    pdf_bytes = Path("tests/fixtures/DE102022115220A1.pdf").read_bytes()

    pages = load_file_pages(pdf_bytes, page_range=[0], settings=default_settings)

    assert len(pages) == 1
    assert pages[0].page_number == 0
    assert pages[0].image.width > 0
    assert pages[0].image.height > 0


def test_ocr_file_accepts_bytes_input(monkeypatch):
    calls = []

    class FakeAgent:
        def run_file(self, file_path, page_range=None):
            calls.append(("sync", file_path, page_range))
            return "sync-result"

        async def run_file_async(self, file_path, page_range=None):
            calls.append(("async", file_path, page_range))
            return "async-result"

    monkeypatch.setattr(api, "_get_agent", lambda: FakeAgent())

    pdf_bytes = Path("tests/fixtures/DE102022115220A1.pdf").read_bytes()

    assert api.ocr_file(pdf_bytes, page_range=[0]) == "sync-result"
    assert asyncio.run(api.ocr_file_async(pdf_bytes, page_range=[1])) == "async-result"
    assert calls[0][0] == "sync"
    assert calls[0][1] == pdf_bytes
    assert calls[0][2] == [0]
    assert calls[1][0] == "async"
    assert calls[1][1] == pdf_bytes
    assert calls[1][2] == [1]


def test_clean_markdown_ocr_header_footer_once_per_layout(monkeypatch):
    pages = [
        RenderedPage(page_number=index, image=Image.new("RGB", (200, 300), "white"), dpi=200)
        for index in range(3)
    ]
    analysis = LayoutAnalysisResult(
        pages=[
            PageLayout(
                page_number=index,
                layout_id=1,
                header_box=(0, 0, 200, 40),
                main_box=(0, 40, 200, 260),
                footer_box=(0, 260, 200, 300),
                is_outlier=False,
            )
            for index in range(3)
        ],
        layouts=[
            LayoutCluster(
                layout_id=1,
                page_numbers=[0, 1, 2],
                header_box=(0, 0, 200, 40),
                main_box=(0, 40, 200, 260),
                footer_box=(0, 260, 200, 300),
                is_outlier=False,
                representative_page=0,
            )
        ],
    )
    monkeypatch.setattr(agent_module, "analyze_layouts", lambda pages, **kwargs: analysis)

    ocr_text_calls = []
    main_calls = []
    agent = ChaocrdanticAgent(
        settings=ChaocrdanticSettings(CLEAN_MARKDOWN=True, MAX_WORKERS=2),
        use_layout=False,
    )

    async def fake_ocr_text_only(image):
        ocr_text_calls.append(image.size)
        return "HEADER" if len(ocr_text_calls) == 1 else "FOOTER"

    async def fake_request_page(page, temperature, top_p, *, max_tokens):
        main_calls.append((page.page_number, page.image.size))
        return f"<p>body {page.page_number}</p>", 10

    agent._ocr_text_only = fake_ocr_text_only
    agent._request_page = fake_request_page

    result = asyncio.run(agent.run_pages_async(pages, file_path="sample.pdf"))

    assert ocr_text_calls == [(200, 40), (200, 40)]
    assert main_calls == [(0, (200, 220)), (1, (200, 220)), (2, (200, 220))]
    assert result.layouts is not None
    assert len(result.layouts) == 1
    assert result.layouts[0].header_text == "HEADER"
    assert result.layouts[0].footer_text == "FOOTER"
    assert [page.layout_id for page in result.pages] == [1, 1, 1]
    assert all(page.header_text == "HEADER" for page in result.pages)
    assert all(page.footer_text == "FOOTER" for page in result.pages)
    assert all("HEADER" not in page.markdown and "FOOTER" not in page.markdown for page in result.pages)
    assert all(page.dimensions.height == 300 for page in result.pages)


def test_clean_markdown_single_page_falls_back_to_standard(caplog):
    page = RenderedPage(page_number=0, image=Image.new("RGB", (100, 100), "white"), dpi=200)
    agent = ChaocrdanticAgent(
        settings=ChaocrdanticSettings(CLEAN_MARKDOWN=True, MAX_WORKERS=1),
        use_layout=False,
    )
    calls = {"count": 0}

    async def fake_request_page(page, temperature, top_p, *, max_tokens):
        calls["count"] += 1
        return "<p>standard</p>", 5

    agent._request_page = fake_request_page

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(agent.run_pages_async([page], file_path="single.pdf"))

    assert calls["count"] == 1
    assert result.layouts is None
    assert "falling back to standard pipeline" in caplog.text
    assert "standard" in result.pages[0].markdown
