import asyncio
import base64
import io

from PIL import Image

from chaocrdantic.agent import ChaocrdanticAgent
from chaocrdantic.config import ChaocrdanticSettings, default_settings, settings
from chaocrdantic.image_utils import RenderedPage, prepare_image_for_inference


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
