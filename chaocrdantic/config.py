"""Configuration for the chaocrdantic OCR agent."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ChaocrdanticSettings(BaseSettings):
    """Settings for the ChaocrdanticAgent."""

    model_config = SettingsConfigDict(
        env_prefix="CHAOCRDANTIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider / model
    # The default model name matches the upstream Chandra deployment.
    MODEL_NAME: str = "chandra-ocr-2-vllm"
    BASE_URL: str = "http://127.0.0.1:12434/v1"
    API_KEY: str = "EMPTY"

    # Inference parameters
    MAX_OUTPUT_TOKENS: int = 12384
    TEMPERATURE: float = 0.0
    TOP_P: float = 0.1
    MAX_RETRIES: int = 6
    MAX_WORKERS: int = 10
    REQUEST_TIMEOUT: float = 500.0

    # Output controls
    INCLUDE_IMAGES: bool = True
    INCLUDE_HEADERS_FOOTERS: bool = False

    # Image rendering
    IMAGE_DPI: int = 192
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536
    INFERENCE_MAX_IMAGE_WIDTH: int = 3072
    INFERENCE_MAX_IMAGE_HEIGHT: int = 2048
    INFERENCE_MIN_IMAGE_WIDTH: int = 1792
    INFERENCE_MIN_IMAGE_HEIGHT: int = 28


# Module-level singleton loaded once at import time.
settings = ChaocrdanticSettings()
default_settings = settings

ChandraOCRSettings = ChaocrdanticSettings
