"""
Configuration for the pydantic-ai OCR agent.

Defaults target a locally-running chandra-ocr-2 model server at
http://127.0.0.1:12434/ but every value is overridable via constructor
arguments or environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ChandraOCRSettings(BaseSettings):
    """Settings for the ChandraOCRAgent."""

    model_config = SettingsConfigDict(
        env_prefix="CHANDRA_PYDANTIC_",
        extra="ignore",
    )

    # Provider / model
    MODEL_NAME: str = "chandra-ocr-2"
    BASE_URL: str = "http://127.0.0.1:12434/v1"
    API_KEY: str = "EMPTY"

    # Inference parameters
    MAX_OUTPUT_TOKENS: int = 12384
    TEMPERATURE: float = 0.0
    TOP_P: float = 0.1

    # Image rendering
    IMAGE_DPI: int = 192
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536


# Module-level singleton — override by passing settings= to the agent
default_settings = ChandraOCRSettings()
