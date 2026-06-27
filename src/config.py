import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration class using Pydantic.
    Automatically loads environment variables from a .env file,
    validates required fields, and provides type casting.
    """
    # LLM Providers (parsed as comma-separated lists in our rotator)
    GEMINI_API_KEYS: str
    GROQ_API_KEYS: str

    # Model Configurations
    GROQ_TEXT_MODEL: str = "llama3-70b-8192"
    GROQ_VISION_MODEL: str = "llama-3.2-11b-vision-preview"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"

    # Vector DB
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "sylvi-memory"

    # Telegram
    TELEGRAM_BOT_TOKEN: str

    # Application
    LOG_LEVEL: str = "INFO"
    DATABASE_PATH: str = "data/sylvi_profile.db"
    DATABASE_URL: str = ""

    # Automatically load from .env file if it exists
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra env vars
    )

# Instantiate a single global settings object
# If any key is missing, this will raise a ValidationError on import.
settings = Settings()
