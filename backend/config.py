from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "AI Indian Rail Travel Assistant"
    app_version: str = "1.0.0"

    # Primary LLM (Gemini)
    gemini_api_key: str
    gemini_model: str = "gemini-3.1-flash-lite"

    # Fallback LLM (Groq) — used automatically when Gemini hits quota
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()