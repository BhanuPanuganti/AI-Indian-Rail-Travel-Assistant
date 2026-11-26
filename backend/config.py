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

    # Optional live running status (RailRadar). Disabled when the key is empty.
    railradar_api_key: str = ""
    railradar_base_url: str = "https://api.railradar.in"
    # Stop calling once this many requests were made in the current month by
    # this process, or when RailRadar reports this few requests left.
    railradar_monthly_budget: int = 90
    railradar_min_remaining: int = 5
    railradar_cache_ttl_seconds: int = 300
    railradar_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()