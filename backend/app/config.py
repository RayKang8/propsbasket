from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://propsbasket:propsbasket@localhost:5432/propsbasket"
    odds_api_key: str = ""
    cors_origins: list[str] = ["http://localhost:3000"]
    debug: bool = False


settings = Settings()
