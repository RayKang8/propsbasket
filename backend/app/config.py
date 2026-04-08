from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://propsbasket:propsbasket@localhost:5432/propsbasket"
    odds_api_key: str = ""
    cors_origins: list[str] = ["http://localhost:3000"]
    debug: bool = False

    # ML artifacts and data paths (relative to backend working directory)
    model_path: str = "../ml/models/artifacts/xgboost.joblib"
    game_logs_path: str = "../ml/data/raw/game_logs_2024-25.parquet"
    team_stats_path: str = "../ml/data/raw/team_stats_2024-25.parquet"


settings = Settings()
