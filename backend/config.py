from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SkillWatch AI"
    app_version: str = "0.1.0"
    camera_source: str = "device"
    camera_index: int = 0
    confidence_threshold: float = 0.5
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/skillwatch"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
