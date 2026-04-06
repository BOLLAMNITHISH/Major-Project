from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Brain Tumor ViT API"
    app_version: str = "1.0.0"

    # SQLite (async)
    database_url: str = "sqlite+aiosqlite:///./data/predictions.db"

    # PyTorch ViT (.pth from model/train_vit_pytorch.py — best_model.pth / last_model.pth)
    vit_checkpoint_path: str | None = None

    # Optional: JSON from train_vit_pytorch.py (defaults to next to checkpoint or ./data/)
    evaluation_report_path: str | None = None

    @field_validator("vit_checkpoint_path", "evaluation_report_path", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str | None) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    uploads_dir: Path = Path("./uploads")
    max_upload_mb: int = 16

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
