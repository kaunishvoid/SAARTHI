import os
import secrets
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
if os.environ.get("APP_ENV", "development").lower() == "production" and not os.environ.get("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be set in production.")


def database_url() -> str:
    value = os.environ.get("DATABASE_URL", "sqlite:///saarthi.db")
    # Accept the common provider URL scheme while selecting psycopg 3 explicitly.
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    return value


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_urlsafe(48)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get(
        "SESSION_COOKIE_SECURE", "true" if os.environ.get("APP_ENV", "development").lower() == "production" else "false"
    ).lower() == "true"
    SESSION_COOKIE_NAME = "saarthi_session"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 12
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    SQLALCHEMY_DATABASE_URI = database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    CROP_MODEL_ARTIFACT_DIR = os.environ.get(
        "CROP_MODEL_ARTIFACT_DIR",
        str(PROJECT_ROOT / "backend" / "app" / "ml" / "artifacts"),
    )
    FERTILIZER_MODEL_ARTIFACT_DIR = os.environ.get(
        "FERTILIZER_MODEL_ARTIFACT_DIR",
        str(PROJECT_ROOT / "backend" / "app" / "ml" / "fertilizer_artifacts"),
    )
