"""
Central configuration for the Legal Contract Intelligence API.
All environment variables and constants are read here ONCE so the
rest of the codebase never touches os.environ directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ---- Core services ----
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    GOOGLE_CLOUD_CREDENTIALS: str = os.getenv("GOOGLE_CLOUD_CREDENTIALS", "")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    # ---- App ----
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    BACKEND_VERSION: str = os.getenv("BACKEND_VERSION", "v1")
    API_PREFIX: str = "/api/v1"
    LEGACY_API_PREFIX: str = "/api"  # kept for backward compatibility

    # ---- Files ----
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    TEMP_DIR: str = "temp"
    UPLOAD_DIR: str = "uploads"
    DOWNLOADS_DIR: str = "downloads"

    # ---- OCR ----
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75"))

    # ---- Gemini ----
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")

    # ---- CORS ----
    ALLOWED_ORIGINS = [
        "https://legal-backend-gold.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ]

    # ---- Cache ----
    CACHE_TTL_HOURS: int = 24


settings = Settings()

if not settings.GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in environment variables")

for _dir in [settings.TEMP_DIR, settings.UPLOAD_DIR, settings.DOWNLOADS_DIR]:
    os.makedirs(_dir, exist_ok=True)
