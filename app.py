"""
Legal Contract Intelligence API - v1 (refactored)

Entry point. Keeps app.py focused purely on wiring:
  - CORS
  - exception -> standardized error envelope
  - router registration (both the new /api/v1 prefix and the legacy
    /api prefix, for backward compatibility with existing frontends)

All business logic lives in services/, all schemas in models/,
all routes in routers/.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from logging_config import configure_logging, get_logger
from utils.response import APIError, error_response
from models.enums import ErrorCode

from routers import health, analyze, upload, compare, clauses, translate, tts, downloads

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Legal Contract Intelligence API",
    description=(
        "AI-powered contract analysis with structured risk data, OCR, "
        "multi-language support, and downloadable reports."
    ),
    version="3.0.0" if settings.BACKEND_VERSION == "v1" else settings.BACKEND_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ---------------- STANDARDIZED ERROR HANDLING -------------
# =========================================================

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    logger.warning(f"APIError: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc.code, exc.message, exc.suggestion),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(x) for x in first.get("loc", [])) or "request"
    return JSONResponse(
        status_code=422,
        content=error_response(
            ErrorCode.VALIDATION_ERROR,
            f"Invalid request: {field} - {first.get('msg', 'validation failed')}",
            "Check the request body against the documented schema at /docs.",
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(ErrorCode.INTERNAL_ERROR, str(exc.detail)),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_response(
            ErrorCode.INTERNAL_ERROR,
            "An unexpected error occurred.",
            "Please retry. If the problem persists, contact support.",
        ),
    )


# =========================================================
# ---------------------- ROUTES -----------------------------
# =========================================================

_versioned_routers = [health, analyze, upload, compare, clauses, translate, tts, downloads]

# health router also mounted at root ("/", "/health") with no prefix
app.include_router(health.router)

for module in [analyze, upload, compare, clauses, translate, tts, downloads]:
    # New, versioned API surface
    app.include_router(module.router, prefix=settings.API_PREFIX)
    # Legacy, unversioned API surface kept for backward compatibility
    app.include_router(module.router, prefix=settings.LEGACY_API_PREFIX, include_in_schema=False)
