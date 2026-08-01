# Legal Contract Intelligence API — v1 (Refactored)

Production-quality, modular rebuild of the original single-file FastAPI backend.
All original functionality is preserved (OCR, Gemini analysis, translation, TTS,
report generation, downloads) — only the architecture and response design changed.

See `docs/API_DOCUMENTATION.md` for full endpoint/request/response documentation.

## Project layout

```
legal_backend/
├── app.py                     # FastAPI app: CORS, exception handlers, router wiring only
├── config.py                  # All environment variables / constants, read once
├── logging_config.py
├── requirements.txt
├── .env.example
├── models/
│   ├── enums.py                # DownloadFormat, OCRQuality, RiskLevel, Severity, ProcessingStatus, ErrorCode...
│   └── schemas.py               # All Pydantic request/response models
├── services/
│   ├── ocr_service.py           # PyMuPDF -> Google Vision -> PyPDF2 fallback chain + quality scoring
│   ├── ai_service.py            # Gemini calls, structured-JSON normalization, comparison, clause extraction
│   ├── report_service.py        # txt/json/markdown/pdf report generation
│   ├── translation_service.py   # Sarvam AI translation
│   ├── audio_service.py         # Sarvam AI text-to-speech
│   ├── privacy_service.py       # PII masking, DLP-swappable strategy pattern
│   └── cache_service.py         # In-memory analysis/audio cache (DB-swappable interface)
├── routers/
│   ├── analyze.py               # POST /analyze  (+ shared run_full_analysis() used by upload.py)
│   ├── upload.py                # POST /upload, POST /ocr-check
│   ├── compare.py               # POST /compare
│   ├── clauses.py               # POST /extract-clauses
│   ├── translate.py             # POST /translate
│   ├── tts.py                   # POST /tts, POST /generate-audio/{analysis_id}
│   ├── downloads.py             # GET /download/text/{id}, GET /download/audio/{filename}
│   └── health.py                # GET /, GET /health
└── docs/
    └── API_DOCUMENTATION.md
```

## Setup

```bash
cd legal_backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY, SARVAM_API_KEY, Google credentials
uvicorn app:app --reload --port 8000
```

Then open `http://localhost:8000/docs`.

## What changed vs. the original

| Area | Before | After |
|---|---|---|
| Response shape | Ad-hoc keys per endpoint | `{success, data, metadata}` / `{success:false, error:{code,message,suggestion}}` everywhere |
| `red_flags` / `recommendations` / etc. | Plain string arrays | Structured objects with `id`, `severity`, `category`, `explanation`, `recommendation` |
| Clauses | Not exposed | Full `clauses[]` array with per-clause risk score + reasoning |
| Risk | `risk_level` + `risk_score` strings | `risk{score, level, confidence, breakdown{...}}` |
| Errors | Mixed `HTTPException` details | Standardized `ErrorCode` enum + suggestion field |
| Routing | `/api/...` only | `/api/v1/...` (primary) **and** `/api/...` (legacy, still works) |
| Code layout | One ~900-line file | Routers / services / models separated, `app.py` is routing-only |
| PII masking | Inline regex | `PrivacyService` strategy pattern — DLP is a drop-in future replacement |
| Cache | Two global dicts | `CacheService` class — swappable for a real DB later |

## Verified

The refactor was smoke-tested end-to-end with `fastapi.testclient.TestClient`, including:
- all routes resolving under both `/api/v1` and legacy `/api`
- validation errors returning the standardized envelope
- 404-style "analysis not found" flows
- a full mocked-Gemini `/analyze` call producing correctly ID-tagged, schema-validated JSON
- all four report downloads (txt/json/md/pdf) generating from a cached analysis

Live calls to Gemini/Sarvam were not exercised in the sandbox (no outbound network access to
those hosts there) — supply real API keys and they'll work identically to the original code,
since the underlying request/response logic against those APIs is unchanged.
