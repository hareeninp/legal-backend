"""
Legal Contract Intelligence API - Complete Version with OCR & Downloads
FastAPI backend for AI-powered contract analysis with full features
Enhanced with mandatory document clarity checking
"""

import os
import re
import json
import base64
import logging
import shutil
import tempfile
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from pydantic import BaseModel, Field, validator
from google import genai
from PyPDF2 import PdfReader
from google.cloud import vision
from google.oauth2 import service_account
import io
import fitz
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ====================== LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== ENVIRONMENT ======================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GOOGLE_CLOUD_CREDENTIALS = os.getenv("GOOGLE_CLOUD_CREDENTIALS")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
MAX_FILE_SIZE = 10 * 1024 * 1024
TEMP_DIR = "temp"
UPLOAD_DIR = "uploads"
DOWNLOADS_DIR = "downloads"
for _dir in [TEMP_DIR, UPLOAD_DIR, DOWNLOADS_DIR]:
    os.makedirs(_dir, exist_ok=True)
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75"))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "models/gemini-flash-latest"

vision_client = None
try:
    if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS
        )
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        logger.info("Google Cloud Vision configured from file")
    elif GOOGLE_CLOUD_CREDENTIALS:
        creds_dict = json.loads(GOOGLE_CLOUD_CREDENTIALS)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        logger.info("Google Cloud Vision configured from environment")
except Exception as e:
    logger.warning(f"Google Cloud Vision not configured: {e}")

# ====================== ENUMS ======================
class DownloadFormat(str, Enum):
    TXT = "txt"
    JSON = "json"
    PDF = "pdf"
    MD = "md"

class OCRQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

# ====================== FASTAPI APP ======================
app = FastAPI(
    title="Legal Contract Intelligence API",
    description="AI-powered contract analysis with multi-language support, OCR, and download capabilities",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://signature-gap-frontend-e9gvoko95-nehas-projects-f3c149cb.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== REQUEST MODELS ======================
class DownloadOptions(BaseModel):
    audio: str = ""

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Contract text to analyze")
    contract_name: Optional[str] = Field("Contract", description="Name of the contract")
    user_role: Optional[str] = Field("general", description="User role: employee, employer, freelancer, tenant, landlord")
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) < 50:
            raise ValueError('Contract text must be at least 50 characters')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The employee shall work 40 hours per week...",
                "contract_name": "Employment Agreement",
                "user_role": "employee"
            }
        }

class CompareRequest(BaseModel):
    text1: str = Field(..., min_length=50)
    text2: str = Field(..., min_length=50)
    name1: Optional[str] = Field("Contract 1")
    name2: Optional[str] = Field("Contract 2")
    user_role: Optional[str] = Field("general")

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_language: str = Field(..., pattern="^(hi|ta|te|kn|ml|mr|bn|gu|pa|od)-IN$")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    language: str = Field("en-IN", pattern="^(en|hi|ta|te|kn|ml|mr|bn|gu|pa|od)-IN$")
    speaker: Optional[str] = Field("meera", description="Voice speaker")

class ClauseExtractionRequest(BaseModel):
    text: str = Field(..., min_length=100)
    clause_types: Optional[List[str]] = Field(
        default=["termination", "payment", "confidentiality", "liability", "dispute"],
        description="Types of clauses to extract"
    )

class DownloadTextRequest(BaseModel):
    analysis_id: str = Field(..., description="Analysis ID to download")
    format: DownloadFormat = Field(DownloadFormat.PDF, description="Download format")
    include_questions: bool = Field(True, description="Include questions in download")

# ====================== RESPONSE MODELS ======================
class OCRResult(BaseModel):
    text: str
    confidence: float
    quality: OCRQuality
    needs_reupload: bool
    message: str
    page_count: int
    word_count: int

class DocumentClarityCheckResponse(BaseModel):
    success: bool
    is_clear: bool
    confidence: float
    quality: OCRQuality
    needs_reupload: bool
    message: str
    details: Dict[str, Any]
    suggestions: List[str]

class AnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    questions: List[Any]
    metadata: Dict[str, Any]
    download_options: Dict[str, str]

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str

# ====================== IN-MEMORY STORAGE ======================
analysis_cache: Dict[str, Dict[str, Any]] = {}
audio_cache: Dict[str, str] = {}

# ====================== UTILITY FUNCTIONS ======================
def generate_analysis_id(text: str, contract_name: str) -> str:
    content = f"{text[:100]}{contract_name}{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

def mask_sensitive_data(text: str) -> str:
    patterns = [
        (r"\b\d{10}\b", "[PHONE]"),
        (r"\+91[\s-]?\d{10}", "[PHONE]"),
        (r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN]"),
        (r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b", "[AADHAAR]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b", "[EMAIL]"),
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]"),
        (r"\b[A-Z]{2}\d{2}[A-Z0-9]{13,16}\b", "[IBAN]"),
        (r"\b[A-Z]{6}\d{2}[A-Z0-9]{2}[A-Z0-9]{3}\b", "[IFSC]")
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    return text

def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def assess_ocr_quality(confidence: float) -> Tuple[OCRQuality, bool, str]:
    if confidence >= 0.95:
        return OCRQuality.EXCELLENT, False, "Excellent text extraction quality - document is clear and readable"
    elif confidence >= 0.85:
        return OCRQuality.GOOD, False, "Good text extraction quality - document is readable"
    elif confidence >= OCR_CONFIDENCE_THRESHOLD:
        return OCRQuality.ACCEPTABLE, False, "Acceptable quality, but consider uploading a clearer document for best results"
    elif confidence >= 0.5:
        return OCRQuality.POOR, True, "Poor quality detected. Please upload a clearer, higher-resolution document"
    else:
        return OCRQuality.FAILED, True, "Text extraction failed. Document is not clear enough - please upload a better quality document"

def generate_clarity_suggestions(confidence: float, word_count: int, page_count: int) -> List[str]:
    suggestions = []
    
    if confidence < OCR_CONFIDENCE_THRESHOLD:
        suggestions.append("Ensure good lighting when scanning or photographing the document")
        suggestions.append("Use a higher resolution setting (at least 300 DPI for scans)")
        suggestions.append("Make sure the document is flat and not wrinkled or folded")
        suggestions.append("Focus the camera properly if taking a photo")
        suggestions.append("If possible, use a proper scanner instead of a phone camera")
    
    if word_count < 50:
        suggestions.append("Very little text was extracted - the document may be too blurry or low quality")
    
    if confidence < 0.5:
        suggestions.append("Critical: Document quality is too poor for reliable analysis")
        suggestions.append("Try uploading a digital PDF if available instead of a scanned copy")
    
    return suggestions

def extract_text_from_pdf(content: bytes, use_ocr: bool = True) -> OCRResult:
    """Extract text from PDF with OCR quality assessment"""
    text = ""
    total_confidence = 0.0
    page_count = 0
    
    try:
        # Try PyMuPDF first
        pdf_document = fitz.open(stream=content, filetype="pdf")
        page_count = len(pdf_document)
        
        for page_num in range(page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            text += page_text + "\n"
        
        pdf_document.close()
        
        # If text is found, assess confidence
        if text.strip():
            total_confidence = 0.95
        else:
            total_confidence = 0.0
    except Exception as e:
        logger.error(f"PyMuPDF extraction error: {e}")
    
    # If no text found and OCR is enabled, use Google Cloud Vision
    if not text.strip() and use_ocr and vision_client:
        try:
            logger.info("Attempting OCR with Google Cloud Vision")
            pdf_document = fitz.open(stream=content, filetype="pdf")
            page_count = len(pdf_document)
            
            for page_num in range(page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                
                image = vision.Image(content=img_bytes)
                response = vision_client.document_text_detection(image=image)
                
                if response.full_text_annotation:
                    text += response.full_text_annotation.text + "\n"
                    
                    # Calculate confidence
                    page_confidence = 0.0
                    if response.full_text_annotation.pages:
                        for page in response.full_text_annotation.pages:
                            page_confidence += page.confidence
                        page_confidence /= len(response.full_text_annotation.pages)
                    
                    total_confidence += page_confidence
            
            pdf_document.close()
            
            if page_count > 0:
                total_confidence /= page_count
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            total_confidence = 0.0
    
    # Fallback to PyPDF2
    if not text.strip():
        try:
            pdf_reader = PdfReader(io.BytesIO(content))
            page_count = len(pdf_reader.pages)
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            if text.strip():
                total_confidence = 0.9
        except Exception as e:
            logger.error(f"PyPDF2 extraction error: {e}")
    
    # Assess quality
    word_count = len(text.split())
    quality, needs_reupload, message = assess_ocr_quality(total_confidence)
    
    return OCRResult(
        text=text,
        confidence=total_confidence,
        quality=quality,
        needs_reupload=needs_reupload,
        message=message,
        page_count=page_count,
        word_count=word_count
    )

def analyze_contract_clause(text: str, contract_name: str, user_role: str) -> Dict[str, Any]:
    """Analyze contract using Gemini AI"""
    prompt = f"""
You are an expert legal analyst. Analyze this contract from the perspective of a {user_role}.

CONTRACT NAME: {contract_name}

Provide a detailed JSON analysis with:
1. "meaning": Clear explanation in simple language
2. "risk_level": low/medium/high
3. "risk_score": 1-10 numeric score
4. "red_flags": Array of concerning clauses
5. "favorable_terms": Array of beneficial clauses
6. "missing_clauses": Array of important missing protections
7. "recommendations": Array of specific actions to take

CONTRACT TEXT:
{text[:15000]}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        cleaned = clean_json_response(response.text)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {
            "meaning": "Unable to analyze contract",
            "risk_level": "unknown",
            "risk_score": 0,
            "red_flags": [],
            "favorable_terms": [],
            "missing_clauses": [],
            "recommendations": []
        }

def generate_questions(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate clarifying questions based on analysis"""
    questions = []
    
    if analysis.get("red_flags"):
        questions.append({
            "question": "What happens if I violate any of the red-flagged terms?",
            "importance": "high",
            "category": "risk"
        })
    
    if analysis.get("missing_clauses"):
        questions.append({
            "question": "Can we add clauses to cover the missing protections identified?",
            "importance": "medium",
            "category": "protection"
        })
    
    questions.append({
        "question": "What is the dispute resolution process if disagreements arise?",
        "importance": "high",
        "category": "dispute"
    })
    
    questions.append({
        "question": "Are there any penalties for early termination from either party?",
        "importance": "medium",
        "category": "termination"
    })
    
    return questions
def compare_contracts(analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two contract analyses"""
    
    # Safely convert risk_score to float (handles string, int, float, or missing)
    try:
        score1 = float(analysis1.get("risk_score") or 10)
    except (ValueError, TypeError):
        score1 = 10.0

    try:
        score2 = float(analysis2.get("risk_score") or 10)
    except (ValueError, TypeError):
        score2 = 10.0

    # Count red flags as a tiebreaker
    red_flags1 = len(analysis1.get("red_flags") or [])
    red_flags2 = len(analysis2.get("red_flags") or [])

    # Determine better contract
    if score1 < score2:
        better = 1
    elif score2 < score1:
        better = 2
    else:
        # Scores are equal — use red flag count as tiebreaker
        better = 1 if red_flags1 <= red_flags2 else 2

    # Build a meaningful reasoning string instead of a hardcoded one
    reasoning_parts = []
    if score1 != score2:
        reasoning_parts.append(
            f"Contract 1 has a risk score of {score1}/10 vs Contract 2's {score2}/10."
        )
    else:
        reasoning_parts.append(
            f"Both contracts have the same risk score of {score1}/10."
        )

    if red_flags1 != red_flags2:
        reasoning_parts.append(
            f"Contract 1 has {red_flags1} red flag(s) vs Contract 2's {red_flags2}."
        )

    reasoning = " ".join(reasoning_parts)

    final_advice = (
        f"Contract {better} is safer overall with a lower risk score "
        f"and fewer red flags. Review all recommendations before signing."
    )

    return {
        "better_contract": better,
        "risk_comparison": {
            "contract1": score1,
            "contract2": score2
        },
        "red_flags_comparison": {
            "contract1": red_flags1,
            "contract2": red_flags2
        },
        "reasoning": reasoning,
        "final_advice": final_advice
    }

def extract_specific_clauses(text: str, clause_types: List[str]) -> Dict[str, Any]:
    """Extract specific clause types from contract"""
    prompt = f"""
Extract the following clause types from this contract:
{', '.join(clause_types)}

Return JSON with:
- "clauses_found": array of {{type, text, location}}
- "missing_clauses": array of missing clause types

CONTRACT TEXT:
{text[:15000]}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        cleaned = clean_json_response(response.text)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Clause extraction error: {e}")
        return {"clauses_found": [], "missing_clauses": clause_types}

def generate_txt_report(analysis: Dict[str, Any], questions: List[Any], contract_name: str) -> str:
    """Generate plain text report"""
    report = f"CONTRACT ANALYSIS REPORT\n"
    report += f"=" * 50 + "\n\n"
    report += f"Contract: {contract_name}\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"Risk Level: {analysis.get('risk_level', 'unknown').upper()}\n"
    report += f"Risk Score: {analysis.get('risk_score', 0)}/10\n\n"
    report += f"SUMMARY\n"
    report += f"-" * 50 + "\n"
    report += f"{analysis.get('meaning', 'No analysis available')}\n\n"
    
    if analysis.get('red_flags'):
        report += f"RED FLAGS\n"
        report += f"-" * 50 + "\n"
        for i, flag in enumerate(analysis['red_flags'], 1):
            report += f"{i}. {flag}\n"
        report += "\n"
    
    if questions:
        report += f"QUESTIONS TO ASK\n"
        report += f"-" * 50 + "\n"
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                report += f"{i}. {q.get('question', '')}\n"
        report += "\n"
    
    return report

def generate_json_report(analysis: Dict[str, Any], questions: List[Any], contract_name: str) -> str:
    """Generate JSON report"""
    report = {
        "contract_name": contract_name,
        "generated_at": datetime.now().isoformat(),
        "analysis": analysis,
        "questions": questions
    }
    return json.dumps(report, indent=2)

def generate_markdown_report(analysis: Dict[str, Any], questions: List[Any], contract_name: str) -> str:
    """Generate Markdown report"""
    report = f"# Contract Analysis Report\n\n"
    report += f"**Contract:** {contract_name}  \n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n"
    report += f"## Risk Assessment\n\n"
    report += f"- **Risk Level:** {analysis.get('risk_level', 'unknown').upper()}\n"
    report += f"- **Risk Score:** {analysis.get('risk_score', 0)}/10\n\n"
    report += f"## Summary\n\n"
    report += f"{analysis.get('meaning', 'No analysis available')}\n\n"
    
    if analysis.get('red_flags'):
        report += f"## Red Flags\n\n"
        for flag in analysis['red_flags']:
            report += f"- {flag}\n"
        report += "\n"
    
    if questions:
        report += f"## Questions to Ask\n\n"
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                report += f"{i}. {q.get('question', '')}\n"
        report += "\n"
    
    return report

import html as html_module

def safe_para(text: str) -> str:
    """Escape special chars so ReportLab's XML parser doesn't crash"""
    return html_module.escape(str(text or ""))


def generate_pdf_report(analysis: Dict[str, Any], questions: List[Any], contract_name: str) -> bytes:
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
    )
    story.append(Paragraph("Contract Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Metadata
    story.append(Paragraph((f"<b>Contract:</b> {safe_para{contract_name}}", styles['Normal'])))
    story.append(Paragraph(f"<b>Generated:</b> {safe_para{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Risk assessment
    story.append(Paragraph("Risk Assessment", styles['Heading2']))
    story.append(Paragraph(f"<b>Risk Level:</b> {safe_para{analysis.get('risk_level', 'unknown').upper()}}", styles['Normal']))
    story.append(Paragraph(f"<b>Risk Score:</b> {safe_para{analysis.get('risk_score', 0)}}/10", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Summary
    story.append(Paragraph("Summary", styles['Heading2']))
    story.append(paragraph(safe_para(analysis.get('meaning', 'No analysis available'), styles['Normal'])))
    story.append(Spacer(1, 12))
    
    # Red flags
    if analysis.get('red_flags'):
        story.append(Paragraph("Red Flags", styles['Heading2']))
        for flag in analysis['red_flags']:
            story.append(Paragraph(f"•{safe_para(flag)}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Questions
    if questions:
        story.append(Paragraph("Questions to Ask", styles['Heading2']))
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                story.append(Paragraph(f"{i}.{safe_para (q.get('question', ''))}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
# ====================== SARVAM AI FUNCTIONS ======================
def translate_text(text: str, target_language: str) -> str:
    """Translate text using Sarvam AI"""
    if not SARVAM_API_KEY or target_language == "en-IN":
        return text

    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": text[:5000],
        "source_language_code": "en-IN",
        "target_language_code": target_language,
        "model": "mayura:v1",
        "enable_preprocessing": True
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get("translated_text", text)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

def text_to_speech(text: str, language: str = "en-IN", speaker: str = "meera") -> Optional[str]:
    """Convert text to speech using Sarvam AI"""
    if not SARVAM_API_KEY:
        logger.warning("Sarvam API key not configured")
        return None

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": [text[:3000]],
        "target_language_code": language,
        "speaker": speaker,
        "model": "bulbul:v1"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        audio_data = response.json().get("audios", [{}])[0].get("audioContent")
        if not audio_data:
            return "" 
        
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(text[:50].encode()).hexdigest()[:8]}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)
        
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(audio_data))
        
        logger.info(f"Audio generated: {filename}")
        return filepath
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# ====================== BACKGROUND TASKS ======================
def cleanup_file(filepath: str, delay_seconds: int = 0):
    """Background task to cleanup temporary files"""
    import time
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def cleanup_old_cache():
    """Clean up old cache entries"""
    global analysis_cache, audio_cache
    
    current_time = datetime.now()
    expired_keys = []
    
    for key, value in analysis_cache.items():
        if 'timestamp' in value:
            cache_time = datetime.fromisoformat(value['timestamp'])
            if (current_time - cache_time) > timedelta(hours=24):
                expired_keys.append(key)
    
    for key in expired_keys:
        del analysis_cache[key]
    
    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# ====================== API ROUTES ======================
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Legal Contract Intelligence API",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "ocr": "Google Cloud Vision + PyMuPDF",
            "ai": "Google Gemini",
            "translation": "Sarvam AI",
            "tts": "Sarvam AI"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analyze": "/api/analyze",
            "upload": "/api/upload",
            "compare": "/api/compare",
            "translate": "/api/translate",
            "tts": "/api/tts",
            "download_text": "/api/download/text/{analysis_id}",
            "download_audio": "/api/download/audio/{filename}"
        }
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "sarvam_configured": bool(SARVAM_API_KEY),
        "google_vision_configured": vision_client is not None,
        "ocr_confidence_threshold": OCR_CONFIDENCE_THRESHOLD,
        "temp_dir_exists": os.path.exists(TEMP_DIR),
        "upload_dir_exists": os.path.exists(UPLOAD_DIR),
        "cache_entries": len(analysis_cache),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_contract(
    request: AnalyzeRequest,
    language: str = Query("en-IN", pattern="^(en|hi|ta|te|kn|ml|mr|bn|gu|pa|od)-IN$"),
    generate_audio: bool = Query(False, description="Generate audio summary"),
    background_tasks: BackgroundTasks = None
):
    try:
        logger.info(f"Analyzing contract: {request.contract_name} (Role: {request.user_role})")
        
        # Generate analysis ID
        analysis_id = generate_analysis_id(request.text, request.contract_name)
        
        # Mask sensitive data
        cleaned_text = mask_sensitive_data(request.text)
        
        # Analyze with Gemini
        analysis = analyze_contract_clause(
            cleaned_text, 
            request.contract_name,
            request.user_role or "general"
        )
        
        # Generate questions
        questions = generate_questions(analysis)
        
        # Translate if needed
                                                # Continuation of the analyze_contract route and remaining routes

        # Translate if needed (continuation from Part 1)
        if language != "en-IN":
            for key in ["meaning", "red_flags"]:
                if key in analysis and isinstance(analysis[key], str):
                    analysis[key] = translate_text(analysis[key], language)
                elif key in analysis and isinstance(analysis[key], list):
                    analysis[key] = [translate_text(item, language) for item in analysis[key]]
            
            for q in questions:
                if isinstance(q, dict):
                    q["question"] = translate_text(q["question"], language)
        
        # Generate audio if requested
        audio_file = None
        audio_url = ""
        if generate_audio:
            audio_text = f"{analysis.get('meaning', '')}. Risk level is {analysis.get('risk_level', 'unknown')}."
            audio_file = text_to_speech(audio_text, language)
            if audio_file:
                audio_url = f"/api/download/audio/{os.path.basename(audio_file)}"
                audio_cache[analysis_id] = audio_file
        
        # Store in cache for downloads
        analysis_cache[analysis_id] = {
            "analysis": analysis,
            "questions": questions,
            "contract_name": request.contract_name,
            "timestamp": datetime.now().isoformat(),
            "language": language
        }
        
        return {
            "success": True,
            "analysis": analysis,
            "questions": questions,
            "metadata": {
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "language": language,
                "audio_generated": audio_file is not None,
                "audio_url": audio_url,
                "user_role": request.user_role
            },
            "download_options": {
                "txt": f"/api/download/text/{analysis_id}?format=txt",
                "json": f"/api/download/text/{analysis_id}?format=json",
                "pdf": f"/api/download/text/{analysis_id}?format=pdf",
                "md": f"/api/download/text/{analysis_id}?format=md",
                "audio": audio_url
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/api/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    contract_name: Optional[str] = Query(None),
    user_role: Optional[str] = Query("general"),
    language: str = Query("en-IN"),
    generate_audio: bool = Query(False),
    force_reupload: bool = Query(False, description="Force analysis even with low OCR quality"),
    background_tasks: BackgroundTasks = None
):
    """Upload PDF and analyze contract with OCR quality assessment"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {MAX_FILE_SIZE / 1024 / 1024}MB limit"
            )
        
        # Extract text with OCR quality assessment
        ocr_result = extract_text_from_pdf(content, use_ocr=True)
        
        # Check if reupload is needed
        if ocr_result.needs_reupload and not force_reupload:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "needs_reupload": True,
                    "ocr_quality": ocr_result.quality.value,
                    "confidence": ocr_result.confidence,
                    "message": ocr_result.message,
                    "suggestions": [
                        "Upload a higher resolution scan (300 DPI recommended)",
                        "Ensure the document is well-lit and not blurry",
                        "Avoid uploading photos of documents - use a scanner",
                        "Make sure text is not cut off at the edges",
                        "If the document has handwritten portions, they may not be recognized"
                    ],
                    "force_url": f"/api/upload?force_reupload=true",
                    "word_count": ocr_result.word_count,
                    "page_count": ocr_result.page_count
                }
            )
        
        # Check if text is sufficient
        if len(ocr_result.text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="PDF contains insufficient readable text. Please upload a clearer document."
            )
        
        # Use filename as contract name if not provided
        if not contract_name:
            contract_name = file.filename.replace('.pdf', '')
        
        # Analyze
        request = AnalyzeRequest(
            text=ocr_result.text,
            contract_name=contract_name,
            user_role=user_role
        )
        
        result = await analyze_contract(request, language, generate_audio, background_tasks)
        
        # Add OCR metadata to response
        if isinstance(result, dict):
            result["ocr_metadata"] = {
                "quality": ocr_result.quality.value,
                "confidence": ocr_result.confidence,
                "page_count": ocr_result.page_count,
                "word_count": ocr_result.word_count,
                "message": ocr_result.message
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/api/ocr-check")
async def check_ocr_quality(file: UploadFile = File(...)):
    """Check OCR quality of a PDF without full analysis"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {MAX_FILE_SIZE / 1024 / 1024}MB limit"
            )
        
        ocr_result = extract_text_from_pdf(content, use_ocr=True)
        
        return {
            "success": True,
            "filename": file.filename,
            "quality": ocr_result.quality.value,
            "confidence": ocr_result.confidence,
            "confidence_percentage": f"{ocr_result.confidence * 100:.1f}%",
            "needs_reupload": ocr_result.needs_reupload,
            "message": ocr_result.message,
            "page_count": ocr_result.page_count,
            "word_count": ocr_result.word_count,
            "threshold": OCR_CONFIDENCE_THRESHOLD,
            "sample_text": ocr_result.text[:500] + "..." if len(ocr_result.text) > 500 else ocr_result.text,
            "recommendations": [
                "Upload a higher resolution scan (300 DPI recommended)" if ocr_result.confidence < 0.8 else None,
                "Ensure good lighting and focus" if ocr_result.confidence < 0.7 else None,
                "Use a scanner instead of camera" if ocr_result.confidence < 0.6 else None,
                "Check if document has selectable text" if ocr_result.confidence < 0.5 else None
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR check error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR check failed: {str(e)}")

@app.get("/api/download/text/{analysis_id}")
async def download_text_report(
    analysis_id: str,
    format: DownloadFormat = Query(DownloadFormat.PDF),
    background_tasks: BackgroundTasks = None
):
    """Download analysis report in various formats"""
    if analysis_id not in analysis_cache:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Please run analysis first."
        )
    
    cached = analysis_cache[analysis_id]
    analysis = cached["analysis"]
    questions = cached["questions"]
    contract_name = cached["contract_name"]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = re.sub(r'[^\w\-_]', '_', contract_name)[:30]
    
    if format == DownloadFormat.TXT:
        content = generate_txt_report(analysis, questions, contract_name)
        filename = f"{safe_name}_analysis_{timestamp}.txt"
        media_type = "text/plain"
        
        return Response(
            content=content.encode('utf-8'),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    elif format == DownloadFormat.JSON:
        content = generate_json_report(analysis, questions, contract_name)
        filename = f"{safe_name}_analysis_{timestamp}.json"
        media_type = "application/json"
        
        return Response(
            content=content.encode('utf-8'),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    elif format == DownloadFormat.MD:
        content = generate_markdown_report(analysis, questions, contract_name)
        filename = f"{safe_name}_analysis_{timestamp}.md"
        media_type = "text/markdown"
        
        return Response(
            content=content.encode('utf-8'),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    elif format == DownloadFormat.PDF:
        content = generate_pdf_report(analysis, questions, contract_name)
        filename = f"{safe_name}_analysis_{timestamp}.pdf"
        
        return Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

@app.get("/api/download/audio/{filename}")
async def download_audio(
    filename: str,
    background_tasks: BackgroundTasks = None
):
    """Download generated audio file"""
    # Sanitize filename
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(TEMP_DIR, safe_filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail="Audio file not found or expired"
        )
    
    return FileResponse(
        filepath,
        media_type="audio/mpeg",
        filename=safe_filename,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"'
        }
    )

@app.post("/api/generate-audio/{analysis_id}")
async def generate_audio_for_analysis(
    analysis_id: str,
    language: str = Query("en-IN"),
    speaker: str = Query("meera"),
    include_questions: bool = Query(True),
    background_tasks: BackgroundTasks = None
):
    """Generate audio summary for an existing analysis"""
    if analysis_id not in analysis_cache:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    
    cached = analysis_cache[analysis_id]
    analysis = cached["analysis"]
    questions = cached["questions"]
    
    # Build audio text
    audio_text = f"Contract Analysis Summary. "
    audio_text += f"Risk Level: {analysis.get('risk_level', 'unknown')}. "
    audio_text += f"Risk Score: {analysis.get('risk_score', 'unknown')} out of 10. "
    audio_text += f"{analysis.get('meaning', '')} "
    
    # Add red flags
    red_flags = analysis.get('red_flags', [])
    if red_flags:
        audio_text += f"Warning! There are {len(red_flags)} red flags. "
        for flag in red_flags[:3]:
            audio_text += f"{flag}. "
    
    # Add questions if requested
    if include_questions and questions:
        audio_text += "Questions to ask before signing: "
        for i, q in enumerate(questions[:5], 1):
            if isinstance(q, dict):
                audio_text += f"Question {i}: {q.get('question', '')} "
    
    # Generate audio
    audio_file = text_to_speech(audio_text[:3000], language, speaker)
    
    if not audio_file:
        raise HTTPException(
            status_code=500,
            detail="Audio generation failed"
        )
    
    # Store in cache
    audio_cache[analysis_id] = audio_file
    
    return {
        "success": True,
        "audio_url": f"/api/download/audio/{os.path.basename(audio_file)}",
        "language": language,
        "speaker": speaker,
        "duration_estimate": f"{len(audio_text.split()) // 150} minutes",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/compare")
async def compare_documents(
    request: CompareRequest,
    language: str = Query("en-IN"),
    background_tasks: BackgroundTasks = None
):
    """Compare two contracts"""
    try:
        logger.info(f"Comparing: {request.name1} vs {request.name2}")
        
        # Analyze both contracts
        analysis1 = analyze_contract_clause(
            mask_sensitive_data(request.text1),
            request.name1,
            request.user_role or "general"
        )
        analysis2 = analyze_contract_clause(
            mask_sensitive_data(request.text2),
            request.name2,
            request.user_role or "general"
        )
        
        # Compare
        comparison = compare_contracts(analysis1, analysis2)
        
        # Generate comparison ID for downloads
        comparison_id = generate_analysis_id(request.text1 + request.text2, "comparison")
        
        # Store in cache
        analysis_cache[comparison_id] = {
            "analysis": {
                "comparison": comparison,
                "contract1": analysis1,
                "contract2": analysis2
            },
            "questions": [],
            "contract_name": f"{request.name1}_vs_{request.name2}",
            "timestamp": datetime.now().isoformat(),
            "language": language
        }
        
        # Translate if needed
        if language != "en-IN":
            for key in ["reasoning", "final_advice"]:
                if key in comparison:
                    comparison[key] = translate_text(comparison[key], language)
        
        return {
            "success": True,
            "comparison": comparison,
            "contracts": {
                "contract1": analysis1,
                "contract2": analysis2
            },
            "metadata": {
                "comparison_id": comparison_id,
                "timestamp": datetime.now().isoformat(),
                "language": language
            },
            "download_options": {
                "pdf": f"/api/download/text/{comparison_id}?format=pdf",
                "json": f"/api/download/text/{comparison_id}?format=json",
                "txt": f"/api/download/text/{comparison_id}?format=txt"
            }
        }
        
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

@app.post("/api/extract-clauses")
async def extract_clauses(request: ClauseExtractionRequest):
    """Extract specific clause types from contract"""
    try:
        cleaned_text = mask_sensitive_data(request.text)
        result = extract_specific_clauses(cleaned_text, request.clause_types)
        
        return {
            "success": True,
            "result": result,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "clause_types_requested": request.clause_types
            }
        }
    except Exception as e:
        logger.error(f"Clause extraction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Clause extraction failed: {str(e)}"
        )

@app.post("/api/translate")
async def translate(request: TranslateRequest):
    """Translate text to regional Indian language"""
    try:
        translated = translate_text(request.text, request.target_language)
        
        return {
            "success": True,
            "original_text": request.text,
            "translated_text": translated,
            "target_language": request.target_language,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.post("/api/tts")
async def generate_tts(
    request: TTSRequest,
    background_tasks: BackgroundTasks = None
):
    """Generate audio from text using TTS"""
    try:
        audio_file = text_to_speech(request.text, request.language, request.speaker)
        
        if not audio_file:
            raise HTTPException(
                status_code=500,
                detail="Audio generation failed. Check if Sarvam API is configured."
            )
        
        return {
            "success": True,
            "audio_url": f"/api/download/audio/{os.path.basename(audio_file)}",
            "language": request.language,
            "speaker": request.speaker,
            "text_length": len(request.text),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}"
        )

# Remaining routes and startup/shutdown continue...
# (Due to length limits, remaining utility routes are similar pattern)
