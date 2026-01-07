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
from typing import Optional, Dict, Any
from google import genai
from PyPDF2 import PdfReader
from google.cloud import vision
from google.oauth2 import service_account
import io
import fitz  # PyMuPDF for better PDF handling
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
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
TEMP_DIR = "temp"
UPLOAD_DIR = "uploads"
DOWNLOADS_DIR = "downloads"
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75"))  # 75% default

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY missing in environment variables")

# Configure Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "models/gemini-flash-latest"

# Configure Google Cloud Vision (optional)
vision_client = None
try:
    if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS
        )
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        logger.info("✅ Google Cloud Vision configured from file")
    elif GOOGLE_CLOUD_CREDENTIALS:
        # Parse JSON credentials from environment variable
        creds_dict = json.loads(GOOGLE_CLOUD_CREDENTIALS)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        logger.info("✅ Google Cloud Vision configured from environment")
except Exception as e:
    logger.warning(f"⚠️ Google Cloud Vision not configured: {e}")
    vision_client = None

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

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://signature-gap-frontend-bhtzcmog5-nehas-projects-f3c149cb.vercel.app/",
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
    """Response model for document clarity check"""
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
# For demo purposes - use Redis/Database in production
analysis_cache: Dict[str, Dict[str, Any]] = {}
audio_cache: Dict[str, str] = {}

# ====================== UTILITY FUNCTIONS ======================
def generate_analysis_id(text: str, contract_name: str) -> str:
    """Generate unique analysis ID"""
    content = f"{text[:100]}{contract_name}{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

def mask_sensitive_data(text: str) -> str:
    """Mask PII data using regex patterns"""
    patterns = [
        (r"\b\d{10}\b", "[PHONE]"),
        (r"\+91[\s-]?\d{10}", "[PHONE]"),
        (r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN]"),
        (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[AADHAAR]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b", "[EMAIL]"),
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]"),
        (r"\b[A-Z]{2}\d{2}[A-Z0-9]{13,16}\b", "[IBAN]"),
        (r"\b[A-Z]{6}\d{2}[A-Z0-9]{2}[A-Z0-9]{3}\b", "[IFSC]")
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    return text

def clean_json_response(text: str) -> str:
    """Clean JSON response from Gemini"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def assess_ocr_quality(confidence: float) -> Tuple[OCRQuality, bool, str]:
    """Assess OCR quality and determine if reupload is needed"""
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
    """Generate specific suggestions for improving document clarity"""
    suggestions = []
    
    if confidence < OCR_CONFIDENCE_THRESHOLD:
        suggestions.append("📸 Ensure good lighting when scanning or photographing the document")
        suggestions.append("🔍 Use a higher resolution setting (at least 300 DPI for scans)")
        suggestions.append("📄 Make sure the document is flat and not wrinkled or folded")
        suggestions.append("🎯 Focus the camera properly if taking a photo")
        suggestions.append("🖨️ If possible, use a proper scanner instead of a phone camera")
    
    if word_count < 50:
        suggestions.append("⚠️ Very little text was extracted - the document may be too blurry or low quality")
    
    if confidence < 0.5:
        suggestions.append("🔴 Critical: Document quality is too poor for reliable analysis")
        suggestions.append("💡 Try uploading a digital PDF if available instead of a scanned copy")
    
    return suggestions

# ====================== OCR FUNCTIONS ======================
def extract_text_with_pypdf(file_content: bytes) -> Tuple[str, float]:
    """Extract text using PyPDF2"""
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        # Estimate confidence based on text quality
        word_count = len(text.split())
        char_count = len(text.replace(" ", "").replace("\n", ""))
        
        # Simple heuristic for confidence
        if word_count > 100 and char_count > 500:
            confidence = 0.85
        elif word_count > 50:
            confidence = 0.70
        else:
            confidence = 0.50
        
        return text, confidence
    except Exception as e:
        logger.error(f"PyPDF2 extraction error: {e}")
        return "", 0.0

def extract_text_with_pymupdf(file_content: bytes) -> Tuple[str, float]:
    """Extract text using PyMuPDF (better quality)"""
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        total_blocks = 0
        readable_blocks = 0
        
        for page in pdf_document:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                total_blocks += 1
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span_text = span["text"].strip()
                            if span_text:
                                text += span_text + " "
                                readable_blocks += 1
            text += "\n"
        
        pdf_document.close()
        
        # Calculate confidence based on readable blocks ratio
        confidence = readable_blocks / max(total_blocks, 1) if total_blocks > 0 else 0.5
        confidence = min(confidence * 1.2, 1.0)  # Boost slightly
        
        return text.strip(), confidence
    except Exception as e:
        logger.error(f"PyMuPDF extraction error: {e}")
        return "", 0.0

def extract_text_with_google_ocr(file_content: bytes) -> Tuple[str, float]:
    """Extract text using Google Cloud Vision OCR"""
    if not vision_client:
        logger.warning("Google Cloud Vision not configured, falling back to PyMuPDF")
        return extract_text_with_pymupdf(file_content)
    
    try:
        # Convert PDF pages to images and OCR each
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        full_text = ""
        total_confidence = 0.0
        page_count = 0
        
        for page_num, page in enumerate(pdf_document):
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_bytes = pix.tobytes("png")
            
            # Send to Google Vision
            image = vision.Image(content=img_bytes)
            response = vision_client.document_text_detection(image=image)
            
            if response.error.message:
                logger.error(f"Google Vision error: {response.error.message}")
                continue
            
            if response.full_text_annotation:
                full_text += response.full_text_annotation.text + "\n"
                
                # Calculate confidence for this page
                page_confidence = 0.0
                block_count = 0
                for page_anno in response.full_text_annotation.pages:
                    for block in page_anno.blocks:
                        page_confidence += block.confidence
                        block_count += 1
                
                if block_count > 0:
                    total_confidence += page_confidence / block_count
                    page_count += 1
        
        pdf_document.close()
        
        # Average confidence across pages
        avg_confidence = total_confidence / max(page_count, 1)
        
        return full_text.strip(), avg_confidence
        
    except Exception as e:
        logger.error(f"Google OCR error: {e}")
        # Fallback to PyMuPDF
        return extract_text_with_pymupdf(file_content)

def extract_text_from_pdf(file_content: bytes, use_ocr: bool = True) -> OCRResult:
    """
    Extract text from PDF with quality assessment
    """
    try:
        # Try PyMuPDF first (fastest)
        text, confidence = extract_text_with_pymupdf(file_content)
        
        # If confidence is low and OCR is enabled, try Google OCR
        if confidence < OCR_CONFIDENCE_THRESHOLD and use_ocr and vision_client:
            logger.info("Low confidence from PyMuPDF, trying Google Cloud OCR...")
            ocr_text, ocr_confidence = extract_text_with_google_ocr(file_content)
            
            if ocr_confidence > confidence:
                text = ocr_text
                confidence = ocr_confidence
        
        # If still low, try PyPDF2 as last resort
        if confidence < 0.5:
            pypdf_text, pypdf_confidence = extract_text_with_pypdf(file_content)
            if pypdf_confidence > confidence:
                text = pypdf_text
                confidence = pypdf_confidence
        
        # Assess quality
        quality, needs_reupload, message = assess_ocr_quality(confidence)
        
        # Count pages
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            page_count = len(pdf_document)
            pdf_document.close()
        except:
            page_count = 1
        
        word_count = len(text.split())
        
        return OCRResult(
            text=text,
            confidence=round(confidence, 3),
            quality=quality,
            needs_reupload=needs_reupload,
            message=message,
            page_count=page_count,
            word_count=word_count
        )
        
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return OCRResult(
            text="",
            confidence=0.0,
            quality=OCRQuality.FAILED,
            needs_reupload=True,
            message=f"Failed to extract text: {str(e)}",
            page_count=0,
            word_count=0
        )

# ====================== NEW ENDPOINT: DOCUMENT CLARITY CHECK ======================
@app.post("/api/v1/check-document-clarity", response_model=DocumentClarityCheckResponse)
async def check_document_clarity(
    file: UploadFile = File(..., description="PDF or image file to check for clarity")
):
    """
    Check if uploaded document is clear enough for processing.
    This endpoint should be called BEFORE attempting full analysis.
    Returns detailed clarity assessment and suggestions if document needs to be re-uploaded.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith(('application/pdf', 'image/')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files and images are supported"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        logger.info(f"Checking clarity for file: {file.filename}")
        
        # Extract text and assess quality
        ocr_result = extract_text_from_pdf(file_content, use_ocr=True)
        
        # Determine if document is clear enough
        is_clear = not ocr_result.needs_reupload
        
        # Generate suggestions
        suggestions = generate_clarity_suggestions(
            ocr_result.confidence,
            ocr_result.word_count,
            ocr_result.page_count
        )
        
        # Build response
        response = DocumentClarityCheckResponse(
            success=True,
            is_clear=is_clear,
            confidence=ocr_result.confidence,
            quality=ocr_result.quality,
            needs_reupload=ocr_result.needs_reupload,
            message=ocr_result.message,
            details={
                "page_count": ocr_result.page_count,
                "word_count": ocr_result.word_count,
                "extracted_text_preview": ocr_result.text[:200] + "..." if len(ocr_result.text) > 200 else ocr_result.text,
                "quality_threshold": OCR_CONFIDENCE_THRESHOLD,
                "has_google_ocr": vision_client is not None
            },
            suggestions=suggestions if ocr_result.needs_reupload else [
                "✅ Document quality is good - you can proceed with analysis"
            ]
        )
        
        logger.info(f"Clarity check complete: is_clear={is_clear}, confidence={ocr_result.confidence}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document clarity check error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking document clarity: {str(e)}"
        )

# ====================== MODIFIED UPLOAD ENDPOINT ======================
@app.post("/api/v1/upload-and-extract")
async def upload_and_extract(
    file: UploadFile = File(...),
    force_process: bool = Query(False, description="Force processing even if quality is poor")
):
    """
    Upload PDF/image and extract text with quality check.
    If document quality is poor and force_process=False, returns error requesting re-upload.
    """
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith(('application/pdf', 'image/')):
            raise HTTPException(400, "Only PDF files and images are supported")
        
        file_content = await file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File too large. Max: {MAX_FILE_SIZE / 1024 / 1024}MB")
        
        logger.info(f"Processing upload: {file.filename}")
        
        # Extract text with quality assessment
        ocr_result = extract_text_from_pdf(file_content, use_ocr=True)
        
        # Check if reupload is needed
        if ocr_result.needs_reupload and not force_process:
            suggestions = generate_clarity_suggestions(
                ocr_result.confidence,
                ocr_result.word_count,
                ocr_result.page_count
            )
            
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Document quality too poor for reliable analysis",
                    "needs_reupload": True,
                    "quality_details": {
                        "confidence": ocr_result.confidence,
                        "quality": ocr_result.quality,
                        "message": ocr_result.message,
                        "word_count": ocr_result.word_count,
                        "page_count": ocr_result.page_count
                    },
                    "suggestions": suggestions,
                    "hint": "Use /api/v1/check-document-clarity endpoint first to verify document quality"
                }
            )
        
        # Document is clear enough or forced - proceed
        return {
            "success": True,
            "text": ocr_result.text,
            "metadata": {
                "filename": file.filename,
                "page_count": ocr_result.page_count,
                "word_count": ocr_result.word_count,
                "extraction_quality": {
                    "confidence": ocr_result.confidence,
                    "quality": ocr_result.quality,
                    "message": ocr_result.message,
                    "needs_improvement": ocr_result.quality in [OCRQuality.ACCEPTABLE, OCRQuality.POOR]
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

# ====================== DOWNLOAD GENERATION FUNCTIONS ======================
def generate_txt_report(analysis: dict, questions: list, contract_name: str) -> str:
    """Generate plain text report"""
    report = f"""
================================================================================
LEGAL CONTRACT ANALYSIS REPORT
================================================================================

Contract Name: {contract_name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Contract Type: {analysis.get('contract_type', 'Unknown')}

--------------------------------------------------------------------------------
RISK ASSESSMENT
--------------------------------------------------------------------------------
Risk Level: {analysis.get('risk_level', 'Unknown')}
Risk Score: {analysis.get('risk_score', 'N/A')}/10

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
{analysis.get('meaning', 'No summary available')}

--------------------------------------------------------------------------------
KEY TERMS
--------------------------------------------------------------------------------

FAVORABLE TERMS:
{chr(10).join(['• ' + term for term in analysis.get('key_terms', {}).get('favorable', [])])}

UNFAVORABLE TERMS:
{chr(10).join(['• ' + term for term in analysis.get('key_terms', {}).get('unfavorable', [])])}

AMBIGUOUS TERMS:
{chr(10).join(['• ' + term for term in analysis.get('key_terms', {}).get('ambiguous', [])])}

--------------------------------------------------------------------------------
CONCERNS
--------------------------------------------------------------------------------
"""
    
    for concern in analysis.get('concerns', []):
        if isinstance(concern, dict):
            report += f"\n[{concern.get('severity', 'MEDIUM')}] {concern.get('concern', '')}"
            report += f"\n   Impact: {concern.get('impact', 'Unknown')}\n"
    
    report += """
--------------------------------------------------------------------------------
RED FLAGS
--------------------------------------------------------------------------------
"""
    for flag in analysis.get('red_flags', []):
        report += f"⚠️ {flag}\n"
    
    report += """
--------------------------------------------------------------------------------
MISSING PROTECTIONS
--------------------------------------------------------------------------------
"""
    for protection in analysis.get('missing_protections', []):
        report += f"• {protection}\n"
    
    report += """
--------------------------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------------------------
"""
    for rec in analysis.get('recommendations', []):
        if isinstance(rec, dict):
            report += f"\n[{rec.get('priority', 'MEDIUM')}] {rec.get('action', '')}"
            report += f"\n   Reason: {rec.get('reason', '')}\n"
    
    if questions:
        report += """
--------------------------------------------------------------------------------
QUESTIONS TO ASK BEFORE SIGNING
--------------------------------------------------------------------------------
"""
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                report += f"\n{i}. [{q.get('priority', 'MEDIUM')}] {q.get('question', '')}"
                report += f"\n   Category: {q.get('category', 'general')}\n"
            else:
                report += f"\n{i}. {q}\n"
    
    report += """
================================================================================
DISCLAIMER: This analysis is AI-generated and should not be considered legal 
advice. Please consult a qualified legal professional for important decisions.
================================================================================
"""
    
    return report

def generate_json_report(analysis: dict, questions: list, contract_name: str) -> str:
    """Generate JSON report"""
    report = {
        "report_metadata": {
            "contract_name": contract_name,
            "generated_at": datetime.now().isoformat(),
            "version": "3.0.0"
        },
        "analysis": analysis,
        "questions": questions,
        "disclaimer": "This analysis is AI-generated and should not be considered legal advice."
    }
    return json.dumps(report, indent=2, ensure_ascii=False)

def generate_markdown_report(analysis: dict, questions: list, contract_name: str) -> str:
    """Generate Markdown report"""
    report = f"""# Legal Contract Analysis Report

**Contract Name:** {contract_name}  
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Contract Type:** {analysis.get('contract_type', 'Unknown')}

---

## 🎯 Risk Assessment

| Metric | Value |
|--------|-------|
| Risk Level | **{analysis.get('risk_level', 'Unknown')}** |
| Risk Score | **{analysis.get('risk_score', 'N/A')}/10** |

---

## 📋 Summary

{analysis.get('meaning', 'No summary available')}

---

## 📊 Key Terms Analysis

### ✅ Favorable Terms
"""
    
    for term in analysis.get('key_terms', {}).get('favorable', []):
        report += f"- {term}\n"
    
    report += "\n### ❌ Unfavorable Terms\n"
    for term in analysis.get('key_terms', {}).get('unfavorable', []):
        report += f"- {term}\n"
    
    report += "\n### ❓ Ambiguous Terms\n"
    for term in analysis.get('key_terms', {}).get('ambiguous', []):
        report += f"- {term}\n"
    
    report += "\n---\n\n## ⚠️ Concerns\n\n"
    for concern in analysis.get('concerns', []):
        if isinstance(concern, dict):
            severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(concern.get('severity', ''), '⚪')
            report += f"### {severity_emoji} {concern.get('concern', '')}\n"
            report += f"**Impact:** {concern.get('impact', 'Unknown')}\n\n"
    
    report += "---\n\n## 🚩 Red Flags\n\n"
    for flag in analysis.get('red_flags', []):
        report += f"- ⚠️ {flag}\n"
    
    report += "\n---\n\n## 🛡️ Missing Protections\n\n"
    for protection in analysis.get('missing_protections', []):
        report += f"- {protection}\n"
    
    report += "\n---\n\n## 💡 Recommendations\n\n"
    # Concerns
    story.append(Paragraph("Concerns", heading_style))
    for concern in analysis.get('concerns', []):
        if isinstance(concern, dict):
            severity = concern.get('severity', 'MEDIUM')
            story.append(Paragraph(f"<b>[{severity}]</b> {concern.get('concern', '')}", normal_style))
            story.append(Paragraph(f"<i>Impact: {concern.get('impact', 'Unknown')}</i>", normal_style))
    
    # Red Flags
    story.append(Paragraph("Red Flags", heading_style))
    for flag in analysis.get('red_flags', []):
        story.append(Paragraph(f"⚠ {flag}", normal_style))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    for rec in analysis.get('recommendations', []):
        if isinstance(rec, dict):
            priority = rec.get('priority', 'MEDIUM')
            story.append(Paragraph(f"<b>[{priority}]</b> {rec.get('action', '')}", normal_style))
    
    # Questions
    if questions:
        story.append(Paragraph("Questions to Ask Before Signing", heading_style))
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                story.append(Paragraph(f"{i}. {q.get('question', '')}", normal_style))
            else:
                story.append(Paragraph(f"{i}. {q}", normal_style))
    
    # Disclaimer
    story.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray
    )
    story.append(Paragraph(
        "<i>Disclaimer: This analysis is AI-generated and should not be considered legal advice. "
        "Please consult a qualified legal professional for important decisions.</i>",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ====================== GEMINI AI FUNCTIONS ======================
def analyze_contract_clause(text: str, contract_name: str = "Contract", user_role: str = "general") -> dict:
    """Analyze contract using Gemini AI with role-specific insights"""
    role_context = {
        "employee": "Focus on employee rights, benefits, and protections",
        "employer": "Focus on employer protections and obligations",
        "freelancer": "Focus on payment terms, deliverables, and IP rights",
        "tenant": "Focus on tenant rights, deposits, and maintenance",
        "landlord": "Focus on property protection and payment terms",
        "general": "Provide balanced analysis for all parties"
    }
    
    context = role_context.get(user_role, role_context["general"])
    
    prompt = f"""You are a legal contract expert. Analyze this contract from the perspective of a {user_role}.

{context}

Return ONLY valid JSON in this format:

{{
  "contract_name": "{contract_name}",
  "contract_type": "employment | rental | freelance | service | nda | other",
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "risk_score": 5,
  "meaning": "Clear explanation of what this clause means in simple terms",
  "who_affected": ["employees", "freshers", "contractors"],
  "key_terms": {{
    "favorable": ["positive aspect 1", "positive aspect 2"],
    "unfavorable": ["negative aspect 1", "negative aspect 2"],
    "ambiguous": ["unclear term 1"]
  }},
  "concerns": [
    {{"concern": "Specific issue", "severity": "HIGH|MEDIUM|LOW", "impact": "What could happen"}}
  ],
  "missing_protections": ["Protection 1", "Protection 2"],
  "legal_compliance": {{
    "labor_laws": "compliant | non-compliant | unclear",
    "notes": "Any compliance issues"
  }},
  "recommendations": [
    {{"action": "What to do", "priority": "HIGH|MEDIUM|LOW", "reason": "Why important"}}
  ],
  "red_flags": ["Critical issue 1", "Critical issue 2"]
}}

Risk Score Scale:
1-3: LOW RISK - Fair terms, standard clauses
4-6: MEDIUM RISK - Some unfavorable terms, negotiable
7-8: HIGH RISK - Significantly unfavorable, requires changes
9-10: CRITICAL RISK - Dangerous terms, do not sign

CONTRACT TEXT:
{text[:15000]}
"""

    try:
        response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt)
        text = response.text
    

        cleaned = clean_json_response(response.text)
        analysis = json.loads(cleaned)
        
        # Add metadata
        analysis["analyzed_at"] = datetime.now().isoformat()
        analysis["user_role"] = user_role
        
        logger.info(f"Analysis complete for {contract_name} (Role: {user_role})")
        return analysis
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse AI response"
        )
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}"
        )

def generate_questions(analysis: dict) -> list:
    """Generate 'Ask Before Signing' questions"""
    prompt = f"""Generate 5-7 specific, actionable questions someone should ask BEFORE signing this contract.

Return ONLY JSON:
{{
  "questions": [
    {{"question": "Could you clarify...?", "category": "payment|termination|liability|general", "priority": "HIGH|MEDIUM|LOW"}}
  ]
}}

ANALYSIS:
{json.dumps(analysis, indent=2)}
"""

    try:
        response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt)
        text = response.text


        cleaned = clean_json_response(response.text)
        result = json.loads(cleaned)
        return result.get("questions", [])
    except Exception as e:
        logger.error(f"Question generation error: {e}")
        return [
            {"question": "Could you clarify the terms mentioned in this clause?", "category": "general", "priority": "MEDIUM"},
            {"question": "What are the consequences if I don't comply with this clause?", "category": "liability", "priority": "HIGH"}
        ]

def compare_contracts(analysis1: dict, analysis2: dict) -> dict:
    """Compare two contract analyses"""
    prompt = f"""Compare these two contracts and provide detailed analysis.

Return ONLY JSON:

{{
  "recommended": "Contract 1 | Contract 2 | Neither",
  "confidence": "HIGH | MEDIUM | LOW",
  "reasoning": "Clear explanation",
  "risk_comparison": {{
    "contract1_risk": "summary with score",
    "contract2_risk": "summary with score",
    "winner": "Contract 1 | Contract 2"
  }},
  "key_differences": [
    {{"aspect": "Payment", "contract1": "desc", "contract2": "desc", "better_in": "Contract 1|Contract 2"}}
  ],
  "final_advice": "What the user should do"
}}

CONTRACT 1:
{json.dumps(analysis1, indent=2)}

CONTRACT 2:
{json.dumps(analysis2, indent=2)}
"""

    try:
        response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt)
        text = response.text

        cleaned = clean_json_response(response.text)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail="Contract comparison failed")

def extract_specific_clauses(text: str, clause_types: List[str]) -> dict:
    """Extract specific clause types from contract"""
    prompt = f"""Extract and analyze these specific clause types from the contract: {', '.join(clause_types)}

Return ONLY JSON:

{{
  "clauses_found": [
    {{
      "type": "termination|payment|confidentiality|etc",
      "text": "exact clause text",
      "analysis": "what this clause means",
      "risk_level": "LOW|MEDIUM|HIGH|CRITICAL"
    }}
  ],
  "missing_clauses": ["clause type 1"]
}}

CONTRACT TEXT:
{text[:15000]}
"""

    try:
        response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt)
        text = response.text

        cleaned = clean_json_response(response.text)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Clause extraction error: {e}")
        return {"clauses_found": [], "missing_clauses": clause_types}

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
    """
    Upload PDF and analyze contract with OCR quality assessment
    
    - **file**: PDF file to upload (max 10MB)
    - **contract_name**: Optional contract name
    - **user_role**: User's role for analysis
    - **language**: Target language
    - **generate_audio**: Generate audio summary
    - **force_reupload**: Force analysis even with low OCR quality
    """
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
    """
    Check OCR quality of a PDF without full analysis
    
    Returns quality assessment and recommendations
    """
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
    """
    Download analysis report in various formats
    
    - **analysis_id**: ID from analysis response
    - **format**: txt, json, pdf, or md
    """
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
    """
    Download generated audio file
    """
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
    """
    Generate audio summary for an existing analysis
    """
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
    """
    Compare two contracts
    """
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
    """
    Extract specific clause types from contract
    """
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
    """
    Translate text to regional Indian language
    """
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
    """
    Generate audio from text using TTS
    """
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

@app.get("/api/languages")
async def get_supported_languages():
    """
    Get list of supported languages
    """
    return {
        "success": True,
        "languages": [
            {"code": "en-IN", "name": "English", "tts_supported": True},
            {"code": "hi-IN", "name": "Hindi", "tts_supported": True},
            {"code": "ta-IN", "name": "Tamil", "tts_supported": True},
            {"code": "te-IN", "name": "Telugu", "tts_supported": True},
            {"code": "kn-IN", "name": "Kannada", "tts_supported": True},
            {"code": "ml-IN", "name": "Malayalam", "tts_supported": True},
            {"code": "mr-IN", "name": "Marathi", "tts_supported": True},
            {"code": "bn-IN", "name": "Bengali", "tts_supported": True},
            {"code": "gu-IN", "name": "Gujarati", "tts_supported": True},
            {"code": "pa-IN", "name": "Punjabi", "tts_supported": True},
            {"code": "od-IN", "name": "Odia", "tts_supported": True}
        ],
        "speakers": [
            {"id": "meera", "gender": "female", "description": "Default female voice"},
            {"id": "pavithra", "gender": "female", "description": "Alternative female voice"},
            {"id": "maitreyi", "gender": "female", "description": "Professional female voice"},
            {"id": "arvind", "gender": "male", "description": "Default male voice"},
            {"id": "karthik", "gender": "male", "description": "Alternative male voice"}
        ]
    }

@app.get("/api/contract-types")
async def get_contract_types():
    """
    Get list of supported contract types
    """
    return {
        "success": True,
        "contract_types": [
            {"type": "employment", "description": "Employment agreements and job contracts"},
            {"type": "rental", "description": "Rental and lease agreements"},
            {"type": "freelance", "description": "Freelance and consulting contracts"},
            {"type": "service", "description": "Service level agreements"},
            {"type": "nda", "description": "Non-disclosure agreements"},
            {"type": "partnership", "description": "Partnership agreements"},
            {"type": "sales", "description": "Sales and purchase agreements"},
            {"type": "other", "description": "Other contract types"}
        ]
    }

@app.get("/api/user-roles")
async def get_user_roles():
    """
    Get list of supported user roles for analysis
    """
    return {
        "success": True,
        "roles": [
            {"role": "employee", "description": "Focus on employee rights and protections"},
            {"role": "employer", "description": "Focus on employer protections and obligations"},
            {"role": "freelancer", "description": "Focus on payment terms, deliverables, IP rights"},
            {"role": "tenant", "description": "Focus on tenant rights, deposits, maintenance"},
            {"role": "landlord", "description": "Focus on property protection and payment terms"},
            {"role": "general", "description": "Balanced analysis for all parties"}
        ]
    }

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Retrieve a cached analysis by ID
    """
    if analysis_id not in analysis_cache:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found or expired"
        )
    
    cached = analysis_cache[analysis_id]
    
    return {
        "success": True,
        "analysis": cached["analysis"],
        "questions": cached["questions"],
        "contract_name": cached["contract_name"],
        "timestamp": cached["timestamp"],
        "download_options": {
            "txt": f"/api/download/text/{analysis_id}?format=txt",
            "json": f"/api/download/text/{analysis_id}?format=json",
            "pdf": f"/api/download/text/{analysis_id}?format=pdf",
            "md": f"/api/download/text/{analysis_id}?format=md"
        }
    }

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete a cached analysis
    """
    if analysis_id in analysis_cache:
        del analysis_cache[analysis_id]
    
    if analysis_id in audio_cache:
        audio_path = audio_cache[analysis_id]
        if os.path.exists(audio_path):
            os.remove(audio_path)
        del audio_cache[analysis_id]
    
    return {
        "success": True,
        "message": "Analysis deleted successfully"
    }

@app.delete("/api/cleanup")
async def manual_cleanup(background_tasks: BackgroundTasks = None):
    """
    Manually trigger cleanup of temporary files and old cache
    """
    try:
        cleaned_files = 0
        cleaned_cache = 0
        
        # Clean temp directory
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, file)
                if os.path.isfile(filepath):
                    # Check if file is older than 1 hour
                    file_age = datetime.now().timestamp() - os.path.getmtime(filepath)
                    if file_age > 3600:  # 1 hour
                        os.remove(filepath)
                        cleaned_files += 1
        
        # Clean old cache entries
        current_time = datetime.now()
        expired_keys = []
        
        for key, value in analysis_cache.items():
            if 'timestamp' in value:
                try:
                    cache_time = datetime.fromisoformat(value['timestamp'])
                    if (current_time - cache_time) > timedelta(hours=24):
                        expired_keys.append(key)
                except:
                    pass
        
        for key in expired_keys:
            del analysis_cache[key]
            cleaned_cache += 1
        
        return {
            "success": True,
            "message": f"Cleaned {cleaned_files} files and {cleaned_cache} cache entries",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )

# ====================== ERROR HANDLERS ======================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None,
            "timestamp": datetime.now().isoformat()
        }
    )

# ====================== STARTUP & SHUTDOWN EVENTS ======================
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    
    logger.info("🚀 Legal Contract Intelligence API started")
    logger.info(f"📁 Temp directory: {TEMP_DIR}")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Downloads directory: {DOWNLOADS_DIR}")
    logger.info(f"🤖 Gemini API configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"🗣️ Sarvam API configured: {bool(SARVAM_API_KEY)}")
    logger.info(f"👁️ Google Vision configured: {vision_client is not None}")
    logger.info(f"📊 OCR confidence threshold: {OCR_CONFIDENCE_THRESHOLD}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    try:
        # Clean up temp files
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            logger.info("🧹 Cleaned up temporary files")
    except Exception as e:
        logger.error(f"Cleanup error on shutdown: {e}")
    
    logger.info("👋 Legal Contract Intelligence API shutdown complete")

