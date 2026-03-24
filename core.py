import os
import re
import json
import base64
import requests
from dotenv import load_dotenv

from google import genai
from google.cloud import vision

from PyPDF2 import PdfReader
from pdf2image import convert_from_path

# ====================== ENV SETUP ======================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY missing")

if not PROJECT_ID:
    raise RuntimeError("❌ GOOGLE_CLOUD_PROJECT missing")

# ====================== GEMINI CLIENT ======================
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-1.5-flash-002"

# ====================== GOOGLE VISION CLIENT ======================
vision_client = vision.ImageAnnotatorClient()

OCR_CONFIDENCE_THRESHOLD = 0.70
MAX_OCR_RETRIES = 3

# ====================== OCR WITH CONFIDENCE ======================
def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    text = ""
    total_confidence = 0.0
    total_words = 0
    pages_processed = 0
    low_confidence_pages = []

    for i, img in enumerate(images):
        temp_file = f"temp_{i}.jpg"
        img.save(temp_file, "JPEG", quality=95)

        with open(temp_file, "rb") as f:
            image = vision.Image(content=f.read())

        response = vision_client.document_text_detection(image=image)

        if response.full_text_annotation:
            text += response.full_text_annotation.text + "\n"
            pages_processed += 1

            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            if hasattr(word, "confidence"):
                                total_confidence += word.confidence
                                total_words += 1

        os.remove(temp_file)

    avg_confidence = (total_confidence / total_words) if total_words else 0.0

    return {
        "text": text,
        "success": pages_processed > 0,
        "confidence": avg_confidence,
        "pages_processed": pages_processed,
        "word_count": len(text.split()) if text else 0,
        "low_confidence_pages": low_confidence_pages,
    }


def extract_text_from_pdf_with_confidence(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        if len(text.strip()) > 100:
            return {
                "text": text,
                "confidence": 1.0,
                "method": "direct",
                "success": True,
                "word_count": len(text.split()),
            }
    except Exception:
        pass

    result = extract_text_with_ocr(pdf_path)
    result["method"] = "ocr"
    return result


def check_extraction_quality(result):
    issues = []

    if not result.get("success"):
        issues.append("Failed to extract text")

    if result.get("confidence", 0) < OCR_CONFIDENCE_THRESHOLD:
        issues.append("Low OCR confidence")

    if result.get("word_count", 0) < 50:
        issues.append("Too few words extracted")

    return len(issues) == 0, issues


def handle_pdf_with_retry(pdf_path):
    result = extract_text_from_pdf_with_confidence(pdf_path)
    is_ok, _ = check_extraction_quality(result)

    return result.get("text", ""), result.get("confidence", 0), is_ok


# ====================== PII MASKING ======================
def mask_sensitive_data(text):
    patterns = [
        (r"\b\d{10}\b", "[PHONE]"),
        (r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN]"),
        (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[AADHAAR]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b", "[EMAIL]"),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    return text


# ====================== CONTRACT ANALYSIS ======================
def analyze_contract(text, name):
    prompt = f"""
Analyze the contract below and return ONLY valid JSON.

{{
  "contract_name": "{name}",
  "risk_level": "LOW | MODERATE | HIGH | CRITICAL",
  "overall_summary": "2–3 sentences",
  "key_risks": [
    {{"risk": "", "severity": "", "impact": ""}}
  ],
  "missing_protections": ["..."],
  "recommendations": [
    {{"action": "", "priority": "HIGH | MEDIUM | LOW"}}
  ]
}}

CONTRACT TEXT:
{text[:15000]}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    cleaned = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


# ====================== ASK BEFORE YOU SIGN ======================
def generate_questions_before_signing(analysis):
    prompt = f"""
Generate 3–5 neutral clarification questions a user should ask BEFORE signing.

Return ONLY JSON:
{{
  "questions": ["question 1", "question 2"]
}}

ANALYSIS:
{json.dumps(analysis)}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    cleaned = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


# ====================== CONTRACT COMPARISON ======================
def compare_contracts(a1, a2, n1, n2):
    prompt = f"""
Compare two contracts and decide which is better for the user.

Return ONLY JSON:
{{
  "better_contract": "{n1} or {n2}",
  "reason": "clear reasoning",
  "risk_comparison": {{
    "{n1}": "risk summary",
    "{n2}": "risk summary"
  }},
  "final_recommendation": "neutral suggestion"
}}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    cleaned = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


# ====================== SARVAM AI TTS ======================
def sarvam_text_to_speech(text, output_file="output.mp3"):
    if not SARVAM_API_KEY:
        return None

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": [text[:3000]],
        "target_language_code": "en-IN",
        "speaker": "meera",
        "model": "bulbul:v1",
    }

    r = requests.post(url, json=payload, headers=headers, timeout=30)

    if r.status_code == 200:
        audio_data = r.json()["audios"][0]["audioContent"]
        with open(output_file, "wb") as f:
            f.write(base64.b64decode(audio_data))
        return output_file

    return None
