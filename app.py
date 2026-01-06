from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Legal backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(data: TextInput):
    return {
        "summary": "This is a demo response",
        "risk_level": "Medium"
    }
