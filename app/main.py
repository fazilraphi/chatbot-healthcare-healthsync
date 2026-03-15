from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback
import sys
import os

import app.predictor as predictor

from app.symptom_extractor import extract_symptoms
from app.triage import check_emergency
from app.question_engine import generate_followups
from app.context_parser import extract_duration, detect_severity


app = FastAPI(title="Healthcare Symptom Checker API")


# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"ERROR: {exc}", file=sys.stderr)
    traceback.print_exc()
    return {
        "error": True, 
        "message": str(exc),
        "traceback": traceback.format_exc()
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cwd": os.getcwd(),
        "files": os.listdir(".")
    }

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")


class SymptomInput(BaseModel):
    symptoms: str


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict")
def predict(data: SymptomInput):

    user_text = data.symptoms.lower()

    # Extract symptoms (Lazy loads symptom_list if needed)
    detected = extract_symptoms(user_text, predictor.get_symptom_list())

    # Extract context
    duration = extract_duration(user_text)
    severity = detect_severity(user_text)

    # Emergency check
    if check_emergency(detected):
        return {
            "emergency": True,
            "message": "Possible emergency detected. Seek immediate medical care."
        }

    # Predict diseases
    predictions = predictor.predict_disease(detected)

    # Follow-up questions
    followups = generate_followups(detected)

    return {
        "input_text": user_text,
        "detected_symptoms": detected,
        "duration": duration,
        "severity": severity,
        "predictions": predictions,
        "follow_up_questions": followups
    }