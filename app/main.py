import os
import sys
import traceback

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import app.predictor as predictor

from app.symptom_extractor import extract_symptoms
from app.triage import check_emergency
from app.question_engine import generate_followups
from app.context_parser import extract_duration, detect_severity


# Prevent XGBoost from spawning many threads (Render memory fix)
os.environ["OMP_NUM_THREADS"] = "1"


app = FastAPI(title="Healthcare Symptom Checker API")


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"ERROR: {exc}", file=sys.stderr)
    traceback.print_exc()

    return {
        "error": True,
        "message": str(exc),
        "traceback": traceback.format_exc()
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "cwd": os.getcwd(),
        "files": os.listdir(".")
    }


# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class SymptomInput(BaseModel):
    symptoms: str


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict")
def predict(data: SymptomInput):

    user_text = data.symptoms.lower()

    detected = extract_symptoms(user_text, predictor.get_symptom_list())

    duration = extract_duration(user_text)
    severity = detect_severity(user_text)

    if check_emergency(detected):
        return {
            "emergency": True,
            "message": "Possible emergency detected. Seek immediate medical care."
        }

    predictions = predictor.predict_disease(detected)

    followups = generate_followups(detected)

    return {
        "input_text": user_text,
        "detected_symptoms": detected,
        "duration": duration,
        "severity": severity,
        "predictions": predictions,
        "follow_up_questions": followups
    }