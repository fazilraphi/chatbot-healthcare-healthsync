from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import app.predictor as predictor

from app.symptom_extractor import extract_symptoms
from app.triage import check_emergency
from app.question_engine import generate_followups
from app.context_parser import extract_duration, detect_severity


app = FastAPI(title="Healthcare Symptom Checker API")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomInput(BaseModel):
    symptoms: str


@app.get("/")
def home():
    return {"message": "Healthcare AI Chatbot API running"}


@app.post("/predict")
def predict(data: SymptomInput):

    # Load model resources ONLY when endpoint is called
    predictor.load_resources()

    user_text = data.symptoms.lower()

    # Extract symptoms
    detected = extract_symptoms(user_text, predictor.symptom_list)

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