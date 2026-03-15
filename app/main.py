from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import predict_disease, symptom_list
from app.symptom_extractor import extract_symptoms
from app.triage import check_emergency
from app.question_engine import generate_followups
from app.context_parser import extract_duration, detect_severity


app = FastAPI(title="Healthcare Symptom Checker API")


# CORS configuration
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # allows all origins
    allow_credentials=True,
    allow_methods=["*"],        # allow all HTTP methods
    allow_headers=["*"],        # allow all headers
)


class SymptomInput(BaseModel):
    symptoms: str


@app.get("/")
def home():
    return {"message": "Healthcare AI Chatbot API running"}


@app.post("/predict")
def predict(data: SymptomInput):

    detected = extract_symptoms(data.symptoms, symptom_list)

    duration = extract_duration(data.symptoms)

    severity = detect_severity(data.symptoms)

    if check_emergency(detected):
        return {
            "emergency": True,
            "message": "Possible emergency detected. Seek immediate medical care."
        }

    predictions = predict_disease(detected)

    followups = generate_followups(detected)

    return {
        "detected_symptoms": detected,
        "duration": duration,
        "severity": severity,
        "predictions": predictions,
        "follow_up_questions": followups
    }