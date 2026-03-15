from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import predict_disease, load_resources, symptom_list
from app.symptom_extractor import extract_symptoms
from app.triage import check_emergency
from app.question_engine import generate_followups
from app.context_parser import extract_duration, detect_severity

app = FastAPI(title="Healthcare Symptom Checker API")

# ---------- Load ML resources once ----------
@app.on_event("startup")
def startup():
    load_resources()


# ---------- CORS ----------
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

    user_text = data.symptoms.lower()

    detected = extract_symptoms(user_text, symptom_list)

    duration = extract_duration(user_text)

    severity = detect_severity(user_text)

    if check_emergency(detected):
        return {
            "emergency": True,
            "message": "Possible emergency detected. Seek immediate medical care."
        }

    predictions = predict_disease(detected)

    followups = generate_followups(detected)

    return {
        "input_text": user_text,
        "detected_symptoms": detected,
        "duration": duration,
        "severity": severity,
        "predictions": predictions,
        "follow_up_questions": followups
    }