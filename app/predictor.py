import joblib
import csv
import numpy as np

from app.knowledge_graph import graph_reasoning, get_disease_details


# --------- GLOBAL OBJECTS (LAZY LOADED) ---------
_model = None
_encoder = None
_symptom_list = None
_symptom_index = None
_treatment_dict = None


# --------- RESOURCE GETTERS (Ensures Lazy Loading) ---------
def get_model():
    global _model
    if _model is None:
        print("Loading ML model (mmap)...")
        _model = joblib.load("models/disease_prediction_model.pkl", mmap_mode='r')
        print("Model loaded successfully.")
    return _model

def get_encoder():
    global _encoder
    if _encoder is None:
        print("Loading label encoder...")
        _encoder = joblib.load("models/label_encoder.pkl")
        print("Encoder loaded.")
    return _encoder

def get_symptom_list():
    global _symptom_list, _symptom_index
    if _symptom_list is None:
        print("Loading symptom list...")
        _symptom_list = joblib.load("models/symptom_list.pkl")
        _symptom_index = {s: i for i, s in enumerate(_symptom_list)}
        print("Symptom list loaded.")
    return _symptom_list

def get_symptom_index():
    if _symptom_list is None: get_symptom_list()
    return _symptom_index

def get_treatment_dict():
    global _treatment_dict
    if _treatment_dict is None:
        _treatment_dict = {}
        try:
            with open("data/disease_treatment_dataset.csv", mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease = row.get("disease")
                    if disease and disease not in _treatment_dict:
                        _treatment_dict[disease] = row
        except Exception as e:
            print(f"Error loading treatment data: {e}")
    return _treatment_dict


# DEPRECATED: load_resources is now handled by individual getters
def load_resources():
    get_model()
    get_encoder()
    get_symptom_list()
    get_treatment_dict()


# --------- CONVERT SYMPTOMS TO VECTOR ---------
def symptoms_to_vector(symptoms):

    symptom_list = get_symptom_list()
    symptom_index = get_symptom_index()

    if isinstance(symptoms, str):
        symptoms = [symptoms]

    symptoms = [s.lower().strip() for s in symptoms]

    # memory optimized vector
    vector = np.zeros(len(symptom_list), dtype=np.int8)

    for symptom in symptoms:

        if symptom in symptom_index:
            idx = symptom_index[symptom]
            vector[idx] = 1

    return vector.reshape(1, -1)


# --------- DISEASE PREDICTION ---------
def predict_disease(symptoms):

    model = get_model()
    encoder = get_encoder()
    treatment_dict = get_treatment_dict()

    vector = symptoms_to_vector(symptoms)

    probs = model.predict_proba(vector)[0]

    top5 = np.argsort(probs)[-5:][::-1]

    results = []
    seen_diseases = set()

    # ---------- ML MODEL PREDICTIONS ----------
    for i in top5:

        disease = encoder.inverse_transform([i])[0]

        # filter unrealistic predictions
        if "pregnan" in disease.lower():
            continue

        probability = round(float(probs[i] * 100), 2)

        treatment = treatment_dict.get(disease, {})

        results.append({
            "disease": disease,
            "probability": probability,
            "medications": treatment.get("medications"),
            "precautions": treatment.get("precautions"),
            "diet": treatment.get("diet"),
            "doctor": treatment.get("doctor_type"),
            "severity": treatment.get("severity"),
            "source": "ml_model"
        })

        seen_diseases.add(disease)


    # ---------- KNOWLEDGE GRAPH REASONING ----------
    graph_predictions = graph_reasoning(symptoms)

    for disease, score in graph_predictions:

        if disease in seen_diseases:
            continue

        details = get_disease_details(disease)

        results.append({
            "disease": disease,
            "probability": round(score * 100, 2),
            "medications": details.get("medications"),
            "precautions": None,
            "diet": None,
            "doctor": details.get("doctor"),
            "severity": details.get("severity"),
            "source": "knowledge_graph"
        })

    # ---------- SORT RESULTS ----------
    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    return results[:5]