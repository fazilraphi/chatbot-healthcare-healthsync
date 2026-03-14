import joblib
import pandas as pd
import numpy as np

from app.knowledge_graph import graph_reasoning, get_disease_details


model = joblib.load("models/disease_prediction_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")
symptom_list = joblib.load("models/symptom_list.pkl")


treatment_df = pd.read_csv("data/disease_treatment_dataset.csv")
treatment_df = treatment_df.drop_duplicates(subset="disease")

treatment_dict = treatment_df.set_index("disease").to_dict(orient="index")


def symptoms_to_vector(symptoms):

    vector = np.zeros(len(symptom_list))

    for symptom in symptoms:

        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            vector[idx] = 1

    return vector.reshape(1, -1)


def predict_disease(symptoms):

    vector = symptoms_to_vector(symptoms)

    probs = model.predict_proba(vector)[0]

    top5 = np.argsort(probs)[-5:][::-1]

    results = []

    seen_diseases = set()

    # ---------- ML MODEL PREDICTIONS ----------
    for i in top5:

        disease = encoder.inverse_transform([i])[0]

        # Filter unrealistic diseases
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

        # avoid duplicate diseases
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
   