import re

from sentence_transformers import SentenceTransformer, util


# load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


SYMPTOM_SYNONYMS = {

    "headache": [
        "headache",
        "head hurts",
        "head pain",
        "pounding head",
        "migraine pain",
        "my head is killing me",
        "pain in head"
    ],

    "fever": [
        "fever",
        "feeling hot",
        "burning up",
        "high temperature"
    ],

    "vomiting": [
        "vomiting",
        "throwing up",
        "throw up",
        "vomit"
    ],

    "abdominal pain": [
        "stomach pain",
        "stomach ache",
        "belly pain"
    ],

    "dizziness": [
        "feeling dizzy",
        "lightheaded"
    ],

    "fatigue": [
        "extremely tired",
        "very tired",
        "exhausted"
    ],

    "shortness of breath": [
        "difficulty breathing",
        "cannot breathe",
        "cant breathe"
    ]
}


def normalize_text(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    return text


def semantic_symptom_match(user_input, symptom_list):

    detected = []

    try:

        user_embedding = model.encode([user_input], convert_to_tensor=True)
        symptom_embeddings = model.encode(symptom_list, convert_to_tensor=True)

        scores = util.cos_sim(user_embedding, symptom_embeddings)[0]

        for i, score in enumerate(scores):

            if float(score) > 0.55:
                detected.append(symptom_list[i])

    except Exception:
        pass

    return detected


def extract_symptoms(user_input, symptom_list):

    user_input = normalize_text(user_input)

    detected = []

    # direct match
    for symptom in symptom_list:

        if symptom in user_input:
            detected.append(symptom)

    # synonym phrase match
    for symptom, phrases in SYMPTOM_SYNONYMS.items():

        for phrase in phrases:

            if phrase in user_input:
                detected.append(symptom)

    # semantic matching
    detected += semantic_symptom_match(user_input, symptom_list)

    return list(set(detected))