import re

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


def extract_symptoms(user_input, symptom_list):

    user_input = normalize_text(user_input)

    detected = []

    # direct symptom match
    for symptom in symptom_list:
        if symptom in user_input:
            detected.append(symptom)

    # synonym phrase match
    for symptom, phrases in SYMPTOM_SYNONYMS.items():
        for phrase in phrases:
            if phrase in user_input:
                detected.append(symptom)

    return list(set(detected))