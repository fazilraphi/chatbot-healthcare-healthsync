EMERGENCY_KEYWORDS = [
    "chest pain",
    "shortness of breath",
    "cannot breathe",
    "severe bleeding",
    "loss of consciousness",
    "stroke",
    "heart attack"
]


def check_emergency(symptoms):

    for symptom in symptoms:
        if symptom in EMERGENCY_KEYWORDS:
            return True

    return False