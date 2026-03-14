FOLLOW_UP_QUESTIONS = {

    "headache": [
        "Is the headache severe or mild?",
        "Do you have sensitivity to light?",
        "Have you experienced nausea with the headache?"
    ],

    "fever": [
        "How high is the fever?",
        "How many days have you had the fever?",
        "Do you also have chills?"
    ],

    "abdominal pain": [
        "Where exactly is the pain located?",
        "Does the pain worsen after eating?",
        "Do you have diarrhea or vomiting?"
    ],

    "shortness of breath": [
        "Does breathing get worse when lying down?",
        "Do you have chest pain as well?",
        "Did this start suddenly?"
    ]
}


def generate_followups(symptoms):

    questions = []

    for symptom in symptoms:
        if symptom in FOLLOW_UP_QUESTIONS:
            questions.extend(FOLLOW_UP_QUESTIONS[symptom])

    return questions[:5]