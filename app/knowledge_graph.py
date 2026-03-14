import json

with open("data/medical_knowledge_graph.json") as f:
    GRAPH = json.load(f)


def graph_reasoning(symptoms):

    scores = {}

    for disease, data in GRAPH.items():

        disease_symptoms = data["symptoms"]

        overlap = len(set(symptoms) & set(disease_symptoms))

        if overlap > 0:
            score = overlap / len(disease_symptoms)
            scores[disease] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:5]


def get_disease_details(disease):

    if disease in GRAPH:

        data = GRAPH[disease]

        return {
            "medications": ";".join(data.get("treatments", [])),
            "doctor": data.get("doctor"),
            "severity": data.get("severity")
        }

    return {}