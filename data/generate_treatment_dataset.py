import pandas as pd
import random


respiratory = [
    "asthma", "bronchitis", "pneumonia", "chronic obstructive pulmonary disease",
    "tuberculosis", "sinusitis", "laryngitis", "influenza", "covid-19",
    "allergic rhinitis", "pleurisy", "pulmonary fibrosis"
]

digestive = [
    "gastritis", "gastroenteritis", "acid reflux", "peptic ulcer",
    "irritable bowel syndrome", "crohns disease", "ulcerative colitis",
    "food poisoning", "constipation", "diarrhea", "appendicitis",
    "gallstones", "pancreatitis", "celiac disease"
]

skin = [
    "eczema", "psoriasis", "acne", "dermatitis", "fungal infection",
    "ringworm", "impetigo", "cellulitis", "rosacea", "scabies",
    "melanoma", "urticaria", "skin abscess"
]

neurological = [
    "migraine", "tension headache", "epilepsy", "parkinsons disease",
    "multiple sclerosis", "meningitis", "brain tumor", "alzheimer disease",
    "peripheral neuropathy", "cluster headache", "stroke", "concussion"
]

endocrine = [
    "diabetes mellitus", "hyperthyroidism", "hypothyroidism",
    "cushing syndrome", "addison disease", "metabolic syndrome",
    "polycystic ovary syndrome", "goiter", "thyroid cancer"
]

infections = [
    "malaria", "dengue fever", "typhoid fever", "hepatitis a",
    "hepatitis b", "hepatitis c", "chickenpox", "measles",
    "mumps", "rabies", "lyme disease", "zika virus"
]

cardio = [
    "hypertension", "coronary artery disease", "heart failure",
    "arrhythmia", "myocardial infarction", "cardiomyopathy",
    "pericarditis", "aortic stenosis", "atrial fibrillation"
]

urology = [
    "kidney stones", "urinary tract infection", "prostatitis",
    "benign prostatic hyperplasia", "kidney infection"
]


all_diseases = (
    respiratory
    + digestive
    + skin
    + neurological
    + endocrine
    + infections
    + cardio
    + urology
)

medications = [
    "paracetamol",
    "ibuprofen",
    "antibiotics",
    "antihistamines",
    "oral rehydration salts",
    "steroids",
    "proton pump inhibitors",
    "bronchodilators",
    "insulin",
    "antivirals"
]

precautions = [
    "rest",
    "hydration",
    "avoid cold exposure",
    "avoid spicy food",
    "maintain hygiene",
    "monitor symptoms",
    "avoid allergens",
    "regular exercise"
]

diets = [
    "light diet",
    "balanced diet",
    "low sugar diet",
    "high fiber diet",
    "low salt diet",
    "soft foods",
    "warm fluids"
]

doctors = [
    "general physician",
    "pulmonologist",
    "gastroenterologist",
    "dermatologist",
    "neurologist",
    "cardiologist",
    "endocrinologist",
    "urologist",
    "infectious disease specialist"
]

severity_levels = ["mild", "moderate", "serious"]


rows = []

# generate ~300 rows
for i in range(300):

    disease = random.choice(all_diseases)

    row = {
        "disease": disease,
        "medications": ";".join(random.sample(medications, 2)),
        "precautions": ";".join(random.sample(precautions, 2)),
        "diet": random.choice(diets),
        "doctor_type": random.choice(doctors),
        "severity": random.choice(severity_levels)
    }

    rows.append(row)


df = pd.DataFrame(rows)

df.to_csv("disease_treatment_dataset_large.csv", index=False)

print("Dataset generated: disease_treatment_dataset_large.csv")