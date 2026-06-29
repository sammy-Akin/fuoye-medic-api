from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import json
import numpy as np
import os

# Initialize app
app = FastAPI(
    title="Nigerian Health Advisory API",
    description="AI-powered health advisory system for Nigerian diseases",
    version="1.0.0"
)

# CORS — allows Flutter app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, encoder and metadata
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'health_classifier.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

with open(os.path.join(BASE_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

SYMPTOMS = metadata['symptoms']
DISEASES = metadata['diseases']

# ─── RED FLAG RULES ─────────────────────────────────────────────────────────
RED_FLAG_RULES = {
    'blood in vomit':        ('EMERGENCY', 'Possible internal bleeding. Go to the nearest hospital immediately.'),
    'coughing up blood':     ('EMERGENCY', 'Possible tuberculosis or lung condition. Seek emergency care now.'),
    'chest pain':            ('URGENT',    'Possible cardiac event. Do not ignore. Visit hospital immediately.'),
    'difficulty breathing':  ('URGENT',    'Respiratory emergency. Seek care immediately.'),
    'loss of consciousness': ('EMERGENCY', 'Call emergency services immediately.'),
    'seizure':               ('EMERGENCY', 'Call emergency services immediately.'),
    'high fever':            ('URGENT',    'Fever above 39°C. Could be malaria or typhoid. See a doctor today.'),
    'severe abdominal pain': ('URGENT',    'Could indicate perforated ulcer or appendicitis. Go to hospital now.'),
    'blood in urine':        ('URGENT',    'Could indicate serious kidney or bladder condition. See a doctor today.'),
    'yellowing of skin':     ('URGENT',    'Possible jaundice or liver condition. See a doctor today.'),
    'unconscious':           ('EMERGENCY', 'Call emergency services immediately.'),
    'not breathing':         ('EMERGENCY', 'Call emergency services immediately.'),
}

def check_red_flags(text: str):
    text_lower = text.lower()
    for keyword, (level, message) in RED_FLAG_RULES.items():
        if keyword in text_lower:
            return level, message
    return None, None

# ─── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────
class SymptomRequest(BaseModel):
    symptoms: List[str]          # list of symptom names user has
    user_text: Optional[str] = ""  # raw voice/text input for red flag check

class PredictionResponse(BaseModel):
    source: str
    disease: Optional[str] = None
    confidence: Optional[float] = None
    level: str
    message: str
    all_predictions: Optional[dict] = None

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Nigerian Health Advisory API is running",
        "version": "1.0.0",
        "diseases": len(DISEASES),
        "symptoms": len(SYMPTOMS)
    }

@app.get("/symptoms")
def get_symptoms():
    """Returns the full list of supported symptoms"""
    return {
        "total": len(SYMPTOMS),
        "symptoms": SYMPTOMS
    }

@app.get("/diseases")
def get_diseases():
    """Returns the full list of supported diseases"""
    return {
        "total": len(DISEASES),
        "diseases": DISEASES
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: SymptomRequest):
    """
    Main prediction endpoint.
    1. Checks red flags first
    2. Runs ML classifier
    3. Returns disease prediction with confidence
    """

    # Step 1 — Red Flag Check
    if request.user_text:
        level, message = check_red_flags(request.user_text)
        if level:
            return PredictionResponse(
                source="RED_FLAG_OVERRIDE",
                level=level,
                message=message
            )

    # Step 2 — Build symptom vector
    symptom_vector = np.zeros(len(SYMPTOMS))
    unrecognized = []

    for symptom in request.symptoms:
        symptom_clean = symptom.lower().strip().replace(' ', '_')
        if symptom_clean in SYMPTOMS:
            idx = SYMPTOMS.index(symptom_clean)
            symptom_vector[idx] = 1
        else:
            unrecognized.append(symptom)

    # Step 3 — ML Prediction
    prediction = model.predict([symptom_vector])[0]
    probabilities = model.predict_proba([symptom_vector])[0]
    disease = le.inverse_transform([prediction])[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    # All disease probabilities
    all_predictions = {
        le.inverse_transform([i])[0]: round(float(p) * 100, 2)
        for i, p in enumerate(probabilities)
        if p > 0.01
    }

    # Step 4 — Confidence threshold check
    if confidence < 60:
        return PredictionResponse(
            source="LOW_CONFIDENCE",
            disease=disease,
            confidence=confidence,
            level="INFO",
            message=f"Symptoms are unclear. Please provide more details or consult a doctor.",
            all_predictions=all_predictions
        )

    return PredictionResponse(
        source="ML_CLASSIFIER",
        disease=disease,
        confidence=confidence,
        level="INFO",
        message=f"Based on your symptoms, this may be {disease}. Please consult a qualified doctor for proper diagnosis.",
        all_predictions=all_predictions
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}