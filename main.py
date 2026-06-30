from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import json
import numpy as np
import os
import httpx

# Initialize app
app = FastAPI(
    title="FUOYE Medic - Nigerian Health Advisory API",
    description="AI-powered health advisory system for Nigerian diseases",
    version="2.0.0"
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

# Gemini API Key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

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
# ─── GEMINI SYMPTOM EXTRACTION ───────────────────────────────────────────────
async def extract_symptoms_with_gemini(user_text: str) -> List[str]:
    """
    Uses Gemini to convert natural language symptom descriptions
    into structured symptom codes matching our trained model.
    """
    if not user_text:
        return []

    symptom_list_str = ", ".join(SYMPTOMS)

    prompt = f"""You are a medical symptom extraction assistant.

A patient described their condition in their own words:
"{user_text}"

Here is the EXACT list of valid symptom codes you must choose from:
{symptom_list_str}

Task: Identify which symptoms from the list above match what the patient described.
Consider synonyms, descriptive language, and context. For example:
- "biting stomach pain" or "tummy pain" → stomach_pain or abdominal_pain
- "can't breathe well" → breathlessness
- "throwing up" → vomiting
- "running stomach" → diarrhoea

Return ONLY a comma-separated list of matching symptom codes from the list above, 
exactly as they appear in the list (with underscores). 
If no symptoms match, return "none".
Do not explain. Do not add extra text. Only return the comma-separated codes."""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            data = response.json()

            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    raw_text = candidate["content"]["parts"][0]["text"].strip()

                    if raw_text.lower() == "none":
                        return []

                    extracted = [s.strip() for s in raw_text.split(",")]
                    valid_symptoms = [s for s in extracted if s in SYMPTOMS]
                    return valid_symptoms

            return []
    except Exception as e:
        return []
# ─── GEMINI ADVISORY ─────────────────────────────────────────────────────────
async def get_gemini_advisory(disease: str, symptoms: List[str], confidence: float) -> str:
    prompt = f"""You are FUOYE Medic, a friendly and professional health advisory assistant for Nigerian patients.

A patient has presented with the following symptoms: {', '.join(symptoms)}.
Based on ML analysis, the predicted condition is: {disease} (confidence: {confidence}%).

Please provide a helpful health advisory response that includes:
1. A brief explanation of {disease} in simple terms
2. Common causes relevant to the Nigerian context
3. What the patient should do next (see a doctor, rest, hydration, etc.)
4. Any warning signs to watch out for
5. General prevention tips

Keep the response friendly, clear, and under 200 words.
Do not provide specific drug dosages.
Always recommend seeing a qualified doctor for proper diagnosis.
End with an encouraging note."""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    return candidate["content"]["parts"][0]["text"]
            error_msg = data.get("error", {}).get("message", "Unknown error")
            return f"Advisory unavailable ({error_msg}). You may have {disease}. Please consult a qualified doctor."
    except Exception as e:
        return f"Advisory service temporarily unavailable. Based on your symptoms, you may have {disease}. Please consult a qualified doctor."

# ─── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────
class SymptomRequest(BaseModel):
    symptoms: List[str]
    user_text: Optional[str] = ""

class PredictionResponse(BaseModel):
    source: str
    disease: Optional[str] = None
    confidence: Optional[float] = None
    level: str
    message: str
    advisory: Optional[str] = None
    all_predictions: Optional[dict] = None

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "FUOYE Medic - Nigerian Health Advisory API is running",
        "version": "2.0.0",
        "diseases": len(DISEASES),
        "symptoms": len(SYMPTOMS)
    }

@app.get("/symptoms")
def get_symptoms():
    return {
        "total": len(SYMPTOMS),
        "symptoms": SYMPTOMS
    }

@app.get("/diseases")
def get_diseases():
    return {
        "total": len(DISEASES),
        "diseases": DISEASES
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: SymptomRequest):
    # Step 1 — Red Flag Check
    if request.user_text:
        level, message = check_red_flags(request.user_text)
        if level:
            return PredictionResponse(
                source="RED_FLAG_OVERRIDE",
                level=level,
                message=message,
                advisory="Please seek immediate medical attention. Do not delay."
            )

    # Step 2 — Extract symptoms from natural language using Gemini
    extracted_symptoms = []
    if request.user_text:
        extracted_symptoms = await extract_symptoms_with_gemini(request.user_text)

    # Combine manually selected chips + Gemini-extracted symptoms (no duplicates)
    all_symptoms = list(set(request.symptoms + extracted_symptoms))

    # Step 3 — Build symptom vector
    symptom_vector = np.zeros(len(SYMPTOMS))

    for symptom in all_symptoms:
        symptom_clean = symptom.lower().strip().replace(' ', '_')
        if symptom_clean in SYMPTOMS:
            idx = SYMPTOMS.index(symptom_clean)
            symptom_vector[idx] = 1

    # Step 4 — ML Prediction
    prediction = model.predict([symptom_vector])[0]
    probabilities = model.predict_proba([symptom_vector])[0]
    disease = le.inverse_transform([prediction])[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    all_predictions = {
        le.inverse_transform([i])[0]: round(float(p) * 100, 2)
        for i, p in enumerate(probabilities)
        if p > 0.01
    }

    # Step 5 — Get Gemini Advisory
    advisory = await get_gemini_advisory(disease, all_symptoms, confidence)

    if confidence < 60:
        return PredictionResponse(
            source="LOW_CONFIDENCE",
            disease=disease,
            confidence=confidence,
            level="INFO",
            message="Symptoms are unclear. Please provide more details or consult a doctor.",
            advisory=advisory,
            all_predictions=all_predictions
        )

    return PredictionResponse(
        source="ML_CLASSIFIER",
        disease=disease,
        confidence=confidence,
        level="INFO",
        message=f"Based on your symptoms, this may be {disease}. Please consult a qualified doctor for proper diagnosis.",
        advisory=advisory,
        all_predictions=all_predictions
    )
    
@app.get("/test-gemini")
async def test_gemini():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json={
                "contents": [{"parts": [{"text": "Say hello in one sentence"}]}]
            })
            return response.json()
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/health")
def health_check():
    return {"status": "healthy"}