from fastapi import FastAPI
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
    version="3.0.0"
)

# CORS
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

# API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─── RED FLAG RULES ──────────────────────────────────────────────────────────
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

# ─── SHARED SYMPTOM EXTRACTION PROMPT ────────────────────────────────────────


def build_extraction_prompt(user_text: str) -> str:
    symptom_list_str = ", ".join(SYMPTOMS)
    return f"""You are a medical symptom extraction assistant trained specifically for Nigerian patients.
You understand both standard English and Nigerian Pidgin English expressions for symptoms.

A patient described their condition in their own words:
"{user_text}"

Here is the EXACT list of valid symptom codes you must choose from:
{symptom_list_str}

Task: Identify which symptoms from the list above match what the patient described.
Think carefully about the medical meaning behind everyday Nigerian descriptions.

Nigerian Pidgin and local expression mappings to guide you:
- "biting stomach" / "stomach dey bite me" / "belle dey pain me" → burning_stomach_pain, epigastric_pain
- "stomach pain at night" / "pain wake me up for night" → stomach_pain_at_night, pain_worsens_at_night
- "hot and cold" / "I dey feel cold then hot" / "body dey do me up and down" → fever, chills, sweating
- "I dey shake" / "my body dey shake" / "shaking anyhow" → shivering, chills
- "body dey hot" / "I get fever" / "feverish" / "temperature dey high" → fever, prolonged_fever
- "I no fit carry myself" / "I dey weak" / "no strength" / "I dey tire anyhow" → fatigue, weakness, body_weakness
- "my head dey bang" / "head dey pain me" / "headache dey do me" → headache, severe_headache
- "I dey vomit" / "throwing up" / "I dey purge from mouth" → vomiting, nausea
- "running stomach" / "stooling" / "I dey purge" / "my yansh dey run" → diarrhoea
- "I no fit chop" / "I no wan eat" / "food no sweet me" → loss_of_appetite
- "my eye don yellow" / "yellow eyes" / "my skin don yellow" → jaundice, yellowing_of_eyes
- "my piss don change colour" / "dark urine" / "my piss yellow well well" → dark_urine
- "I dey urinate too much" / "I dey piss anyhow" → frequent_urination
- "I dey thirst well well" / "I dey drink water anyhow" → excessive_thirst
- "I don lose weight" / "I don slim down" / "my cloth don big for me" → unexplained_weight_loss, weight_loss
- "my bone dey pain me" / "joint dey pain me" / "body ache anyhow" → severe_bone_pain, joint_pain, muscle_pain
- "chest dey pain me" / "my chest tight" → chest_pain, shortness_of_breath
- "I dey cough blood" / "blood dey come out when I cough" → coughing_blood, blood_in_sputum
- "I dey sweat for night" / "night sweating" → night_sweats, sweating
- "I dey cough anyhow" / "cough no gree stop" → persistent_cough, cough, dry_cough
- "I dey see double" / "my eye dey blur" → blurred_vision
- "my belle don big" / "stomach don swell" → bloating
- "I dey belch anyhow" / "gas dey release" → belching, indigestion
- "my skin dey itch" / "body dey scratch me" → itchy_skin
- "wound no dey heal" / "cut no dey close" → slow_healing_wounds
- "I dey urinate with pain" / "my piss dey pain me" → blood_in_urine
- "my hand and leg dey numb" / "I no dey feel my leg" → numbness_in_feet, tingling_in_hands_feet
- "I dey sweat well well" / "sweat dey pour me" → sweating, night_sweats
- "my throat dey pain me" / "sore throat dey do me" → sore_throat
- "I dey breathe anyhow" / "breathing hard" / "breath short" → difficulty_breathing, shortness_of_breath, rapid_breathing
- "my heart dey beat fast" / "heart dey do gbim gbim" → rapid_heartbeat, palpitations
- "I don pale" / "I don fade" → pale_skin, anaemia
- "I no fit sleep" / "sleep no come" → difficulty_sleeping
- "my nose dey bleed" / "blood dey comot from nose" → nosebleed

Return ONLY a comma-separated list of matching symptom codes from the list above,
exactly as they appear in the list (with underscores).
If no symptoms match, return "none".
Do not explain. Do not add extra text. Only return the comma-separated codes."""


def build_advisory_prompt(disease: str, symptoms: List[str], confidence: float) -> str:
    return f"""You are FUOYE Medic, a friendly and professional health advisory assistant for Nigerian patients.

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


def strip_think_tags(text: str) -> str:
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def parse_symptoms(raw_text: str) -> List[str]:
    # Strip thinking tags from reasoning models
    import re
    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    if not raw_text or raw_text.lower().strip() == "none":
        return []
    extracted = [s.strip() for s in raw_text.split(",")]
    return [s for s in extracted if s in SYMPTOMS]

# ─── GROQ FUNCTIONS ───────────────────────────────────────────────────────────


async def extract_symptoms_with_groq(user_text: str) -> List[str]:
    if not user_text:
        return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3.6-27b",
                    "messages": [{"role": "user", "content": build_extraction_prompt(user_text)}],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            )
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                raw_text = data["choices"][0]["message"]["content"].strip()
                return parse_symptoms(raw_text)
            return []
    except Exception:
        return []


async def get_groq_advisory(disease: str, symptoms: List[str], confidence: float) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3.6-27b",
                    "messages": [{"role": "user", "content": build_advisory_prompt(disease, symptoms, confidence)}],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return strip_think_tags(data["choices"][0]["message"]["content"].strip())
            return ""
    except Exception:
        return ""

# ─── GEMINI FUNCTIONS (FALLBACK) ──────────────────────────────────────────────


async def extract_symptoms_with_gemini(user_text: str) -> List[str]:
    if not user_text:
        return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GEMINI_URL,
                json={"contents": [
                    {"parts": [{"text": build_extraction_prompt(user_text)}]}]}
            )
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    raw_text = candidate["content"]["parts"][0]["text"].strip()
                    return parse_symptoms(raw_text)
            return []
    except Exception:
        return []


async def get_gemini_advisory(disease: str, symptoms: List[str], confidence: float) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GEMINI_URL,
                json={"contents": [
                    {"parts": [{"text": build_advisory_prompt(disease, symptoms, confidence)}]}]}
            )
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    return candidate["content"]["parts"][0]["text"]
            return ""
    except Exception:
        return ""

# ─── COMBINED FUNCTIONS (GROQ FIRST, GEMINI FALLBACK) ────────────────────────


async def extract_symptoms(user_text: str) -> List[str]:
    symptoms = await extract_symptoms_with_groq(user_text)
    if not symptoms:
        symptoms = await extract_symptoms_with_gemini(user_text)
    return symptoms


async def get_advisory(disease: str, symptoms: List[str], confidence: float) -> str:
    advisory = await get_groq_advisory(disease, symptoms, confidence)
    if not advisory:
        advisory = await get_gemini_advisory(disease, symptoms, confidence)
    if not advisory:
        advisory = f"Based on your symptoms, you may have {disease}. Please consult a qualified doctor for proper diagnosis and treatment."
    return advisory

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
        "version": "3.0.0",
        "diseases": len(DISEASES),
        "symptoms": len(SYMPTOMS),
        "llm": "Groq (Llama3) primary + Gemini fallback"
    }


@app.get("/symptoms")
def get_symptoms():
    return {"total": len(SYMPTOMS), "symptoms": SYMPTOMS}


@app.get("/diseases")
def get_diseases():
    return {"total": len(DISEASES), "diseases": DISEASES}


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

    # Step 2 — Extract symptoms (Groq first, Gemini fallback)
    extracted_symptoms = []
    if request.user_text:
        extracted_symptoms = await extract_symptoms(request.user_text)

    # Step 3 — Combine with manually selected symptoms
    all_symptoms = list(set(request.symptoms + extracted_symptoms))

    # Step 4 — Build symptom vector
    symptom_vector = np.zeros(len(SYMPTOMS))
    for symptom in all_symptoms:
        symptom_clean = symptom.lower().strip().replace(' ', '_')
        if symptom_clean in SYMPTOMS:
            idx = SYMPTOMS.index(symptom_clean)
            symptom_vector[idx] = 1

    # Step 5 — ML Prediction
    prediction = model.predict([symptom_vector])[0]
    probabilities = model.predict_proba([symptom_vector])[0]
    disease = le.inverse_transform([prediction])[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    all_predictions = {
        le.inverse_transform([i])[0]: round(float(p) * 100, 2)
        for i, p in enumerate(probabilities)
        if p > 0.01
    }

    # Step 6 — Get Advisory (Groq first, Gemini fallback)
    advisory = await get_advisory(disease, all_symptoms, confidence)

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

# ─── CONVERSATION STATE MODEL ─────────────────────────────────────────────────


class ConversationRequest(BaseModel):
    user_text: str
    conversation_history: Optional[List[dict]] = []
    accumulated_symptoms: Optional[List[str]] = []
    question_count: Optional[int] = 0


class ConversationResponse(BaseModel):
    type: str  # "question" or "prediction"
    question: Optional[str] = None
    source: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    level: str
    message: str
    advisory: Optional[str] = None
    all_predictions: Optional[dict] = None
    accumulated_symptoms: Optional[List[str]] = []
    question_count: Optional[int] = 0


def _fallback_question(symptoms: List[str], question_count: int) -> str:
    questions = [
        "Do you also have fever or chills along with these symptoms?",
        "Have you been vomiting or feeling nauseous recently?",
        "Are you feeling weak or losing your appetite?",
        "Is there any yellowing of your eyes or skin?",
        "How long have you been experiencing these symptoms?"
    ]
    return questions[question_count % len(questions)]

# ─── FOLLOW-UP QUESTION GENERATOR ────────────────────────────────────────────


async def generate_follow_up_question(
    symptoms_so_far: List[str],
    question_count: int,
    top_diseases: dict
) -> str:
    top_2 = list(top_diseases.items())[:2]
    disease_context = " or ".join([d for d, _ in top_2])

    prompt = f"""You are FUOYE Medic, a friendly Nigerian doctor having a conversation with a patient.

The patient has described these symptoms so far: {', '.join(symptoms_so_far) if symptoms_so_far else 'vague discomfort'}.

This is follow-up question number {question_count + 1} of maximum 3.

Your job is to ask ONE smart, clinically relevant follow-up question.

Clinical follow-up rules:
- headache → ask about fever, chills, neck stiffness, or sensitivity to light
- fever / hot body / body dey hot → ask about chills, sweating, duration, or vomiting
- stomach pain / belle pain / epigastric pain → ask about timing (night/after eating), vomiting, or stool colour
- cough → ask about blood in sputum, night sweats, chest pain, or duration
- weakness / fatigue / I dey weak → ask about weight loss, appetite, or frequent urination
- yellow eyes / jaundice → ask about dark urine, pale stool, or abdominal pain
- bone pain / joint pain → ask about swelling, skin colour, or previous episodes
- frequent urination / excessive thirst → ask about weight loss, blurred vision, or fatigue
- chest pain → ask about breathing difficulty, sweating, or arm/jaw pain
- vomiting → ask about blood in vomit, frequency, or abdominal pain
- diarrhea / running stomach → ask about blood in stool, frequency, or dehydration signs
- weight loss → ask about appetite, cough, night sweats, or fatigue
- skin rash / itching → ask about fever, joint pain, or recent travel
- shortness of breath → ask about cough, chest pain, or leg swelling

General rules:
- Never ask about unrelated body systems
- Keep question short and friendly in simple English
- Understand Nigerian Pidgin expressions
- Do not mention disease names
- If symptoms unclear, ask about fever first as it's most common Nigerian complaint

Return ONLY the question. Nothing else."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3.6-27b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            )
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                raw = data["choices"][0]["message"]["content"].strip()
                content = strip_think_tags(raw)
                # Extra safety — take only last line if multiple lines
                if content:
                    lines = [l.strip() for l in content.split('\n') if l.strip()]
                    return lines[-1] if lines else _fallback_question(symptoms_so_far, question_count)
                return _fallback_question(symptoms_so_far, question_count)
            
            return _fallback_question(symptoms_so_far, question_count)
    except Exception:
        return "Can you tell me more about your symptoms? Do you have fever, vomiting, or any other discomfort?"

# ─── CONVERSATIONAL ENDPOINT ──────────────────────────────────────────────────


@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    # Step 1 — Red Flag Check
    level, message = check_red_flags(request.user_text)
    if level:
        return ConversationResponse(
            type="prediction",
            source="RED_FLAG_OVERRIDE",
            level=level,
            message=message,
            advisory="Please seek immediate medical attention. Do not delay.",
            accumulated_symptoms=request.accumulated_symptoms,
            question_count=request.question_count
        )

    # Step 2 — Extract symptoms from current message
    new_symptoms = await extract_symptoms(request.user_text)

    # Step 3 — Combine with accumulated symptoms
    all_symptoms = list(set(request.accumulated_symptoms + new_symptoms))

    # Step 4 — Build symptom vector and predict
    symptom_vector = np.zeros(len(SYMPTOMS))
    for symptom in all_symptoms:
        symptom_clean = symptom.lower().strip().replace(' ', '_')
        if symptom_clean in SYMPTOMS:
            idx = SYMPTOMS.index(symptom_clean)
            symptom_vector[idx] = 1

    prediction = model.predict([symptom_vector])[0]
    probabilities = model.predict_proba([symptom_vector])[0]
    disease = le.inverse_transform([prediction])[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    all_predictions = {
        le.inverse_transform([i])[0]: round(float(p) * 100, 2)
        for i, p in enumerate(probabilities)
        if p > 0.01
    }

    # Step 5 — Decide: ask follow-up or give prediction
    max_questions = 3
    should_ask = (
        confidence < 70 and
        request.question_count < max_questions and
        len(all_symptoms) < 5
    )

    if should_ask:
        # Generate smart follow-up question
        question = await generate_follow_up_question(
            all_symptoms,
            request.question_count,
            all_predictions
        )
        return ConversationResponse(
            type="question",
            question=question,
            level="INFO",
            message="I need a bit more information to help you better.",
            accumulated_symptoms=all_symptoms,
            question_count=request.question_count + 1
        )

    # Step 6 — Enough info, give final prediction
    advisory = await get_advisory(disease, all_symptoms, confidence)

    return ConversationResponse(
        type="prediction",
        source="ML_CLASSIFIER",
        disease=disease,
        confidence=confidence,
        level="INFO",
        message=f"Based on your symptoms, this may be {disease}. Please consult a qualified doctor for proper diagnosis.",
        advisory=advisory,
        all_predictions=all_predictions,
        accumulated_symptoms=all_symptoms,
        question_count=request.question_count
    )


@app.get("/test-groq")
async def test_groq():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3.6-27b",
                    "messages": [{"role": "user", "content": "Say hello in one sentence"}],
                    "max_tokens": 50
                }
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/test-gemini")
async def test_gemini():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GEMINI_URL,
                json={"contents": [
                    {"parts": [{"text": "Say hello in one sentence"}]}]}
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/test-extraction")
async def test_extraction():
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen/qwen3.6-27b",
                    "messages": [{"role": "user", "content": build_extraction_prompt("I have headache and fever")}],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            )
            data = response.json()
            raw = data["choices"][0]["message"]["content"] if "choices" in data else "no choices"
            return {"raw_response": raw, "full_data": data}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/health")
def health_check():
    return {"status": "healthy"}
