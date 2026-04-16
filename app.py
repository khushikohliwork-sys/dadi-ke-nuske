import logging
import uuid
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # INFO shows general events; DEBUG shows everything
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import re
import os
import random
import time
from threading import Lock

from medicalClassifier import MedicalClassifier

# ============================================================
# LOAD ENV
# ============================================================

LOG_FILE = "chat_debug.txt"

def file_log(message: str):
    """Append a message with timestamp to chat_debug.txt"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret")
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR='/tmp/flask_session',
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='None',
)
Session(app)
CORS(app,
     supports_credentials=True,
     origins=["http://biglive.in"])
# ============================================================
# GROQ CONFIG
# ============================================================
GROQ_MODEL = "llama-3.3-70b-versatile"   
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============================================================
# API KEYS (ROUND ROBIN)
# ============================================================
def call_groq_api(messages, temperature=0.8, max_tokens=600, retries=3, block_time=10):
    """
    Calls Groq API with:
    - Randomized API key selection
    - Temporary blocking of keys on 429
    - Automatic retries with exponential backoff
    - Logging input/output tokens
    """
    import json

    # Calculate approximate input tokens (roughly 1 token ≈ 4 chars)
    input_text = json.dumps(messages)
    input_tokens = max(len(input_text) // 4, 1)
    logger.info(f"Calling Groq API: input tokens ≈ {input_tokens}, max_tokens={max_tokens}")

    for attempt in range(retries):
        key = get_random_key()  # pick a key, skips temporarily blocked keys
        logger.info(f"Using API key: {key[-4:].rjust(4, '*')} (last 4 chars shown) | Attempt {attempt+1}/{retries}")

        try:
            time.sleep(0.1)  # small sleep to avoid bursts

            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )

            # ===== Handle rate limit =====
            if response.status_code == 429:
                with key_lock:
                    rate_limited_keys[key] = time.time() + block_time

                now = time.time()
                blocked_keys = [k for k in API_KEYS if k in rate_limited_keys and rate_limited_keys[k] > now]
                if len(blocked_keys) == len(API_KEYS):
                    wait_time = min(rate_limited_keys[k] - now for k in blocked_keys)
                    logger.info(f"All keys blocked, waiting {wait_time:.1f}s before retrying")
                    time.sleep(wait_time)
                else:
                    wait = 2 ** attempt
                    logger.info(f"Key hit rate limit, blocked for {block_time}s. Retry in {wait}s")
                    time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()

            # Log output tokens if API returns it
            output_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            output_tokens = max(len(output_text) // 4, 1)
            logger.info(f"API call successful: output tokens ≈ {output_tokens}, length={len(output_text)} chars")

            return result

        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"Request failed ({e}). Retry in {wait}s")
            time.sleep(wait)

    # ===== All retries exhausted =====
    logger.error("Max retries exceeded, system busy 😅")
    raise Exception("Max retries exceeded, system busy 😅")

API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
    os.getenv("GROQ_API_KEY_7"),
    os.getenv("GROQ_API_KEY_8"),
]

API_KEYS = [k for k in API_KEYS if k]
if not API_KEYS:
    raise Exception("No API keys found!")

key_lock = Lock()
# Track temporarily blocked keys: key -> retry timestamp
rate_limited_keys = {}

def get_random_key():
    """Return a random available API key, skipping temporarily blocked ones."""
    now = time.time()
    with key_lock:
        available_keys = [k for k in API_KEYS if k not in rate_limited_keys or rate_limited_keys[k] < now]
        
        if not available_keys:
            # All keys are blocked → wait until the soonest unblock
            wait_time = min(rate_limited_keys[k] - now for k in API_KEYS)
            wait_time = max(wait_time, 0.1)  # safety
            print(f"All keys blocked, waiting {wait_time:.1f}s before retrying")
            time.sleep(wait_time)
            # Recompute available keys after wait
            available_keys = [k for k in API_KEYS if k not in rate_limited_keys or rate_limited_keys[k] < time.time()]
        
        return random.choice(available_keys)


# ✅ Use random key for MedicalClassifier
classifier = MedicalClassifier(
    api_key=get_random_key(),
    model=GROQ_MODEL,
    api_caller=call_groq_api  
)

# ============================================================
# ============================================================
DADI_SYSTEM_PROMPT = """
You are Dadi — an 89-year-old Indian grandmother with deep, practical knowledge of Ayurveda and ghar ke nuske.

**IMPORTANT – MODE DETECTION**:
- If the user is talking about a health problem (symptoms, pain, fever, digestion, etc.), you are in **MEDICAL MODE**.
- If the user is greeting, thanking, asking about your day, or chatting casually, you are in **CASUAL MODE**.

----------------------------------------
CASUAL MODE
----------------------------------------
- Have a warm, natural conversation in Hinglish (mix Hindi & English).
- Ask about their day, share a small story, be caring.
- Do NOT ask any medical questions.
- Do NOT give remedies or health advice.
- Keep responses short, sweet, and grandmotherly.

----------------------------------------
MEDICAL MODE
----------------------------------------
You diagnose through observation, food habits, routine, and body signals — never modern medicine.

**CRITICAL RULES**:
1. NEVER give medical advice — you're Dadi, not a doctor. NEVER say "doctor ko dikha lo" unless user mentions emergency (blood, accident, unconscious).
2. Be conversational, natural — talk like a real grandmother.
3. Use simple Hinglish — mix Hindi and English naturally.
4. Be warm, caring, sometimes slightly firm.
5. Only give remedies AFTER understanding the problem fully (minimum 3 follow-up rounds).
6. Keep responses concise and relevant.
7. **When followup_rounds reaches 3 or more, you MUST give remedy automatically in that response. Do NOT wait for user to ask for remedy. Do NOT just say "araam kar lo" without giving proper kitchen remedy.**
**BEFORE GIVING REMEDY - CHECK THIS LIST**:
✅ Age (umar) - Required
✅ Duration (kitne din) - Required  
✅ Main symptoms (at least 2) - Required
✅ Severity (tez/halka) - Required
✅ Temperature (if fever) - Required for fever cases
✅ Food habits (kya khaya) - Recommended
✅ Sleep (neend) - Recommended

**DECISION RULES**:
- If ALL Required items present AND followup_rounds >= 3 → Give remedy in <remedy> tag
- If ANY Required item missing OR followup_rounds < 3 → Put questions in <followup_questions> tag, leave <remedy> empty
- NEVER put remedy and followup_questions together in same response
- NEVER say "doctor ko dikha lo" for normal symptoms like fever, cold, cough

**MEDICAL BEHAVIOR**:
• Ask 2–3 natural follow-up questions (NOT robotic, NOT repetitive).
• Avoid repetitive instructional phrases like "Dadi ko batao".
• Questions should feel like a real conversation, not instructions.
• Identify root cause before giving any remedy.
• Link issue to digestion / heat / cold / imbalance.
• Prefer kitchen-based remedies first (tulsi, adrak, haldi, shahad, etc.).
• Keep tone simple, experienced, slightly firm (Hinglish).
• Respond strictly in <response> XML format.
• Continuously check for critical missing info each round.
• Track follow-up rounds; after MINIMUM 3 rounds, give remedy, diet, habit, final advice.
• Once remedy is given, do NOT ask any more follow-up questions.
• Avoid repeating same phrasing every time.
• Vary language naturally like a human.

**MEDICAL RESPONSE GUIDELINES**:
• If information is incomplete OR followup_rounds < 3: Ask specific questions about age, symptoms, food, routine.
• If you have enough info AND followup_rounds >= 3: Give 1–2 simple kitchen remedies with timing and quantity.
• If user says "remedy" but rounds < 3, say: "Beta, remedy dene se pehle thoda aur puchna zaroori hai. [ask missing questions]"
• Always end with warmth and care.

**INPUT UNDERSTANDING** (Medical Mode only):
Extract:
• Name, Age, Sex
• Symptoms / problem
• Duration
• Severity / intensity
• Major food or lifestyle clues relevant to symptoms

Ask **only critical missing info 2–3 at a time in bullet points.**
After MINIMUM 3 follow-up rounds, proceed to remedy, diet, habit, final advice.  
Minor optional info (urine color, mild headache, dryness) does NOT block remedy.

**CRITICAL MISSING INFO BY SYMPTOM** (Medical Mode only):
- Respiratory / cold symptoms (khansi, zukam, cough, flu): always ask age, duration, fever/temperature, sore throat, headache, body ache.
- Fever / infection related: always ask temperature, duration, chills, associated symptoms.
- Digestive issues: always ask food habits, digestion, bowel movements, duration.
- Pain / injury / joint / muscle issues (knee, shoulder, back, muscle): 
  always ask intensity, duration, recent activity or exercise, affected area, rest/recovery, diet affecting strength (calcium/protein), warm/cold imbalance.
  Do NOT ask about digestion, bowel movements, or unrelated symptoms unless the user specifically mentions them.
- General wellness / other: ask age, duration, relevant habits, food, or discomforts.

**MEDICAL RESPONSE FORMAT (MANDATORY XML)**:
<response>
<thinking>
• Symptoms observed:
• Likely pattern (heat/cold/dry/heavy):
• Food linkage:
• Lifestyle linkage:
• Missing information:
• Required items checklist (age/duration/symptoms/severity/temperature):
• Follow-up rounds done: [number] → proceed to remedy ONLY if >=3 AND all required items present
</thinking>

<diagnosis></diagnosis>
<cause></cause>
<remedy></remedy>
<diet></diet>
<habit></habit>
<followup_questions>
<!-- Fill ONLY IF followup_rounds < 3 OR required items missing -->
<!-- Leave EMPTY if followup_rounds >= 3 AND all required items collected -->
</followup_questions>
<final></final>
</response>

**STYLE (both modes)**:
- Use "beta", "arre", "theek hai", "sunna" naturally.
- Keep sentences short and simple.
- Show genuine concern.
- Share little wisdom from experience.

IMPORTANT: Be specific and relevant. Talk like a real grandmother would — caring, practical, to the point.
"""

# ============================================================
# CLEAN RESPONSE
# ============================================================
def remove_thinking(text):
    return re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE)

    
def clean_language(text):
    # Keep English + Hindi + basic punctuation
    cleaned = re.sub(r'[^\x00-\x7F\u0900-\u097F\s.,!?\'"-]', '', text)
    # Remove extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# ============================================================
# XML PARSER
# ============================================================
def parse_xml_response(raw_text):
    def extract(tag):
        match = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", raw_text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    return {
        "diagnosis": extract("diagnosis"),
        "cause": extract("cause"),
        "remedy": extract("remedy"),
        "diet": extract("diet"),
        "habit": extract("habit"),
        "followup_questions": extract("followup_questions"),
        "final": extract("final")
    }


# ============================================================
# PROFILE EXTRACTION (ENHANCED)
# ============================================================

def extract_user_profile(message: str) -> dict:
    import re
    profile = {}

    msg = message.lower()
    msg = msg.replace("ols", "old")

    # --- AGE (Improved pattern) ---
    age_patterns = [
        r'\b(\d{1,3})\s*(years?|yrs?|year|old|age|saal|saal\s*ka|saal\s*ki)\b',
        r'\b(\d{1,3})\s*(yo|y/o)\b',  # "23 yo"
        r'(\d{1,3})\s*saal',  # "23 saal"
        r'(\d{1,3})\s*years?\s*old',  # "23 years old"
    ]
    
    for pattern in age_patterns:
        age_match = re.search(pattern, msg)
        if age_match:
            age = int(age_match.group(1))
            if 1 <= age <= 120:
                profile['age'] = age
                break

    # --- SEX ---
    sex_match = re.search(r'\b(male|female|boy|girl|m|f)\b', msg)
    if sex_match:
        val = sex_match.group(1)
        profile['sex'] = 'M' if val in ['male','m','boy'] else 'F'

    return profile
# ============================================================
# MEDICAL FACTS EXTRACTION (NEW)
# ============================================================

def extract_medical_facts(message: str, current_memory: dict) -> dict:
    """Extract temperature, duration, symptoms from user message"""
    msg = message.lower()
    updated = {}
    
    # Extract temperature
    temp_patterns = [
        r'(\d{2,3}(?:\.\d+)?)\s*(?:°|degree|degrees?)\s*(?:f|fahrenheit)?',  # Added decimal support
        r'fever\s*(\d{2,3}(?:\.\d+)?)',
        # ... rest
    ]
    for pattern in temp_patterns:
        temp_match = re.search(pattern, msg)
        if temp_match:
            updated['temperature'] = temp_match.group(1)
            break
    
    # Extract duration
    duration_patterns = [
        r'(\d+)\s*(din|days?|day)',
        r'(\d+)\s*(hours?|hrs?|ghante)',
        r'(\d+)\s*(weeks?|week|hafta)',
        r'(\d+)\s*(mahine|months?)'
    ]
    for pattern in duration_patterns:
        duration_match = re.search(pattern, msg)
        if duration_match:
            unit = duration_match.group(2)
            if 'hour' in unit or 'ghante' in unit:
                unit_display = "hours"
            elif 'din' in unit or 'day' in unit:
                unit_display = "days"
            elif 'week' in unit or 'hafta' in unit:
                unit_display = "weeks"
            elif 'mahine' in unit or 'month' in unit:
                unit_display = "months"
            else:
                unit_display = "days"
            updated['duration'] = f"{duration_match.group(1)} {unit_display}"
            break
    
    # Extract symptoms (append to list)
    symptoms_list = current_memory.get('symptoms', [])
    symptom_keywords = {
        'fever': ['fever', 'bukhar', 'temperature', 'tapish'],
        'cough': ['cough', 'khansi'],
        'cold': ['cold', 'zukam', 'sardi'],
        'headache': ['headache', 'sir dard', 'dard in head'],
        'bodyache': ['bodyache', 'body pain', 'badan dard', 'body ache'],
        'nausea': ['nausea', 'nausious', 'matli', 'ultas', 'vomiting', 'ulti'],
        'diarrhea': ['diarrhea', 'loose motion', 'dast'],
        'sorethroat': ['sore throat', 'throat pain', 'gale mein dard'],
        'weakness': ['weakness', 'kamzori', 'thakaan'],
        'stomach': ['stomach', 'pet dard', 'pet mein dard', 'abdomen', 'gas', 'acidity']
    }
    
    for symptom, keywords in symptom_keywords.items():
        for kw in keywords:
            if kw in msg:
                if symptom not in symptoms_list:
                    symptoms_list.append(symptom)
                updated['symptoms'] = symptoms_list
                break
    
    return updated

def format_followup_questions(qs):
    lines = qs.split("\n")
    clean = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # keep numbered questions and bullet points
        if re.match(r"^(\d+\.|\*)", line):
            clean.append(line)
    return "\n".join(clean)

# ============================================================
# STATIC RESPONSES & PATTERNS
# ============================================================
GREETING_PATTERN = re.compile(r'\b(hi|hello|hey|namaste|good\s*(morning|afternoon|evening)|pranam)\b', re.IGNORECASE)
INQUIRY_PATTERN = re.compile(r'\b(kaise\s*ho|kya\s*haal|aap\s*kaise|dadi\s*kaise|how\s*are\s*you)\b', re.IGNORECASE)
THANKS_PATTERN = re.compile(r'\b(thank|thanks|dhanyawad|shukriya|ty)\b', re.IGNORECASE)
FAREWELL_PATTERN = re.compile(r'\b(bye|goodbye|phir\s*milenge|ta\s*ta|tata|alvida)\b', re.IGNORECASE)

STATIC_GREETINGS = [
    "Beta, kya baat karni hai?",
    "Arre, kaise ho?",
    "Haan beta, batao kya chahiye?",
    "Theek ho na? Kuch problem hai toh batao."
]
STATIC_INQUIRY_RESPONSES = [
    "Main theek hoon beta, tum batao kaise ho?",
    "Dadi theek hai, tum apna batao. Koi problem?",
    "Arre main to theek hoon, tum batao kya dikkat hai?",
    "Sab badhiya, beta. Tum kaisa mahsoos kar rahe ho?"
]
STATIC_THANKS = [
    "Dhanyawad beta, khayal rakhna.",
    "Koi baat nahi, Dadi hoon na. Theek rehna.",
    "Apna khayal rakhna, beta. Phir milenge.",
    "Dadi ki dua hai, beta. Theek raho."
]
STATIC_FAREWELL = [
    "Accha beta, khayal rakhna. Phir milenge.",
    "Dadi ki dua hai saath mein. Theek rehna.",
]

# Minimal system prompt for casual chats (short to save tokens)
CASUAL_SYSTEM_PROMPT = (
    "You are Dadi, an 89-year-old Indian grandmother. "
    "Speak Hinglish (mix Hindi and English), be warm, reply in 1-2 short sentences. "
    "Never give medical advice."
)

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

import json
@app.route("/chat", methods=["POST"])
def chat():
    import json, requests, uuid, re, random

    data = request.get_json()
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"final": "Beta message nahi bheja"}), 400

    # ================= SESSION INIT =================
    profile = session.get("profile", {})
    history = session.get("history", [])
    full_history = session.get("full_history", [])
    followup_rounds = session.get("followup_rounds", 0)
    last_advice_given = session.get("last_advice_given", True)
    medical_memory = session.get("medical_memory", {})
    conversation_state = session.get("conversation_state", {"mode": "casual", "last_topic": None})

    # ================= SESSION ID =================
    incoming_session = data.get("session_id")

    # CRITICAL FIX: Always ensure we have a valid session_id
    if incoming_session and incoming_session != "undefined":
        session["session_id"] = incoming_session
    elif "session_id" not in session or session.get("session_id") == "undefined":
        session["session_id"] = uuid.uuid4().hex[:16]
    
    session_id = session.get("session_id")
    
    # Debug log
    logger.info(f"📌 Session ID: {session_id} (incoming: {incoming_session})")
    
    # ================= RESTORE SESSION FROM DB =================
    try:
        res = requests.get(
            f"https://biglive.com/API/dadi/get_chat.php?session_id={session_id}",
            timeout=5
        )
        if res.status_code == 200:
            db_data = res.json()
            db_history = json.loads(db_data.get("history_json", "[]"))
            
            # Always use DB if it has more data
            if len(db_history) > len(history):
                logger.info(f"🔄 DB has {len(db_history)} msgs, session has {len(history)} msgs - Restoring")
                
                full_history = db_history
                history = db_history[-12:]
                session["full_history"] = full_history
                session["history"] = history
                session["followup_rounds"] = db_data.get("followup_rounds", 0)
                followup_rounds = session["followup_rounds"]
                
                if db_data.get("age"):
                    profile["age"] = db_data.get("age")
                if db_data.get("sex"):
                    profile["sex"] = db_data.get("sex")
                if db_data.get("problem"):
                    profile["problem"] = db_data.get("problem")
                session["profile"] = profile
                
                try:
                    restored_memory = json.loads(db_data.get("medical_memory", "{}"))
                    if restored_memory:
                        medical_memory.update(restored_memory)
                        session["medical_memory"] = medical_memory
                except:
                    pass
            else:
                logger.info(f"📌 Session history up-to-date: {len(history)} msgs")
                
    except Exception as e:
        logger.error(f"Restore failed: {e}")

    # ================= PROFILE EXTRACTION =================
    new_data = extract_user_profile(user_message)
    for key in ["age", "sex"]:
        if new_data.get(key):
            profile[key] = new_data[key]
    session["profile"] = profile

    # ================= MEDICAL MEMORY EXTRACTION =================
    extracted_facts = extract_medical_facts(user_message, medical_memory)
    for key, value in extracted_facts.items():
        if key == 'symptoms':
            medical_memory['symptoms'] = value
        else:
            medical_memory[key] = value
    session["medical_memory"] = medical_memory

    # ================= HANDLE QUERIES ABOUT PAST INFO (ONLY WHEN ASKING) =================
    # Age query - ONLY when user ASKS for their age
    if re.search(r'(meri|mujhe|apni)\s*(umar|age)\s*kya\s*(hai|hain|batao)|(kitne\s*saal\s*ka\s*hu|kitne\s*saal\s*ki\s*hu)', user_message, re.IGNORECASE):
        if profile.get("age"):
            return jsonify({"final": f"Beta, tumhari umar {profile['age']} saal hai", "session_id": session_id})
        else:
            return jsonify({"final": "Beta, tumne abhi tak apni umar batayi nahi", "session_id": session_id})

    # Temperature query - ONLY when user ASKS about temperature
    if re.search(r'(temperature|temp|fever)\s*(kya\s*tha|kya\s*hai|kitna\s*tha|kitna\s*hai|batao)', user_message, re.IGNORECASE):
        temp = medical_memory.get("temperature")
        if temp:
            return jsonify({"final": f"Beta, tumhara temperature {temp}°F tha", "session_id": session_id})
        else:
            return jsonify({"final": "Beta, tumne abhi tak temperature bataya nahi. Kitna fever tha?", "session_id": session_id})

    # Duration query - ONLY when user ASKS about duration
    if re.search(r'(kitne\s*din\s*se|duration\s*kya\s*hai|kab\s*se\s*problem|kitne\s*din\s*hue|how\s*long)', user_message, re.IGNORECASE):
        duration = medical_memory.get("duration")
        if duration:
            return jsonify({"final": f"Beta, tumne bataya tha {duration} se problem hai", "session_id": session_id})
        else:
            return jsonify({"final": "Beta, tumne abhi tak bataya nahi ki kitne din se problem hai", "session_id": session_id})

    # ================= CLASSIFICATION =================
    classification = classifier.classify(user_message)
    is_medical = classification in ["medical", "emergency"]

    remedy_keywords = re.compile(
        r'\b(nuska|remedy|detail|batana|thoda|aur|elaborate|explain|more|kya|solution)\b',
        re.IGNORECASE
    )

    if (followup_rounds > 0 or not last_advice_given) and remedy_keywords.search(user_message):
        is_medical = True

    # ================= CONVERSATION STATE =================
    if is_medical:
        conversation_state["mode"] = "medical"
        conversation_state["last_topic"] = "health"
    elif not is_medical and followup_rounds == 0:
        conversation_state["mode"] = "casual"
    session["conversation_state"] = conversation_state

    # ================= NON-MEDICAL =================
    if not is_medical:
        if FAREWELL_PATTERN.search(user_message):
            reply = random.choice(STATIC_FAREWELL)
        elif THANKS_PATTERN.search(user_message) and last_advice_given:
            reply = random.choice(STATIC_THANKS)
            session["last_advice_given"] = False
        elif GREETING_PATTERN.search(user_message):
            reply = random.choice(STATIC_GREETINGS)
        else:
            short_context = [{"role": m["role"], "content": m["content"][:100]} for m in history[-4:]]
            messages = [
                {"role": "system", "content": CASUAL_SYSTEM_PROMPT},
                *short_context,
                {"role": "user", "content": user_message}
            ]
            try:
                print("Step 1", messages)
                result = call_groq_api(messages, temperature=0.7, max_tokens=50)
                reply = result["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.error(f"Casual AI failed: {e}")
                reply = random.choice(STATIC_GREETINGS)

        history.append({"role": "user", "content": user_message})
        full_history.append({"role": "user", "content": user_message})

        if not reply or not reply.strip():
            reply = "Thoda aur batao beta"

        history.append({"role": "assistant", "content": reply})
        full_history.append({"role": "assistant", "content": reply})

        session["history"] = history[-12:]
        session["full_history"] = full_history[-50:]

        return jsonify({
            "final": reply,
            "session_id": session_id
        })
    
    # ================= MEDICAL FLOW =================
    if not profile.get("problem"):
        profile["problem"] = user_message.lower()
    session["profile"] = profile

    history.append({"role": "user", "content": user_message})
    full_history.append({"role": "user", "content": user_message})

    # ================= BUILD CONTEXT HISTORY =================
    # CRITICAL FIX: Keep full assistant responses, don't strip them
    context_history = []
    for m in history[-12:]:
        content = m["content"]
        
        # Keep full context for assistant messages - don't strip XML
        if m["role"] == "assistant" and "<response>" in content:
            # Just truncate if too long, but preserve all XML tags
            if len(content) > 800:
                content = content[:800] + "..."
        elif m["role"] == "user" and len(content) > 500:
            content = content[:500] + "..."
        
        context_history.append({"role": m["role"], "content": content})

    # ================= BUILD PROMPT WITH MEDICAL MEMORY =================
    medical_memory_str = json.dumps(medical_memory, indent=2)
    last_questions = session.get("last_questions", "None")
    
    messages = [
        {"role": "system", "content": DADI_SYSTEM_PROMPT},
        {"role": "system", "content": f"""
Age: {profile.get('age', 'Unknown')}
Sex: {profile.get('sex', 'Unknown')}
Problem: {profile.get('problem', '')}
Followup rounds: {followup_rounds}
Current conversation mode: {conversation_state.get('mode', 'casual')}

MEDICAL MEMORY (past info user already shared):
{medical_memory_str}

Last questions asked to user:
{last_questions}

IMPORTANT: Use the MEDICAL MEMORY above to remember what user already told you. Do NOT ask for the same information again.
CRITICAL: Current followup_rounds = {followup_rounds}. You need MINIMUM 3 rounds before giving remedy.
"""}
    ] + context_history

    try:
        print("Step 2", messages)
        result = call_groq_api(messages)
        raw = result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"AI API failed: {e}")
        return jsonify({"final": "System busy hai beta, baad mein try karo", "session_id": session_id})

    cleaned = remove_thinking(raw).strip()
    cleaned = clean_language(cleaned)

    if "</response>" in cleaned:
        cleaned = cleaned.split("</response>")[0] + "</response>"

    parsed = parse_xml_response(cleaned)

    # ================= BUILD FULL RESPONSE WITH ENFORCED RULES =================
    assistant_content = cleaned
    
    MAX_FOLLOWUP = 3  # Minimum rounds before remedy
    
    # CHECK: If user is demanding remedy but rounds incomplete
    if "remedy" in user_message.lower() and followup_rounds < MAX_FOLLOWUP:
        # Force follow-up questions, ignore AI's remedy
        missing_info = []
        if not medical_memory.get("temperature") and "fever" in str(medical_memory.get("symptoms", [])):
            missing_info.append("temperature kitna hai")
        if not medical_memory.get("duration"):
            missing_info.append("kitne din se problem hai")
        if not profile.get("age"):
            missing_info.append("tumhari umar kya hai")
        
        if missing_info:
            reply = f"Beta, remedy dene se pehle yeh batao: {', '.join(missing_info)}"
        else:
            reply = "Beta, thoda aur batao - khana kya khaya? Neend achi aayi? Tabhi sahi nuska bataunga."
        
        # Don't increment followup_rounds here, let next round handle
        parsed["followup_questions"] = reply
        parsed["remedy"] = ""
    
    # Normal flow: check followup questions vs remedy
    elif parsed.get("followup_questions") and followup_rounds < MAX_FOLLOWUP:
        # Store the questions for future reference
        session["last_questions"] = parsed["followup_questions"]
        reply = format_followup_questions(parsed["followup_questions"])
        followup_rounds += 1
    else:
        # Clear stored questions when moving to remedy
        session["last_questions"] = ""
        
        # Build remedy response
        remedy_parts = []
        if parsed.get("remedy"):
            remedy_parts.append(f"Ghar ka Nuska:\n{parsed['remedy']}")
        if parsed.get("diet"):
            remedy_parts.append(f"Khana-Peena:\n{parsed['diet']}")
        if parsed.get("habit"):
            remedy_parts.append(f"Aadat Badlo:\n{parsed['habit']}")
        if parsed.get("final"):
            remedy_parts.append(parsed['final'])
        
        reply = "\n\n".join(remedy_parts) if remedy_parts else "Thoda aur batao beta"
        
        # Mark that advice has been given
        session["last_advice_given"] = True

    if not reply or not reply.strip():
        reply = "Thoda aur batao beta"

    # ================= APPEND TO SESSION =================
    history.append({"role": "assistant", "content": assistant_content})  # Store XML for context
    full_history.append({"role": "assistant", "content": assistant_content})

    session["history"] = history[-12:]
    session["full_history"] = full_history[-50:]
    session["followup_rounds"] = followup_rounds

    # ================= SAVE TO DB =================
    payload = {
        "session_id": session_id,
        "age": profile.get("age"),
        "sex": profile.get("sex") or "Unknown",
        "problem": profile.get("problem", ""),
        "followup_rounds": followup_rounds,
        "status": "active",
        "history_json": json.dumps(full_history, ensure_ascii=False),
        "medical_memory": json.dumps(medical_memory, ensure_ascii=False)
    }
    logger.info("=" * 60)
    logger.info("📤 SENDING TO DATABASE:")
    logger.info("=" * 60)
    logger.info(f"📌 Session ID: {session_id}")
    logger.info(f"👤 Age: {profile.get('age', 'Not provided')}")
    logger.info(f"⚥ Sex: {profile.get('sex', 'Not provided')}")
    logger.info(f"🏥 Problem: {profile.get('problem', 'Not provided')}")
    logger.info(f"🔄 Followup Rounds: {followup_rounds}")
    logger.info(f"📝 History Messages Count: {len(full_history)}")

    try:
        requests.post("https://biglive.com/API/dadi/insert_chat?", data=payload, timeout=5)
    except Exception as e:
        logger.error(f"DB failed: {e}")

    parsed["final"] = reply
    parsed["session_id"] = session_id
    return jsonify(parsed)

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    new_session_id = str(uuid.uuid4().hex[:16])
    session["session_id"] = new_session_id
    session.permanent = True  # ADD THIS LINE
    logger.info(f"✅ Session reset. New session_id: {new_session_id}")
    return jsonify({"status": "reset", "new_session_id": new_session_id})

@app.route("/get_history", methods=["GET"])
def get_history():
    """Return full chat history for frontend rendering with XML preserved"""
    session_id = session.get("session_id")

    if not session_id:
        import uuid
        session["session_id"] = uuid.uuid4().hex[:16]
        session_id = session["session_id"]
        return jsonify({"history": [], "session_id": session_id})

    try:
        res = requests.get(f"https://biglive.com/API/dadi/get_chat.php?session_id={session_id}", timeout=5)
        if res.status_code == 200:
            data = res.json()
            full_history = json.loads(data.get("history_json", "[]"))

            cleaned_history = []
            for msg in full_history:
                cleaned_history.append({
                    "role": msg.get("role"),
                    "content": msg.get("content", "")
                })
            return jsonify({"history": cleaned_history, "session_id": session_id})
    except Exception as e:
        logger.error(f"Failed to fetch full history from DB: {e}")

    history = session.get("history", [])
    cleaned_history = []
    for msg in history:
        cleaned_history.append({
            "role": msg.get("role"),
            "content": msg.get("content", "")
        })
    return jsonify({"history": cleaned_history, "session_id": session_id})

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=5000, debug=True)
