@app.route("/chat", methods=["POST"])
def chat():
    print("\n" + "🔥" * 20)
    print(f"NEW CHAT REQUEST RECEIVED")
    print("🔥" * 20 + "\n")
    
    import json, requests, re, random

    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()

    # =====================================================
    # ✅ FIX: Restore from DB FIRST, before initializing session vars
    # =====================================================
    session_id = data.get("session_id") or session.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4().hex[:16])

    session["session_id"] = session_id

    # ================= RESTORE FROM DB IMMEDIATELY =================
    profile = {}
    history = []
    full_history = []
    followup_rounds = 0
    
    try:
        res = requests.get(
            f"https://biglive.com/API/dadi/get_chat.php?session_id={session_id}",
            timeout=5
        )
        
        if res.status_code == 200:
            db = res.json()
            
            profile = {
                "age": db.get("age"),
                "sex": db.get("sex"),
                "problem": db.get("problem")
            }
            
            restored = json.loads(db.get("history_json", "[]"))
            history = restored[-10:] if restored else []
            full_history = restored if restored else []
            followup_rounds = db.get("followup_rounds", 0)
            
            # Update session with restored data
            session["profile"] = profile
            session["history"] = history
            session["full_history"] = full_history
            session["followup_rounds"] = followup_rounds
            
            print(f"🔵 RESTORED FROM DB: age={profile.get('age')}, problem={profile.get('problem')}, followup_rounds={followup_rounds}, history_count={len(full_history)}")
            
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        # Fallback to session data
        profile = session.get("profile", {})
        history = session.get("history", [])
        full_history = session.get("full_history", [])
        followup_rounds = session.get("followup_rounds", 0)

    if not user_message:
        return jsonify({"final": "Beta message nahi bheja"}), 400

    # ================= EXTRACT PROFILE FROM NEW MESSAGE =================
    new_data = extract_user_profile(user_message)
    profile.update({k: v for k, v in new_data.items() if v})
    
    # ================= CLASSIFY =================
    classification = classifier.classify(user_message)
    is_medical = classification in ["medical", "emergency"]

    if followup_rounds > 0:
        is_medical = True
    
    # CRITICAL: Update problem if not set
    if not profile.get("problem") and is_medical:
        profile["problem"] = user_message.lower()
    
    session["profile"] = profile

    # ================= ADD USER MESSAGE =================
    history.append({"role": "user", "content": user_message})
    full_history.append({"role": "user", "content": user_message})

    reply = ""
    parsed = {}

    # =====================================================
    # 1. EXIT FLOW
    # =====================================================
    EXIT_PATTERN = re.compile(r'\b(thank|thanks|bye|goodbye|ok|okay|theek hai|thik hai)\b', re.IGNORECASE)
    
    if EXIT_PATTERN.search(user_message):
        reply = random.choice(STATIC_THANKS + STATIC_FAREWELL)

    # =====================================================
    # 2. GREETING FLOW
    # =====================================================
    elif GREETING_PATTERN.search(user_message):
        reply = random.choice(STATIC_GREETINGS)

    # =====================================================
    # 3. INQUIRY FLOW
    # =====================================================
    elif INQUIRY_PATTERN.search(user_message):
        reply = random.choice(STATIC_INQUIRY_RESPONSES)

    # =====================================================
    # 4. THANKS / FAREWELL
    # =====================================================
    elif THANKS_PATTERN.search(user_message):
        reply = random.choice(STATIC_THANKS)

    elif FAREWELL_PATTERN.search(user_message):
        reply = random.choice(STATIC_FAREWELL)

    # =====================================================
    # 5. MEDICAL FLOW
    # =====================================================
    elif is_medical:

        if not profile.get("problem"):
            profile["problem"] = user_message.lower()

        session["profile"] = profile

        context = [
            {"role": m["role"], "content": m["content"][:300]}
            for m in history[-10:]
        ]

        messages = [
            {"role": "system", "content": DADI_SYSTEM_PROMPT},
            {"role": "system", "content": f"""
Age: {profile.get('age')}
Sex: {profile.get('sex')}
Problem: {profile.get('problem')}
Followup rounds: {followup_rounds}
"""}
        ] + context

        try:
            result = call_groq_api(messages)
            raw = result["choices"][0]["message"]["content"]

            cleaned = clean_language(remove_thinking(raw).strip())
            parsed = parse_xml_response(cleaned)

            # ================= FOLLOWUP LOGIC =================
            followups = parsed.get("followup_questions", "").strip()

            if followups and followup_rounds < 2:
                reply = format_followup_questions(followups)
                followup_rounds += 1
            else:
                reply = (
                    (parsed.get("final") or "") +
                    "\n" +
                    (parsed.get("remedy") or "")
                ).strip()

            if not reply.strip():
                reply = "Beta, mujhe thoda aur detail batao 😊"

        except Exception as e:
            print(f"❌ Medical flow error: {e}")
            reply = "System busy hai beta, baad mein try karo"

    # =====================================================
    # 6. CASUAL FLOW
    # =====================================================
    else:
        try:
            messages = [
                {"role": "system", "content": CASUAL_SYSTEM_PROMPT},
                *history[-4:],
                {"role": "user", "content": user_message}
            ]

            result = call_groq_api(messages, temperature=0.7, max_tokens=80)
            reply = result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"❌ Casual flow error: {e}")
            reply = random.choice(STATIC_GREETINGS)

        parsed = {"final": reply}

    # ================= SAVE HISTORY =================
    history.append({"role": "assistant", "content": reply})
    full_history.append({"role": "assistant", "content": reply})

    session["history"] = history[-10:]
    session["full_history"] = full_history[-50:]
    session["followup_rounds"] = followup_rounds

    # ================= SAVE DB =================
    print("=" * 50)
    print(f"🔵 SAVING TO DATABASE:")
    print(f"   session_id: {session_id}")
    print(f"   age: {profile.get('age')}")
    print(f"   sex: {profile.get('sex')}")
    print(f"   problem: {profile.get('problem', '')}")
    print(f"   followup_rounds: {followup_rounds}")
    print(f"   history messages: {len(full_history)}")
    if len(full_history) >= 2:
        print(f"   last user msg: {full_history[-2].get('content', '')[:50]}")
    print("=" * 50)

    payload = {
        "session_id": session_id,
        "age": profile.get("age"),
        "sex": profile.get("sex"),
        "problem": profile.get("problem", ""),
        "followup_rounds": followup_rounds,
        "history_json": json.dumps(full_history, ensure_ascii=False)
    }

    print(f"📤 Sending payload to DB")

    try:
        response = requests.post(
            "https://biglive.com/API/dadi/insert_chat.php",
            data=payload,
            timeout=5
        )
        print(f"📥 DB Response Status: {response.status_code}")
        print(f"📥 DB Response: {response.text}")
        
        if response.status_code == 200:
            print(f"✅ DATA SAVED SUCCESSFULLY!")
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ ERROR saving to DB: {e}")

    return jsonify({
        "session_id": session_id,
        "final": reply,
        "diagnosis": parsed.get("diagnosis", ""),
        "cause": parsed.get("cause", ""),
        "remedy": parsed.get("remedy", ""),
        "diet": parsed.get("diet", ""),
        "habit": parsed.get("habit", ""),
        "followup_questions": parsed.get("followup_questions", "")
    })
