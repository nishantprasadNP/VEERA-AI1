import time
import os
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from openai import OpenAI
except:
    OpenAI = None

app = FastAPI()

@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "YourTeamName",
        "model": "gpt-4o-mini",
        "version": "1.0.0"
    }

# ---------------- ERROR HANDLER ---------------- #

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(await request.body())},
    )

# ---------------- MODELS ---------------- #

class ContextInput(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None

class TickInput(BaseModel):
    now: str
    available_triggers: List[str]

class ReplyInput(BaseModel):
    conversation_id: str
    merchant_id: str
    message: str
    turn_number: int

# ---------------- GLOBALS ---------------- #

contexts = {}
merchant_memory = {}

# ---------------- HELPERS ---------------- #

def clean_doctor_name(name: str) -> str:
    if not name:
        return "there"
    for suffix in ["Dental Clinic", "Clinic", "Hospital", "Center", "Centre"]:
        name = name.replace(suffix, "")
    return name.strip()

# ---------------- STAGE ENGINE ---------------- #

def get_stage(merchant_id):
    return merchant_memory.get(f"{merchant_id}_stage", 0)

def update_stage(merchant_id):
    stage = get_stage(merchant_id) + 1
    merchant_memory[f"{merchant_id}_stage"] = stage
    return stage

# ---------------- SIGNAL ENGINE ---------------- #

def extract_signals(ctx):
    merchant = ctx["merchant"]
    trigger = ctx["trigger"]

    perf = merchant.get("performance", {})
    views = perf.get("views", 0)
    calls = perf.get("calls", 0)
    ctr = perf.get("ctr", 0)

    return {
        "low_ctr": ctr < 0.02,
        "ctr_gap": max(0, 0.03 - ctr),
        "recall_due": trigger.get("kind") == "recall_due",
        "perf_dip": trigger.get("kind") == "perf_dip",
    }

def score_signals(signals):
    scores = {
        "recall": 10 if signals["recall_due"] else 0,
        "dip": 8 if signals["perf_dip"] else 0,
        "ctr": 6 + signals["ctr_gap"] * 10 if signals["low_ctr"] else 0,
    }
    best = max(scores, key=scores.get)
    return best, scores

# ---------------- DECISION ENGINE ---------------- #

ACTIONS = {
    "recall": "recall_customer",
    "dip": "launch_offer",
    "ctr": "improve_listing"
}

def decide_action(best_signal, scores, merchant_id):
    score = scores[best_signal]
    if score < 5:
        return None

    action = ACTIONS[best_signal]
    last = merchant_memory.get(merchant_id)

    # allow repeat but downgrade slightly
    if last == action:
        return {
            "action": action,
            "reason": "repeat_escalation",
            "score": score - 1
        }

    return {
        "action": action,
        "reason": best_signal,
        "score": score
    }

# ---------------- STRATEGY ENGINE ---------------- #

def build_strategy(ctx, action):
    merchant = ctx["merchant"]
    category = ctx["category"]

    perf = merchant.get("performance", {})
    ctr = perf.get("ctr", 0)
    views = perf.get("views", 0)
    calls = perf.get("calls", 0)

    peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0)

    strategy = {
        "merchant_name": merchant["identity"]["name"],
        "category": category["slug"],
        "views": views,
        "calls": calls,
        "ctr": ctr,
        "peer_ctr": peer_ctr,
        "ctr_gap": max(0, peer_ctr - ctr),
        "locality": merchant["identity"].get("locality", ""),
        "action": action,
        "goal": "",
        "lever": "",
        "loss": int(views * max(0, peer_ctr - ctr))
    }

    if action == "launch_offer":
        strategy["goal"] = "increase conversions"
        strategy["lever"] = "adding a limited-time first visit offer"

    elif action == "recall_customer":
        strategy["goal"] = "increase repeat visits"
        strategy["lever"] = "sending reminders to past customers"

    elif action == "improve_listing":
        strategy["goal"] = "improve CTR"
        strategy["lever"] = "fixing listing headline, photos, and highlights"

    return strategy

# ---------------- CTA ---------------- #

def generate_cta(action):
    if action == "launch_offer":
        return "Reply YES to launch this"
    elif action == "recall_customer":
        return "Reply YES to send reminders"
    return "Reply YES and I’ll fix this"

# ---------------- LLM ---------------- #

def generate_message_with_llm(strategy, stage):
    api_key = os.getenv("OPENAI_API_KEY")

    name = clean_doctor_name(strategy["merchant_name"])
    if strategy["category"] == "dentists":
        name = f"Dr. {name}"

    cta = generate_cta(strategy["action"])

    # 🔥 STRONG FALLBACK
    fallback = (
        f"{name}, you're getting {strategy['views']} views but only {strategy['calls']} calls — "
        f"that's ~{strategy['loss']} potential customers lost. "
        f"Nearby businesses are at {strategy['peer_ctr']:.1%} CTR vs your {strategy['ctr']:.1%}. "
        f"I can fix this by {strategy['lever']}. {cta}"
    )

    if not api_key or OpenAI is None:
        return fallback

    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
You are an AI growth assistant helping local businesses get more customers.

Your task is to write ONE WhatsApp message that is extremely specific, data-driven, and persuasive.

---

## ⚠️ HARD RULES (STRICT — MUST FOLLOW)

* Maximum 2–3 sentences ONLY
* No greetings (no "Hi", "Hello")
* No fluff or filler
* Do NOT use:
  * apostrophes (')
  * quotation marks
  * emojis
* Only plain text, clean formatting
* Every sentence must contain useful information

If any rule is broken → rewrite before returning.

---

## 📊 CONTEXT

Name: {name}
Category: {strategy['category']}
Locality: {strategy['locality']}

Views: {strategy['views']}
Calls: {strategy['calls']}
CTR: {strategy['ctr']:.2%}
Peer CTR: {strategy['peer_ctr']:.2%}
CTR Gap: {strategy['ctr_gap']:.2%}
Estimated Lost Customers: {strategy['loss']}

Action: {strategy['action']}
Specific lever: {strategy['lever']}

Stage: {stage}

---

## 🧠 CATEGORY LANGUAGE (MANDATORY)

* If dentist → use "patients", "appointments"
* If restaurant → use "orders", "diners"
* Otherwise → use "customers"

---

## 🧠 REQUIRED ELEMENTS (ALL MUST BE PRESENT)

1. LOSS FRAMING
   * Must include a NUMBER
   * Example: "losing ~25 patients"

2. SOCIAL PROOF
   * Compare with nearby businesses
   * Example: "others nearby are at 3.2% CTR"

3. ROOT CAUSE (IMPORTANT)
   * Give a likely reason
   * Example: "likely due to missing first visit offers"

4. SPECIFIC ACTION
   * Must use the given lever
   * Example: "adding a first time offer"

5. EFFORT REMOVAL
   * Must include: "I can do this" or similar

6. CURIOSITY HOOK
   * Must trigger reply
   * Example: "want me to fix this?"

If ANY element is missing → rewrite.

---

## 🔁 STAGE BEHAVIOR

Stage 1:
* Focus on insight and gap

Stage 2:
* Add urgency + competitor advantage

Stage 3:
* Strong push + consequence of inaction

---

## 📏 OUTPUT STRUCTURE (STRICT)

Sentence 1:
* Views + Calls + Lost customers

Sentence 2:
* Competitor comparison + cause

Sentence 3:
* Exact action + effort removal + CTA

---

## 🎯 STYLE

* Direct, sharp, slightly urgent
* Use numbers wherever possible
* Sounds like insider business insight
* Not marketing language

---

## ✅ FINAL OUTPUT

Return ONLY the message text.

End with CTA:
{cta}
"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return res.choices[0].message.content.strip() or fallback

    except:
        return fallback

# ---------------- API ---------------- #

@app.post("/v1/context")
async def context_api(data: ContextInput):
    contexts[(data.scope, data.context_id)] = data.payload
    return {"accepted": True}   # ← was {"ok": True}

@app.post("/v1/tick")
async def tick(data: TickInput):
    actions = []

    for trg_id in data.available_triggers:
        trigger = contexts.get(("trigger", trg_id))
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        merchant = contexts.get(("merchant", merchant_id))
        category = contexts.get(("category", merchant.get("category_slug")))

        ctx = {
            "trigger": trigger,
            "merchant": merchant,
            "category": category
        }

        signals = extract_signals(ctx)
        best_signal, scores = score_signals(signals)

        decision = decide_action(best_signal, scores, merchant_id)
        if not decision:
            continue

        stage = update_stage(merchant_id)

        strategy = build_strategy(ctx, decision["action"])

        message = generate_message_with_llm(strategy, stage)

        actions.append({
            "conversation_id": f"{merchant_id}_{trg_id}",
            "merchant_id": merchant_id,
            "send_as": "vera",
            "trigger_id": trg_id,
            "body": message,
            "cta": "YES/STOP"
        })

        merchant_memory[merchant_id] = decision["action"]

    return {"actions": actions}

@app.post("/v1/reply")
async def reply(data: ReplyInput):
    msg = data.message.lower()

    if "yes" in msg:
        return {
            "action": "send",
            "body": "Got it — setting this up now. You should start seeing results soon."
        }

    if "stop" in msg or "no" in msg:
        return {"action": "end"}

    if data.turn_number > 2:
        return {"action": "end"}

    return {
        "action": "send",
        "body": "I can show you exactly what’s causing the drop — want to see?"
    }