import time
import os
from datetime import datetime, timezone
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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"DEBUG: Validation Error: {exc.errors()}")
    print(f"DEBUG: Request Body: {await request.body()}")
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

START_TIME = time.monotonic()

contexts = {}
conversations = {}
sent_suppression_keys = set()
merchant_memory = {}

# ---------------- HELPERS ---------------- #

def clean_doctor_name(name: str) -> str:
    if not name:
        return "there"
    for suffix in ["Dental Clinic", "Clinic", "Hospital", "Center", "Centre"]:
        name = name.replace(suffix, "")
    return name.strip()

def extract_doctor_name(name):
    """Helper to extract a doctor's name from a clinic name."""
    return clean_doctor_name(name)

# ---------------- SIGNAL ENGINE ---------------- #

def extract_signals(ctx):
    merchant = ctx["merchant"]
    trigger = ctx["trigger"]

    perf = merchant.get("performance", {})
    views = perf.get("views", 0)
    calls = perf.get("calls", 0)
    ctr = perf.get("ctr", 0)

    return {
        "high_views_low_calls": views > 100 and calls < 5,
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

    if merchant_memory.get(merchant_id) == action:
        return None

    return {
        "action": action,
        "reason": best_signal,
        "score": score
    }

# ---------------- STRATEGY ENGINE ---------------- #

def build_strategy(ctx, action, reason):
    merchant = ctx["merchant"]
    category = ctx["category"]

    perf = merchant.get("performance", {})
    ctr = perf.get("ctr", 0)
    views = perf.get("views", 0)
    calls = perf.get("calls", 0)

    peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0)

    offers = merchant.get("offers", [])
    offer = offers[0]["title"] if offers else None

    strategy = {
        "merchant_name": merchant["identity"]["name"],
        "category": category["slug"],
        "views": views,
        "calls": calls,
        "ctr": ctr,
        "peer_ctr": peer_ctr,
        "ctr_gap": max(0, peer_ctr - ctr),
        "offer": offer,
        "locality": merchant["identity"].get("locality", ""),
        "action": action,
        "reason": reason,
        "goal": "",
        "lever": ""
    }

    if action == "launch_offer":
        strategy["goal"] = "increase conversions"
    elif action == "recall_customer":
        strategy["goal"] = "increase repeat visits"
    elif action == "improve_listing":
        strategy["goal"] = "improve CTR"

    return strategy

# ---------------- CTA ---------------- #

def generate_cta(action):
    if action == "launch_offer":
        return "Reply YES to launch this"
    elif action == "recall_customer":
        return "Reply YES to send reminders"
    return "Want me to do this? Reply YES"

# ---------------- LLM ---------------- #

def generate_message_with_llm(ctx, strategy, action):
    api_key = os.getenv("OPENAI_API_KEY")

    name = clean_doctor_name(strategy["merchant_name"])
    if strategy["category"] == "dentists":
        name = f"Dr. {name}"

    cta = generate_cta(action)

    fallback = (
        f"{name}, you got {strategy['views']} views but only {strategy['calls']} calls. "
        f"At {strategy['ctr']:.1%} CTR vs {strategy['peer_ctr']:.1%}, you're losing customers. "
        f"Add an offer to boost conversions. {cta}"
    )

    if not api_key or OpenAI is None:
        return fallback

    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
Write ONE WhatsApp message.

Include:
- {strategy['views']} views, {strategy['calls']} calls
- CTR {strategy['ctr']:.2%} vs {strategy['peer_ctr']:.2%}
- Loss framing
- Action: {strategy['action']}
- Goal: {strategy['goal']}
- Name: {name}

Max 3 sentences. End with CTA: {cta}
"""

        print("LLM INPUT:", prompt)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        msg = res.choices[0].message.content.strip()
        print("LLM OUTPUT:", msg)

        return msg or fallback

    except Exception as e:
        print("LLM ERROR:", e)
        return fallback

# ---------------- API ---------------- #

@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Antigravity",
        "model": "gpt-4o-mini",
        "version": "1.1.0"
    }

@app.post("/v1/context")
async def context_api(data: ContextInput):
    contexts[(data.scope, data.context_id)] = data.payload
    return {"ok": True}

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
        print("SIGNALS:", signals)

        best_signal, scores = score_signals(signals)
        print("SCORES:", scores)

        decision = decide_action(best_signal, scores, merchant_id)
        print("DECISION:", decision)

        if not decision:
            continue

        strategy = build_strategy(ctx, decision["action"], decision["reason"])
        print("STRATEGY:", strategy)

        message = generate_message_with_llm(ctx, strategy, decision["action"])
        print("FINAL MESSAGE:", message)

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
        return {"action": "send", "body": "Done — setting it up now."}
    if "stop" in msg:
        return {"action": "end"}

    return {"action": "send", "body": "Got it — tell me more."}