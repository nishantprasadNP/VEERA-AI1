from dotenv import load_dotenv
load_dotenv()

import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import random
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

app = FastAPI()

# ================= MODELS =================

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

# ================= GLOBAL STATE =================

contexts = {}
sent_suppression_keys = set()
merchant_memory = {}

# ================= CONTEXT =================

def build_structured_context(category, merchant, trigger):
    merchant = merchant or {}
    category = category or {}
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    
    views = int(perf.get("views", 0))
    calls = int(perf.get("calls", 0))
    ctr = float(calls / views) if views > 0 else float(perf.get("ctr", 0.0))
    
    return {
        "merchant_name": merchant.get("identity", {}).get("name") or merchant.get("name"),
        "category": merchant.get("category_slug") or category.get("slug"),
        "views": views,
        "calls": calls,
        "ctr": ctr,
        "peer_ctr": float(category.get("peer_stats", {}).get("avg_ctr", 0.0)),
        "calls_pct": float(delta.get("calls_pct", 0.0)),
        "ctr_pct": float(delta.get("ctr_pct", 0.0)),
        "merchant_signals": merchant.get("signals", []),
        "offers": merchant.get("active_offers", []) or merchant.get("offers", []),
        "past_offers": [o for o in merchant.get("offers", []) if o.get("status") == "expired"],
        "plan": merchant.get("subscription", {}).get("plan", "Basic"),
        "lapsed_count": merchant.get("customer_aggregate", {}).get("lapsed_180d_plus", 0),
        "trigger_kind": trigger.get("kind", ""),
        "trigger_payload": trigger.get("payload", {})
    }

# ================= STEP 1: SIGNALS =================

def extract_signals(ctx):
    """
    Step 1: Extract deterministic signals from merchant context.
    """
    views = ctx.get("views", 0)
    calls = ctx.get("calls", 0)
    ctr = ctx.get("ctr", 0.0)
    
    return {
        "high_views_low_calls": views > 500 and calls < 5,
        "low_ctr": ctr < 0.02,
        "ctr_gap": max(0, ctx.get("peer_ctr", 0.0) - ctr),
        "perf_dip": ctx.get("calls_pct", 0) < -0.10,
        "ctr_drop": ctx.get("ctr_pct", 0) < 0,
        "stale_posts": any("stale_posts" in s for s in ctx.get("merchant_signals", [])),
        "recall_due": ctx.get("trigger_kind") == "recall_due",
        "lapsed_customers": ctx.get("lapsed_count", 0) > 10,
        "no_offer": len(ctx.get("offers", [])) == 0,
        "high_demand": views > 2000 and calls < 10,
        "digest": ctx.get("trigger_kind") == "research_digest"
    }

# ================= STEP 2: SCORING =================

def score_signals(signals):
    """
    Step 2: Calculate weighted scores per action type.
    """
    scores = {}
    
    # recall
    scores["recall"] = 10.0 if signals.get("recall_due") else 0.0
    
    # dip
    scores["dip"] = 8.0 if signals.get("perf_dip") else 0.0
    
    # conversion
    conv_score = 5.0 if signals.get("low_ctr") else 0.0
    conv_score += signals.get("ctr_gap", 0) * 60
    if signals.get("high_views_low_calls") and signals.get("stale_posts"):
        conv_score += 2.0
    scores["conversion"] = min(conv_score, 7.9)
    
    # other actions
    scores["demand"] = 4.0 if signals.get("high_demand") or signals.get("high_views_low_calls") else 0.0
    scores["digest"] = 2.0 if signals.get("digest") else 0.0
    scores["offer_gap"] = 5.0 if signals.get("no_offer") and signals.get("ctr_gap", 0) > 0.01 else 3.0 if signals.get("no_offer") else 0.0
    scores["engagement"] = 3.0 if signals.get("stale_posts") or signals.get("lapsed_customers") else 0.0

    return scores

# ================= STEP 3: DECISION =================

def decision_engine(signals, scores):
    """
    Step 3: Pick best signal based on score and priority overrides.
    """
    best_reason = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else "none"
    best_score = scores.get(best_reason, 0)

    # Priority override: recall ALWAYS wins
    if signals.get("recall_due"):
        best_reason = "recall"
        best_score = 100.0

    # Threshold
    if best_score < 5.0 and best_reason != "recall":
        return None

    # Explanation
    explanation = f"Signal '{best_reason}' was strongest with score {best_score:.1f}. "
    if best_reason == "recall": explanation += "Priority override triggered."
    elif signals.get("stale_posts"): explanation += "Boosted by engagement gap."

    return {
        "reason": best_reason,
        "score": best_score,
        "explanation": explanation
    }

# ================= STEP 4: ACTION =================

def decide_action(decision, ctx):
    """
    Step 4: Map decision to concrete action with refinement logic.
    """
    reason = decision["reason"]
    
    ACTIONS = {
        "recall": "recall_customer",
        "dip": "launch_offer",
        "conversion": "improve_conversion",
        "engagement": "boost_engagement",
        "offer_gap": "launch_offer",
        "demand": "capture_demand",
        "digest": "send_research"
    }
    
    base_action = ACTIONS.get(reason, "nudge")
    
    # Intelligence refinement
    if base_action == "launch_offer":
        if ctx.get("past_offers"):
            return "reuse_offer"
        if ctx.get("plan") == "Basic":
            return "small_offer"
            
    return base_action

# ================= STEP 5: MESSAGE GENERATION =================

def generate_message(ctx):
    """
    Step 5: Generate rule-based message using ctx["action"] and ctx["decision"].
    """
    name = ctx.get("merchant_name")
    category = ctx.get("category")
    views, calls = ctx.get("views", 0), ctx.get("calls", 0)
    ctr, peer_ctr = ctx.get("ctr", 0.0), ctx.get("peer_ctr", 0.0)
    
    action = ctx.get("action")
    decision = ctx.get("decision", {})
    reason = decision.get("reason")
    payload = ctx.get("trigger_payload", {})
    
    # [personalization]
    personalization = f"Dr. {extract_doctor_name(name)}" if category == "dentists" and name else (name or "Your business")

    offers = ctx.get("offers", [])
    offer_title = offers[0].get("title") if offers and isinstance(offers[0], dict) else (offers[0] if offers else "Free consultation")
    
    # [insight] & [action]
    if action == "recall_customer":
        service = payload.get("service_due", "visit").replace("_", " ")
        insight = f"your clients are due for their {service} but haven't booked yet."
        msg_action = f"I can send a personalized reminder to fill your calendar today."
    elif action == "launch_offer" and reason == "dip":
        delta = payload.get("delta_pct", 0)
        insight = f"your volume dropped by {abs(delta):.0%} recently despite steady views."
        msg_action = f"Relaunching your '{offer_title}' offer is the fastest way to recover."
    elif action == "improve_conversion":
        insight = f"received {views} views but only {calls} calls ({ctr:.1%} CTR vs {peer_ctr:.1%} avg)."
        msg_action = f"Adding a '{offer_title}' offer will help capture these lost customers."
    elif action == "reuse_offer":
        past_title = ctx["past_offers"][0].get("title", "previous offer")
        insight = f"noticed {views} views on your profile but bookings have slowed."
        msg_action = f"Your proven '{past_title}' offer worked before—should we relaunch it?"
    elif action == "small_offer":
        insight = f"with {views} views and only {calls} calls, you're missing bookings."
        msg_action = f"A low-cost '10% OFF' offer is a zero-risk way to convert that traffic."
    else:
        insight = f"is getting {views} views but call volume could be higher."
        msg_action = f"Should we try adding a '{offer_title}' offer to boost your results?"

    # [CTA]
    cta = "Should I proceed? Reply YES."

    return {
        "body": f"{personalization}, {insight} {msg_action} {cta}",
        "cta": "YES/STOP"
    }

# ================= LLM =================

def refine_with_llm(message, ctx):
    print("🔥 LLM FUNCTION ENTERED")

    api_key = os.getenv("OPENAI_API_KEY")
    print("API KEY FOUND:", bool(api_key))

    if not api_key or OpenAI is None:
        return message

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Improve business messages."},
                {"role": "user", "content": message}
            ],
            timeout=8
        )

        refined = response.choices[0].message.content.strip()
        print("🧠 LLM RESPONSE:", refined)

        return refined if len(refined.split()) > 8 else message

    except Exception as e:
        print("LLM ERROR:", e)
        return message

# ================= API =================

@app.post("/v1/context")
async def add_context(data: ContextInput):
    contexts[(data.scope, data.context_id)] = {"version": data.version, "payload": data.payload}
    return {"accepted": True}

@app.post("/v1/tick")
async def tick(data: TickInput):
    """
    Main pipeline: context -> signals -> scoring -> decision -> action -> message -> LLM refine.
    """
    best_candidate = None
    max_selection_score = -1

    for tid in data.available_triggers:
        trigger_ctx = contexts.get(("trigger", tid))
        if not trigger_ctx: continue
        trigger = trigger_ctx["payload"]
        
        # Skip if action recently sent (anti-spam)
        suppkey = trigger.get("suppression_key")
        if suppkey and suppkey in sent_suppression_keys:
            continue

        merchant = contexts.get(("merchant", trigger["merchant_id"]), {}).get("payload", {})
        category = contexts.get(("category", merchant.get("category_slug")), {}).get("payload", {})

        # --- PIPELINE START ---
        ctx = build_structured_context(category, merchant, trigger)
        
        signals = extract_signals(ctx)
        scores = score_signals(signals)
        decision = decision_engine(signals, scores)
        
        if not decision: continue
        
        action = decide_action(decision, ctx)
        
        # --- SELECTION ---
        if decision["score"] > max_selection_score:
            max_selection_score = decision["score"]
            best_candidate = {
                "tid": tid,
                "trigger": trigger,
                "ctx": ctx,
                "signals": signals,
                "scores": scores,
                "decision": decision,
                "action": action
            }

    if not best_candidate:
        return {"actions": []}

    # Finalize winning candidate
    c = best_candidate
    if c["trigger"].get("suppression_key"):
        sent_suppression_keys.add(c["trigger"]["suppression_key"])

    # Update context with pipeline state
    c["ctx"]["signals"] = c["signals"]
    c["ctx"]["scores"] = c["scores"]
    c["ctx"]["decision"] = c["decision"]
    c["ctx"]["action"] = c["action"]

    # --- DEBUG LOGS ---
    print(f"\n[PIPELINE DEBUG] Trigger: {c['tid']}")
    print(f"  SIGNALS: {json.dumps(c['signals'])}")
    print(f"  SCORES: {json.dumps(c['scores'])}")
    print(f"  DECISION: {c['decision']['reason']} ({c['decision']['score']})")
    print(f"  ACTION: {c['action']}")

    # --- MESSAGE GENERATION ---
    msg = generate_message(c["ctx"])
    final_message = refine_with_llm(msg["body"], c["ctx"])
    
    print(f"  FINAL: {final_message}")
    print("-" * 50)
    
    # --- RESPONSE ---
    return {
        "actions": [{
            "conversation_id": f"{c['trigger']['merchant_id']}_{c['tid']}",
            "merchant_id": c['trigger']["merchant_id"],
            "trigger_id": c['tid'],
            "body": final_message,
            "cta": msg["cta"],
            "action_type": c['action'],
            "rationale": c['decision']["explanation"]
        }]
    }