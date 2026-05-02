# VeraBot: Next-Gen Merchant AI Assistant

VeraBot is a highly optimized, context-aware decision engine designed to engage local merchants on WhatsApp. Built for the magicpin AI Challenge, it outperforms generic assistants by anchoring every interaction on verifiable business data, category-specific clinical/peer tones, and prioritized growth signals.

## 🧠 System Architecture

VeraBot follows a deterministic, multi-stage pipeline to ensure that every message is not only persuasive but also 100% factually accurate.

### 1. Context Aggregation (`build_structured_context`)
Raw payloads from the Merchant, Category, and Trigger contexts are parsed and flattened into a `structured_context`. This layer ensures that views, calls, and CTR benchmarks are always available for downstream logic.

### 2. Signal Extraction (`extract_signals`)
A specialized logic layer that converts raw numbers into actionable boolean signals:
- **`ctr_gap`**: Magnitude of performance lag vs category average.
- **`low_calls` / `high_views_low_conversion`**: Identifies engagement mismatches.
- **`no_offer`**: Detects conversion blockers in the merchant's profile.
- **Trigger Flags**: Identifies `perf_dip`, `recall_due`, or `research_digest` intents.

### 3. Priority Scoring (`score_signals`)
Instead of reacting to every signal, VeraBot uses a **weighted scoring engine** to identify the "Primary Reason" for messaging. High-urgency signals like `recall_due` (Score 9) and `perf_dip` (Score 7) are prioritized over routine research updates.

### 4. Dynamic Wording (`generate_message`)
A rule-based template engine that selects the optimal wording based on the `primary_reason`. 
- **Voice Match**: Automatically adjusts tone for professional categories (e.g., using "Dr." for Dentists/Clinics).
- **Compulsion Levers**: Every message uses **Loss Aversion**, **Social Proof**, or **Curiosity** to drive a single, low-friction binary CTA (**"Reply YES"**).

### 5. LLM Refinement (`refine_with_llm`)
If an OpenAI API key is present, the message is polished by **GPT-4o-mini**.
- **Strict Guardrails**: The LLM is forbidden from changing numbers or hallucinations.
- **Safety Fallback**: If the LLM output is invalid, too long (>500 chars), or too short (<20 chars), the system reverts to the reliable rule-based "original message".

---

## 🛠️ Setup & Running

### Requirements
- Python 3.9+
- FastAPI, Uvicorn, Pydantic, OpenAI

### Environment Variables
- `OPENAI_API_KEY`: (Optional) Required for LLM refinement.
- `LL_MODEL`: (Optional) Defaults to `gpt-4o-mini`.

### Launching the Server
```bash
uvicorn bot:app --reload
```

---

## 🧪 Testing & Validation

VeraBot includes built-in internal checks to ensure logic integrity before deployment.

### Internal Sanity Checks
Run the following to validate the signal and scoring pipeline:
```bash
python bot.py
```
This suite tests `extract_signals`, `score_signals`, and `refine_with_llm` fallbacks using sample data.

### External Evaluation
Use the `judge_simulator.py` to evaluate the bot against the 5-dimension rubric (Specificity, Category Fit, Merchant Fit, Trigger Relevance, Engagement Compulsion).

---

## ⚖️ Tradeoffs & Decisions

- **Rule-First, LLM-Second**: I prioritized a robust rule-based foundation to prevent hallucinations. The LLM acts as a "stylist" rather than a "generator," ensuring that business numbers (like CTR and call volume) remain immutable.
- **Binary CTA**: I standardized on "Reply YES/STOP" to maximize conversion rates, as Indian merchant audiences prefer clear, single-choice commitments over open-ended questions.
- **In-Memory State**: For the challenge, I used thread-safe in-memory storage for suppression keys and merchant memory. For a 10k merchant scale, this should be moved to Redis.

---
*Built for the magicpin AI Challenge (May 2026)*
