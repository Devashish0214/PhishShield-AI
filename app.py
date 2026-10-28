"""
PhishShield AI — Real-time Phishing & Smishing Detection Engine
Hackathon-ready FastAPI backend with TF-IDF + Logistic Regression + heuristics.

Runs fully locally on 127.0.0.1:8000. No cloud APIs. Messages never leave the machine.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ML: convert text → numbers (TF-IDF), then classify (Logistic Regression)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import re          # regex for URL / pattern extraction
import time        # measure analysis latency
import uuid        # unique request IDs
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from collections import deque  # fixed-size history buffer
import uvicorn     # ASGI server

# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------
# CORS allows the frontend (index.html / localhost:3000) to call this API
# from the browser without being blocked by same-origin policy.
app = FastAPI(
    title="PhishShield AI",
    description="Real-time phishing & smishing detection — Hackathon Edition",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # open for local demo; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory telemetry (demo only — resets when you stop the server)
# ---------------------------------------------------------------------------
# ANALYSIS_HISTORY: last N analyses (newest first)
# STATS: simple counters shown on /api/stats and used for health info
ANALYSIS_HISTORY: deque = deque(maxlen=100)
STATS = {
    "total_analyzed": 0,
    "high_risk": 0,
    "medium_risk": 0,
    "low_risk": 0,
    "started_at": datetime.now(timezone.utc).isoformat(),
}

# ---------------------------------------------------------------------------
# Synthetic training corpus
# Label 0 = safe, Label 1 = phishing / smishing
# In a real product you would train on a much larger labeled dataset.
# ---------------------------------------------------------------------------
SAFE_MESSAGES = [
    "Hi, your order #12345 has been shipped. Track it here: https://www.amazon.com/track/12345",
    "Meeting tomorrow at 10am in conference room B. Please bring the Q3 report.",
    "Your package is out for delivery today. Expected between 2-4pm.",
    "Happy birthday! Hope you have a wonderful day.",
    "The team lunch is confirmed for Friday at the Italian place on 5th Ave.",
    "Reminder: password reset link expires in 24 hours. Use the official portal only.",
    "Thanks for your payment. Receipt is attached to this email.",
    "Flight confirmation: AA123 departing 08:30. Check-in opens 24h before departure.",
    "Your Netflix subscription will renew on Nov 5. No action needed.",
    "Great work on the presentation yesterday! Looking forward to the next sprint.",
    "Your bank statement is ready in the secure online banking app.",
    "Doctor appointment confirmed for next Tuesday at 3pm. Reply STOP to cancel.",
    "The weather looks good for the weekend hike. See you at the trailhead.",
    "Please review the attached contract and sign via the official DocuSign link.",
    "Your Uber receipt for today's ride is available in the app under Trip History.",
    "Zoom invite: Product sync — Thursday 2pm. Join with company SSO.",
    "Your library book is due next Monday. Renew online or return to any branch.",
    "HR: Open enrollment ends Friday. Update benefits in Workday if needed.",
    "Your prescription is ready for pickup at CVS on Main Street.",
    "Weekly status report attached. Let me know if you have questions.",
]

PHISHING_MESSAGES = [
    "URGENT: Your account has been locked! Click here immediately to verify: http://secure-login-verify.ru/update",
    "You have won $10,000! Claim your prize now before it expires: http://prize-claim.tk/winner",
    "Security alert: Unusual login detected. Confirm your identity within 1 hour: http://bank-secure.xyz/login",
    "Your package could not be delivered. Pay $1.99 redelivery fee: http://dhl-redelivery.info/pay",
    "Final notice: Your Apple ID will be suspended. Verify now: http://appleid-verify.net",
    "IRS tax refund pending. Submit details here: http://irs-refund.online/claim",
    "Your crypto wallet needs verification. Act fast: http://wallet-secure.bit/verify",
    "Someone tried to access your email. Reset password immediately: http://outlook-security.ga",
    "Limited time offer! Claim free iPhone 15: http://free-iphone.win/claim-now",
    "Bank account compromised. Call this number or click: http://secure-bank-login.cc",
    "Your Netflix payment failed. Update card details: http://netflix-billing.support",
    "Urgent: Confirm your delivery address or package will be returned: http://ups-track.delivery",
    "You have 3 missed package attempts. Reschedule: http://fedex-reschedule.biz",
    "Account suspension notice – click to prevent permanent lock: http://account-recover.me",
    "Congratulations! You are selected for a $500 gift card. Claim: http://gift-card.claim.today",
    "We detected suspicious activity on your PayPal. Verify within 30 minutes: http://paypal-secure.tk",
    "Your Microsoft 365 license expires today. Renew now: http://ms-license-renew.xyz",
    "ATTENTION: Wire transfer pending approval. Confirm ASAP: http://wire-approve.info",
    "Your social security benefits require update. Click here: http://ssa-update.online",
    "Free Bitcoin giveaway! Send 0.01 BTC to receive 1 BTC back: http://btc-double.win",
]

# Build training arrays: texts + binary labels
X_train = SAFE_MESSAGES + PHISHING_MESSAGES
y_train = [0] * len(SAFE_MESSAGES) + [1] * len(PHISHING_MESSAGES)

# ---------------------------------------------------------------------------
# ML pipeline: TF-IDF vectorizer → Logistic Regression classifier
# Trained once at startup (fast on this small synthetic set)
# ---------------------------------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=4000,       # keep top 4000 n-grams
        ngram_range=(1, 2),      # unigrams + bigrams
        stop_words="english",
        lowercase=True,
        sublinear_tf=True,       # dampen very frequent terms
    )),
    ("clf", LogisticRegression(
        max_iter=1200,
        C=1.2,                   # inverse regularization strength
        class_weight="balanced", # handle any class imbalance
        random_state=42,
    )),
])

print("🛡️  Training PhishShield AI model...")
model.fit(X_train, y_train)
print("✅  Model ready. Serving on http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Heuristic signals (rule-based, complementary to ML)
# ---------------------------------------------------------------------------
# Words/phrases that create pressure to act quickly
URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "act now", "limited time", "expires",
    "final notice", "account locked", "suspended", "verify now", "click here",
    "confirm immediately", "within 24 hours", "last chance", "security alert",
    "unusual activity", "compromised", "claim now", "free prize", "winner",
    "act fast", "within 1 hour", "within 30 minutes", "permanent lock",
    "wire transfer", "gift card", "double your", "send bitcoin",
]

# TLDs / substrings often seen in throwaway phishing domains
SUSPICIOUS_TLDS = [
    ".ru", ".tk", ".xyz", ".top", ".info", ".cc", ".ga", ".ml", ".cf", ".gq",
    ".bit", ".win", ".online", ".support", ".biz", ".me", ".today", ".delivery",
    ".claim", ".verify", ".secure", ".login",
]

# Extra URL patterns: raw IPs, brand-looking hyphens, common shorteners
SUSPICIOUS_PATTERNS = [
    r"http://\d+\.\d+\.\d+\.\d+",                                    # raw IP
    r"https?://[^/]*-(login|secure|verify|account|bank|update)\.",  # brand impersonation
    r"bit\.ly|tinyurl|t\.co|goo\.gl",                                # shorteners (demo signal)
]

def extract_urls(text: str) -> List[str]:
    """Find http(s) and www. URLs in free text."""
    pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    return re.findall(pattern, text, re.IGNORECASE)


def count_urgency(text: str) -> int:
    """Count how many urgency keywords appear in the message."""
    lower = text.lower()
    return sum(1 for kw in URGENCY_KEYWORDS if kw in lower)


def analyze_urls(urls: List[str]) -> Dict[str, Any]:
    """
    Check each URL for suspicious TLDs, raw IPs, or known bad patterns.
    Returns { suspicious: bool, reasons: [str, ...] }.
    """
    suspicious = False
    reasons = []
    for url in urls:
        u = url.lower()
        if any(tld in u for tld in SUSPICIOUS_TLDS):
            suspicious = True
            reasons.append("suspicious TLD")
        if re.search(r"https?://\d+\.\d+\.\d+\.\d+", u):
            suspicious = True
            reasons.append("raw IP address")
        for pat in SUSPICIOUS_PATTERNS:
            if re.search(pat, u):
                suspicious = True
                reasons.append("suspicious URL pattern")
                break
    return {"suspicious": suspicious, "reasons": list(set(reasons))}


def calculate_threat_score(
    ml_prob: float,
    urgency: int,
    has_url: bool,
    url_info: Dict[str, Any],
) -> float:
    """
    Fuse ML probability with heuristic boosts into a 0–100 score.

    Weights (approx):
      - ML probability  → up to 65 points
      - Urgency keywords → up to 21 points
      - Any URL present  → +6
      - Suspicious URL   → +14
    """
    score = ml_prob * 65.0
    score += min(urgency * 7.0, 21.0)
    if has_url:
        score += 6.0
    if url_info.get("suspicious"):
        score += 14.0
    return min(round(score, 1), 100.0)


def verdict_from_score(score: float) -> str:
    """Map numeric score → human-readable verdict."""
    if score >= 70:
        return "HIGH RISK – Likely Phishing / Smishing"
    if score >= 40:
        return "MEDIUM RISK – Suspicious"
    return "LOW RISK – Appears Safe"


def advice_from_score(score: float) -> str:
    """Actionable guidance shown in the dashboard."""
    if score >= 70:
        return (
            "Do NOT click any links or share credentials. Report the message to your "
            "security team and delete it. Verify any claim only through official apps or known phone numbers."
        )
    if score >= 40:
        return (
            "Exercise caution. Hover over links (do not click). Contact the purported "
            "sender through a trusted channel you already know (official website or app)."
        )
    return (
        "Message appears legitimate. Still avoid clicking unexpected links and never "
        "share passwords or one-time codes via email/SMS."
    )

# ---------------------------------------------------------------------------
# Pydantic request / response schemas (API contract)
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class AnalyzeResponse(BaseModel):
    request_id: str
    verdict: str
    threat_score: float
    ml_probability: float
    red_flags: List[str]
    mitigation_advice: str
    features: Dict[str, Any]
    analysis_time_ms: float
    model_version: str = "2.0.0-tfidf-lr"
    timestamp: str

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Service info and available endpoints."""
    return {
        "service": "PhishShield AI",
        "version": "2.0.0",
        "status": "online",
        "endpoints": ["/api/analyze", "/api/health", "/api/stats", "/api/history"],
    }


@app.get("/api/health")
def health():
    """
    Lightweight health check used by the dashboard "Engine Online" indicator.
    """
    return {
        "status": "healthy",
        "model_loaded": True,
        "uptime_since": STATS["started_at"],
        "total_analyzed": STATS["total_analyzed"],
    }


@app.get("/api/stats")
def stats():
    """Aggregate counters for the current server process."""
    return {
        **STATS,
        "history_size": len(ANALYSIS_HISTORY),
    }


@app.get("/api/history")
def history(limit: int = 20):
    """Return the most recent analyses stored in memory."""
    limit = max(1, min(limit, 100))
    return {"items": list(ANALYSIS_HISTORY)[:limit]}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Main detection endpoint.

    1. Run ML model → phishing probability
    2. Extract URLs + urgency keywords
    3. Fuse into threat_score (0–100)
    4. Build red flags, verdict, advice
    5. Update in-memory stats/history
    6. Return structured JSON for the dashboard
    """
    t0 = time.perf_counter()
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content is required")

    # --- ML prediction (probability of class 1 = phishing) ---
    ml_prob = float(model.predict_proba([text])[0][1])

    # --- Heuristics ---
    urls = extract_urls(text)
    urgency = count_urgency(text)
    has_url = len(urls) > 0
    url_info = analyze_urls(urls)

    # --- Fuse score + labels ---
    threat_score = calculate_threat_score(ml_prob, urgency, has_url, url_info)
    verdict = verdict_from_score(threat_score)
    advice = advice_from_score(threat_score)

    # --- Human-readable red flags for the UI ---
    red_flags: List[str] = []
    if ml_prob >= 0.55:
        red_flags.append(f"ML model phishing confidence: {ml_prob * 100:.1f}%")
    if urgency > 0:
        red_flags.append(f"Urgency language detected ({urgency} keyword(s))")
    if has_url:
        red_flags.append(f"Contains {len(urls)} URL(s)")
    if url_info["suspicious"]:
        red_flags.append("Suspicious domain / TLD / pattern detected")
        for r in url_info["reasons"]:
            if r not in " ".join(red_flags).lower():
                red_flags.append(f"URL signal: {r}")
    if not red_flags:
        red_flags.append("No strong red flags detected")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    request_id = str(uuid.uuid4())[:8]
    ts = datetime.now(timezone.utc).isoformat()

    # --- Update process-wide stats ---
    STATS["total_analyzed"] += 1
    if threat_score >= 70:
        STATS["high_risk"] += 1
    elif threat_score >= 40:
        STATS["medium_risk"] += 1
    else:
        STATS["low_risk"] += 1

    # Full result (includes preview for server-side history only)
    result = {
        "request_id": request_id,
        "verdict": verdict,
        "threat_score": threat_score,
        "ml_probability": round(ml_prob, 4),
        "red_flags": red_flags,
        "mitigation_advice": advice,
        "features": {
            "url_count": len(urls),
            "urls": urls[:8],  # cap list size in response
            "urgency_keyword_count": urgency,
            "has_suspicious_tld": url_info["suspicious"],
            "url_reasons": url_info["reasons"],
        },
        "analysis_time_ms": elapsed_ms,
        "model_version": "2.0.0-tfidf-lr",
        "timestamp": ts,
        "preview": text[:120] + ("…" if len(text) > 120 else ""),
    }

    ANALYSIS_HISTORY.appendleft(result)

    # Response model excludes "preview" (kept only in server history)
    return AnalyzeResponse(**{k: v for k, v in result.items() if k != "preview"})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all so the API always returns JSON on unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal analysis error", "error": str(exc)[:200]},
    )

# ---------------------------------------------------------------------------
# Entrypoint — run: python app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # host 127.0.0.1 = local only; reload=True auto-restarts on code changes
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)