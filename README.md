# PhishShield AI — Hackathon SOC Console

**Real-time phishing & smishing detection** with a cybersecurity operations dashboard.

Built to win: polished dark SOC UI, live ML scoring, heuristics, session history, explainability, and one-click report download.

---

## Stack

| Layer | Tech |
|-------|------|
| Backend | Python 3.10+, FastAPI, scikit-learn (TF-IDF + Logistic Regression) |
| Frontend | Single-page dashboard — Tailwind CSS + Chart.js (CDN, no build step) |
| Runtime | Local only — `127.0.0.1:8000` (API) + browser or static server |

---

## Features that win demos

- **ML + heuristics fusion** — TF-IDF + Logistic Regression trained on synthetic safe/phishing messages, combined with urgency keywords and suspicious URL/TLD signals into a 0–100 threat score
- **SOC-style dashboard** — dark glass UI, animated scan line, doughnut threat meter, live status
- **Explainability** — “Why this score?” panel breaks down ML probability, urgency, and URL signals
- **Session history** — last 50 analyses stored in the browser; click any row to reload the result
- **Live stats strip** — total / high / medium / low risk counts
- **One-click export** — download last analysis or full session report as JSON
- **Quick samples** — high-risk, safe, and suspicious messages for instant demos
- **Keyboard** — `Ctrl+Enter` to analyze
- **Health probe** — engine Online/Offline indicator (polls `/api/health`)

---

## Quick start (Windows)

### 1. Install dependencies (once)

```cmd
cd C:\Users\devas\OneDrive\Desktop\PhishShield-AI
pip install -r requirements.txt
```

Or:

```cmd
pip install fastapi uvicorn scikit-learn pydantic numpy
```

### 2. Start the backend (keep this terminal open)

```cmd
python app.py
```

You should see:

```
🛡️  Training PhishShield AI model...
✅  Model ready. Serving on http://127.0.0.1:8000
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 3. Open the dashboard

**Option A** — double-click `index.html`

**Option B** — second terminal:

```cmd
python -m http.server 3000
```

Then open **http://localhost:3000**

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Body: `{ "text": "..." }` → verdict, score, flags, advice |
| `GET`  | `/api/health`  | Model + uptime status |
| `GET`  | `/api/stats`   | Aggregate counters |
| `GET`  | `/api/history` | Recent server-side analyses |

---

## Demo script (judges love this)

1. Click **High-risk phishing** sample → Analyze → score jumps to HIGH, red glow, toast alert  
2. Click **Safe shipping notice** → LOW RISK, green  
3. Paste a custom message → show “Why this score?”  
4. Hit **Export Report** → download JSON for the judges  

---

## Project layout

```
PhishShield-AI/
├── app.py              # FastAPI + ML model
├── index.html          # Interactive SOC dashboard
├── requirements.txt
└── README.md
```

---

## Notes

- Model trains on startup (synthetic dataset). No external API keys required.
- History is client-side (`localStorage`) so it survives refresh.
- Server history/stats reset when you restart `app.py`.
- Designed for **local demo only** — do not expose to the public internet without auth.

Good luck — go win the hackathon. 🛡️
