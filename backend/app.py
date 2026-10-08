"""
OS Study Assistant — Gemini Proxy Backend
==========================================
Keeps the API key in .env (never sent to the browser).
Exposes two endpoints:
  POST /api/stream   -> proxies Gemini SSE streaming (chat Q&A)
  POST /api/generate -> proxies Gemini non-streaming (JSON for quiz / expected Q&A)
"""

import os
import time
import json
import requests
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
# .env is in the same folder as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL  = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
GEMINI_BASE     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
FALLBACK_BASE   = f"https://generativelanguage.googleapis.com/v1beta/models/{FALLBACK_MODEL}"

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from backend/.env")

app = Flask(__name__)
CORS(app)   # Allow the frontend (port 8080) to call us (port 5000)

RETRY_CODES  = {429, 500, 502, 503, 504}
MAX_RETRIES  = 3
RETRY_DELAY  = 2   # seconds between retries


# ── Helper: build the Gemini request body ─────────────────────────────────────
def gemini_body(prompt: str) -> dict:
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        },
    }

def api_headers() -> dict:
    return {
        "Content-Type":   "application/json",
        "X-goog-api-key": GEMINI_KEY,
    }


# ── /api/stream — SSE streaming proxy for live chat answers ───────────────────
@app.route("/api/stream", methods=["POST"])
def stream():
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "")

    def generate():
        # Try primary model, fall back if unavailable
        bases_to_try = [GEMINI_BASE, FALLBACK_BASE]
        for attempt, base in enumerate(bases_to_try):
            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    with requests.post(
                        base + ":streamGenerateContent?alt=sse",
                        headers=api_headers(),
                        json=gemini_body(prompt),
                        stream=True,
                        timeout=60,
                    ) as resp:
                        if resp.status_code in RETRY_CODES and retries < MAX_RETRIES:
                            retries += 1
                            time.sleep(RETRY_DELAY * retries)
                            continue
                        if not resp.ok:
                            if attempt == 0:
                                break   # try fallback model
                            error_text = resp.text
                            yield f"data: {json.dumps({'error': error_text})}\n\n"
                            return
                        for raw_line in resp.iter_lines():
                            if raw_line:
                                decoded = (
                                    raw_line.decode("utf-8")
                                    if isinstance(raw_line, bytes)
                                    else raw_line
                                )
                                yield decoded + "\n"
                        return  # success
                except Exception as exc:
                    if retries < MAX_RETRIES:
                        retries += 1
                        time.sleep(RETRY_DELAY * retries)
                    else:
                        if attempt == 0:
                            break   # try fallback
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                        return
                retries += 1
        yield f"data: {json.dumps({'error': 'All models busy. Please try again in a moment.'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /api/generate — single-shot for JSON responses (quiz, Q&A) ────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "")

    bases_to_try = [GEMINI_BASE, FALLBACK_BASE]
    for attempt, base in enumerate(bases_to_try):
        retries = 0
        while retries <= MAX_RETRIES:
            try:
                resp = requests.post(
                    base + ":generateContent",
                    headers=api_headers(),
                    json=gemini_body(prompt),
                    timeout=60,
                )
                if resp.status_code in RETRY_CODES and retries < MAX_RETRIES:
                    retries += 1
                    time.sleep(RETRY_DELAY * retries)
                    continue
                if not resp.ok:
                    if attempt == 0:
                        break   # try fallback model
                    return jsonify({"error": resp.text}), resp.status_code
                return jsonify(resp.json())
            except Exception as exc:
                if retries < MAX_RETRIES:
                    retries += 1
                    time.sleep(RETRY_DELAY * retries)
                else:
                    if attempt == 0:
                        break   # try fallback
                    return jsonify({"error": str(exc)}), 500
            retries += 1

    return jsonify({"error": "All models busy. Please try again in a moment."}), 503


# ── /api/health ───────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "model":    GEMINI_MODEL,
        "fallback": FALLBACK_MODEL,
    })


if __name__ == "__main__":
    print("\n[OK] OS Study Assistant backend starting")
    print(f"   Primary model  : {GEMINI_MODEL}")
    print(f"   Fallback model : {FALLBACK_MODEL}")
    print(f"   URL            : http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
