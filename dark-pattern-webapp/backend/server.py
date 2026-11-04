"""
Dark Pattern Detector - Backend API
Loads the trained model + vectorizer and exposes a /predict endpoint
that the frontend website calls over HTTP.

Run with:
    uvicorn server:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse
import ipaddress
import joblib
import os
import re
import json
import csv
import socket
import string
import threading
from pathlib import Path

app = FastAPI(title="Dark Pattern Detector API")

# ---------------------------------------------------------------------------
# CORS: without this, a browser will BLOCK requests from your frontend
# (running on e.g. http://127.0.0.1:5500) to your backend (http://127.0.0.1:8000)
# even though both are "localhost". This is a browser security rule, not a bug.
# For a real deployed site, replace "*" with your actual frontend URL.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trained artifacts live in the main project (dark-pattern-detector/models).
# Override with the DP_ROOT env var if the webapp is moved elsewhere.
PROJECT_ROOT = Path(os.environ.get("DP_ROOT", Path(__file__).resolve().parents[2]))
MODELS_DIR = PROJECT_ROOT / "models"
CCPA_MAPPING = PROJECT_ROOT / "configs" / "ccpa_mapping.tsv"

model = None
vectorizer = None
mc_model = None
mc_vectorizer = None
ccpa_map = {}  # mathur_category -> [{category, description}, ...]
threshold = 0.5  # P(dark) at or above this is flagged
model_info = {}  # model class names + training time, reported by /health

# Limits that keep one request from hogging the server
MAX_TEXT_CHARS = 5000
MAX_URL_CHARS = 2048
MAX_CONCURRENT_SCANS = 2       # each URL scan runs its own headless Chromium
PAGE_LOAD_TIMEOUT_MS = 30000
ACTION_TIMEOUT_MS = 15000      # any later browser step (scroll, read text)
_scan_slots = threading.Semaphore(MAX_CONCURRENT_SCANS)


@app.on_event("startup")
def load_model():
    """Load models once when the server starts, not on every request.
    A missing or broken optional file (threshold, multi-class, CCPA) only
    disables that feature with a warning; it never stops the server."""
    global model, vectorizer, mc_model, mc_vectorizer, ccpa_map, threshold
    try:
        model = joblib.load(MODELS_DIR / "binary_model.pkl")
        vectorizer = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")
        print("Binary model and vectorizer loaded successfully.")
    except Exception as e:
        model = vectorizer = None
        print(f"WARNING: could not load binary model from {MODELS_DIR}: {e}")
        return
    model_info["binary_model"] = type(model).__name__
    model_info["models_trained_at"] = datetime.fromtimestamp(
        (MODELS_DIR / "binary_model.pkl").stat().st_mtime).isoformat(timespec="seconds")

    # Decision threshold saved by train_model.py (0.5 if absent or unreadable)
    threshold_file = MODELS_DIR / "binary_threshold.json"
    if threshold_file.exists():
        try:
            value = float(json.loads(threshold_file.read_text())["threshold"])
            if not 0 < value < 1:
                raise ValueError(f"{value} is not between 0 and 1")
            threshold = value
        except Exception as e:
            print(f"WARNING: could not read {threshold_file} ({e}); using 0.50")
    print(f"Dark-pattern threshold: P(dark) >= {threshold:.2f}")

    try:
        mc_model = joblib.load(MODELS_DIR / "multiclass_model.pkl")
        mc_vectorizer = joblib.load(MODELS_DIR / "multiclass_vectorizer.pkl")
        model_info["multiclass_model"] = type(mc_model).__name__
        print("Multi-class model loaded successfully.")
    except Exception as e:
        mc_model = mc_vectorizer = None
        print(f"WARNING: multi-class model unavailable, categories disabled: {e}")

    if CCPA_MAPPING.exists():
        try:
            with open(CCPA_MAPPING, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    ccpa_map.setdefault(row["mathur_category"], []).append(
                        {"category": row["ccpa_category"], "description": row["ccpa_description"]}
                    )
        except Exception as e:
            ccpa_map.clear()
            print(f"WARNING: could not read {CCPA_MAPPING} ({e}); CCPA tags disabled")


def categorize(texts: List[str]):
    """Return (category, ccpa_list) per text; (None, []) if no multiclass model.
    A failure here only drops the category tags, not the whole response."""
    if mc_model is None:
        return [(None, [])] * len(texts)
    try:
        cats = mc_model.predict(mc_vectorizer.transform(texts))
    except Exception as e:
        print(f"WARNING: category prediction failed: {e}")
        return [(None, [])] * len(texts)
    return [(str(c), ccpa_map.get(str(c), [])) for c in cats]


def require_model():
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py so models/binary_model.pkl and "
            "models/binary_vectorizer.pkl exist (or set DP_ROOT), then restart the server.",
        )


def dark_probabilities(texts: List[str]):
    """P(dark) per text. Raised as an HTTPException (not a bare crash) so the
    JSON error still carries CORS headers and the frontend can show it."""
    try:
        return model.predict_proba(vectorizer.transform(texts))[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"The model failed on this input: {e}")


class CcpaItem(BaseModel):
    category: str
    description: str


class TextInput(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)


class PredictionResult(BaseModel):
    is_dark_pattern: bool
    confidence: float
    label: str
    category: Optional[str] = None
    ccpa: List[CcpaItem] = []


@app.get("/")
def root():
    return {"status": "ok", "message": "Dark Pattern Detector API is running"}


@app.get("/health")
def health():
    """Frontend can call this on load to check the backend + model are ready.
    Also reports which models and threshold are live, to spot a stale server."""
    return {
        "server": "up",
        "model_loaded": model is not None and vectorizer is not None,
        "categories_loaded": mc_model is not None,
        "ccpa_loaded": bool(ccpa_map),
        "threshold": threshold,
        **model_info,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(input: TextInput):
    require_model()
    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    p_dark = float(dark_probabilities([text])[0])
    pred = int(p_dark >= threshold)
    confidence = p_dark if pred else 1 - p_dark

    category, ccpa = categorize([text])[0] if pred == 1 else (None, [])
    return PredictionResult(
        is_dark_pattern=bool(pred),
        confidence=round(confidence, 4),
        label="Dark Pattern" if pred == 1 else "Not Dark Pattern",
        category=category,
        ccpa=ccpa,
    )


# ---------------------------------------------------------------------------
# URL analysis: load the real page with a headless browser (so JS-rendered
# e-commerce sites work), pull out small "leaf" text chunks (buttons, badges,
# banners, labels), and classify each one separately.
# ---------------------------------------------------------------------------

def is_junk(text: str) -> bool:
    """Same filter prepare_dataset.py applies to training snippets (copied, not
    imported, so the backend stays independent of the training repo)."""
    text = text.strip()
    if len(text) < 4 or len(text) > 300:
        return True
    if re.match(r'^[\d\s.,₹$%/\-:;]+$', text):          # pure numbers / prices
        return True
    if all(c in string.punctuation + ' \t' for c in text):
        return True
    if re.match(r'^[\d.\s]*[★☆⭐]+[\d.\s]*$', text):     # star ratings
        return True
    return len(set(text)) <= 1


BLOCKED_MSG = ("This site blocked our automated browser (common on sites with bot protection, "
               "such as Myntra and Nykaa). Paste the text you want to check into the text box instead.")
BLOCK_STATUSES = {401, 403, 429, 503}
BLOCK_PHRASES = ("access denied", "captcha", "are you a robot", "verify you are human",
                 "request blocked", "unusual traffic", "pardon our interruption")
# Network errors that, for a site that exists, usually mean bot protection cut us off
BLOCK_NET_ERRORS = ("ERR_HTTP2_PROTOCOL_ERROR", "ERR_CONNECTION_RESET", "ERR_EMPTY_RESPONSE")


def validate_url(raw: str) -> str:
    """Accept only http(s) URLs of public websites. The server fetches the
    URL itself, so without this anyone could make it load localhost or
    private-network pages (server-side request forgery)."""
    url = raw.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Please provide a full URL starting with http:// or https://")
    host = urlparse(url).hostname
    if not host:
        raise HTTPException(status_code=400, detail="That URL has no site name in it.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        raise HTTPException(status_code=400, detail=f"Could not find the site '{host}'. Check the address.")
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            raise HTTPException(status_code=400, detail="Only public websites can be scanned.")
    return url


def scrape_page_chunks(url: str, max_chunks: int = 1500) -> List[str]:
    """
    Return deduplicated text snippets from a page, extracted the same way the
    training data was (scraper.py + prepare_dataset.py): body innerText split
    into lines, plus button/link texts. Matching the training extraction
    matters: "Only <b>3</b> left" is one line here but three leaf elements.

    Every failure becomes an HTTPException with a message the user can act on,
    and the browser is always closed (try/finally), even when a step fails.
    """
    try:
        from playwright.sync_api import sync_playwright, Error as PWError, TimeoutError as PWTimeout
    except ImportError:
        raise HTTPException(status_code=500, detail="Playwright not installed. Run: pip install playwright && playwright install chromium")
    with sync_playwright() as p:
        try:
            # Some sites (e.g. Myntra, Nykaa) reset HTTP/2 connections from headless
            # Chromium; falling back to HTTP/1.1 gets most of them to load.
            browser = p.chromium.launch(headless=True, args=[
                "--disable-http2", "--disable-blink-features=AutomationControlled"])
        except PWError as e:
            hint = ("Run: playwright install chromium" if "Executable doesn't exist" in str(e)
                    else str(e).splitlines()[0])
            raise HTTPException(status_code=500, detail=f"Could not start the headless browser. {hint}")
        try:
            page = browser.new_page(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ), viewport={"width": 1366, "height": 900})
            page.set_default_timeout(ACTION_TIMEOUT_MS)
            try:
                response = page.goto(url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")
            except PWTimeout:
                raise HTTPException(status_code=504, detail=(
                    f"The page took longer than {PAGE_LOAD_TIMEOUT_MS // 1000} s to load. Some sites "
                    "(e.g. Myntra, Nykaa) stall automated browsers this way. Paste the text into "
                    "the text box instead, or try again."))
            except PWError as e:
                if any(code in str(e) for code in BLOCK_NET_ERRORS):
                    raise HTTPException(status_code=502, detail=BLOCKED_MSG)
                code = re.search(r"net::(ERR_[A-Z0-9_]+)", str(e))
                raise HTTPException(status_code=502, detail=(
                    f"Could not load the page ({code.group(1) if code else str(e).splitlines()[0]})."))
            if response is not None and response.status in BLOCK_STATUSES:
                raise HTTPException(status_code=502, detail=f"{BLOCKED_MSG} (HTTP {response.status})")
            if response is not None and response.status >= 400:
                raise HTTPException(status_code=502, detail=f"The site returned an error page (HTTP {response.status}).")

            try:
                page.wait_for_timeout(1500)  # let JS-rendered content settle
                for _ in range(4):           # scroll to trigger lazy-loaded sections
                    page.mouse.wheel(0, 2500)
                    page.wait_for_timeout(500)
                body = page.inner_text("body")
                buttons = page.eval_on_selector_all(
                    "button, a, input[type=submit]",
                    "els => els.map(e => (e.innerText || e.value || '').trim().slice(0, 80))")
            except PWError as e:
                raise HTTPException(status_code=502, detail=f"Could not read the page: {str(e).splitlines()[0]}")
        finally:
            browser.close()

    # A bot-check page loads fine (HTTP 200) but is short and says so
    if len(body) < 2000 and any(phrase in body.lower() for phrase in BLOCK_PHRASES):
        raise HTTPException(status_code=502, detail=BLOCKED_MSG)

    lines = body.split("\n") + buttons
    seen, chunks = set(), []
    for line in lines:
        t = re.sub(r"\s+", " ", line).strip()
        if t not in seen and not is_junk(t):
            seen.add(t)
            chunks.append(t)
    return chunks[:max_chunks]


class URLInput(BaseModel):
    url: str = Field(..., max_length=MAX_URL_CHARS)


class FlaggedChunk(BaseModel):
    text: str
    confidence: float
    category: Optional[str] = None
    ccpa: List[CcpaItem] = []


class URLAnalysisResult(BaseModel):
    url: str
    chunks_scanned: int
    dark_patterns_found: int
    verdict: str
    flagged: List[FlaggedChunk]


@app.post("/analyze-url", response_model=URLAnalysisResult)
def analyze_url(input: URLInput):
    require_model()
    url = validate_url(input.url)

    # Each scan starts a whole browser; refuse extra ones instead of piling them up
    if not _scan_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The server is already scanning other pages. Try again in a few seconds.")
    try:
        chunks = scrape_page_chunks(url)
    finally:
        _scan_slots.release()
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any readable text from that page.")

    p_dark = dark_probabilities(chunks)

    idxs = [i for i in range(len(chunks)) if p_dark[i] >= threshold]
    cats = categorize([chunks[i] for i in idxs]) if idxs else []
    flagged = [
        FlaggedChunk(
            text=chunks[i],
            confidence=round(float(p_dark[i]), 3),
            category=cat,
            ccpa=ccpa,
        )
        for i, (cat, ccpa) in zip(idxs, cats)
    ]
    flagged.sort(key=lambda c: -c.confidence)

    return URLAnalysisResult(
        url=url,
        chunks_scanned=len(chunks),
        dark_patterns_found=len(flagged),
        verdict="Dark patterns detected" if flagged else "No dark patterns detected",
        flagged=flagged[:40],
    )
