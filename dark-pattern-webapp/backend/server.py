"""
Dark Pattern Detector - Backend API
Loads the trained model + vectorizer and exposes a /predict endpoint
that the frontend website calls over HTTP.

Run with:
    uvicorn server:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import os
import re
import json
import csv
import string
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


@app.on_event("startup")
def load_model():
    """Load models once when the server starts, not on every request."""
    global model, vectorizer, mc_model, mc_vectorizer, ccpa_map, threshold
    try:
        model = joblib.load(MODELS_DIR / "binary_model.pkl")
        vectorizer = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")
        print("Binary model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"WARNING: could not load binary model from {MODELS_DIR}: {e}")
        return
    # Decision threshold tuned by train_model.py on CV (default 0.5 if absent)
    threshold_file = MODELS_DIR / "binary_threshold.json"
    if threshold_file.exists():
        with open(threshold_file) as f:
            threshold = float(json.load(f)["threshold"])
    print(f"Dark-pattern threshold: P(dark) >= {threshold:.2f}")
    try:
        mc_model = joblib.load(MODELS_DIR / "multiclass_model.pkl")
        mc_vectorizer = joblib.load(MODELS_DIR / "multiclass_vectorizer.pkl")
        print("Multi-class model loaded successfully.")
    except Exception as e:
        print(f"WARNING: multi-class model unavailable, categories disabled: {e}")
    if CCPA_MAPPING.exists():
        with open(CCPA_MAPPING, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                ccpa_map.setdefault(row["mathur_category"], []).append(
                    {"category": row["ccpa_category"], "description": row["ccpa_description"]}
                )


def categorize(texts: List[str]):
    """Return (category, ccpa_list) per text; (None, []) if no multiclass model."""
    if mc_model is None:
        return [(None, [])] * len(texts)
    cats = mc_model.predict(mc_vectorizer.transform(texts))
    return [(str(c), ccpa_map.get(str(c), [])) for c in cats]


class CcpaItem(BaseModel):
    category: str
    description: str


class TextInput(BaseModel):
    text: str


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
    """Frontend can call this on load to check the backend + model are ready."""
    return {
        "server": "up",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(input: TextInput):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py so models/binary_model.pkl and "
            "models/binary_vectorizer.pkl exist (or set DP_ROOT), then restart the server.",
        )

    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    p_dark = float(model.predict_proba(vectorizer.transform([text]))[0][1])
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


def scrape_page_chunks(url: str, max_chunks: int = 1500) -> List[str]:
    """
    Return deduplicated text snippets from a page, extracted the same way the
    training data was (scraper.py + prepare_dataset.py): body innerText split
    into lines, plus button/link texts. Matching the training extraction
    matters: "Only <b>3</b> left" is one line here but three leaf elements.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise HTTPException(status_code=500, detail="Playwright not installed. Run: pip install playwright && playwright install chromium")
    with sync_playwright() as p:
        # Some sites (e.g. Myntra, Nykaa) reset HTTP/2 connections from headless
        # Chromium; falling back to HTTP/1.1 gets most of them to load.
        browser = p.chromium.launch(headless=True, args=[
            "--disable-http2", "--disable-blink-features=AutomationControlled"])
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ), viewport={"width": 1366, "height": 900})
        try:
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)  # let JS-rendered content settle
            for _ in range(4):           # scroll to trigger lazy-loaded sections
                page.mouse.wheel(0, 2500)
                page.wait_for_timeout(500)
        except Exception as e:
            browser.close()
            raise HTTPException(status_code=400, detail=f"Could not load page: {e}")

        lines = page.inner_text("body").split("\n")
        lines += page.eval_on_selector_all(
            "button, a, input[type=submit]",
            "els => els.map(e => (e.innerText || e.value || '').trim().slice(0, 80))")
        browser.close()

    seen, chunks = set(), []
    for line in lines:
        t = re.sub(r"\s+", " ", line).strip()
        if t not in seen and not is_junk(t):
            seen.add(t)
            chunks.append(t)
    return chunks[:max_chunks]


class URLInput(BaseModel):
    url: str


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
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py so models/binary_model.pkl and "
            "models/binary_vectorizer.pkl exist (or set DP_ROOT), then restart the server.",
        )

    url = input.url.strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Please provide a full URL starting with http:// or https://")

    chunks = scrape_page_chunks(url)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any readable text from that page.")

    probas = model.predict_proba(vectorizer.transform(chunks))

    idxs = [i for i in range(len(chunks)) if probas[i][1] >= threshold]
    cats = categorize([chunks[i] for i in idxs]) if idxs else []
    flagged = [
        FlaggedChunk(
            text=chunks[i],
            confidence=round(float(probas[i][1]), 3),
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
