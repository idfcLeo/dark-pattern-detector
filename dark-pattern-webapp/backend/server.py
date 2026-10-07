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
import csv
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


@app.on_event("startup")
def load_model():
    """Load models once when the server starts, not on every request."""
    global model, vectorizer, mc_model, mc_vectorizer, ccpa_map
    try:
        model = joblib.load(MODELS_DIR / "binary_model.pkl")
        vectorizer = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")
        print("Binary model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"WARNING: could not load binary model from {MODELS_DIR}: {e}")
        return
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
            detail="Model not loaded. Make sure baseline_model.pkl and "
            "baseline_vectorizer.pkl are in the backend/ folder and restart the server.",
        )

    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    vec = vectorizer.transform([text])
    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))

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

def scrape_page_chunks(url: str, max_chunks: int = 300) -> List[str]:
    """Return a deduplicated list of short visible text chunks from a page."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise HTTPException(status_code=500, detail="Playwright not installed. Run: pip install playwright && playwright install chromium")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ))
        try:
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)  # let JS-rendered content settle
        except Exception as e:
            browser.close()
            raise HTTPException(status_code=400, detail=f"Could not load page: {e}")

        # Collect text from "leaf" elements only (no child elements) so we get
        # small microcopy pieces instead of one giant blob of the whole page.
        chunks = page.evaluate("""
            () => {
                const els = document.querySelectorAll('body *');
                const out = new Set();
                els.forEach(el => {
                    if (el.children.length === 0) {
                        const t = (el.innerText || '').trim().replace(/\\s+/g, ' ');
                        if (t.length >= 3 && t.length <= 300) out.add(t);
                    }
                });
                return Array.from(out);
            }
        """)
        browser.close()

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
            detail="Model not loaded. Make sure baseline_model.pkl and "
            "baseline_vectorizer.pkl are in the backend/ folder and restart the server.",
        )

    url = input.url.strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Please provide a full URL starting with http:// or https://")

    chunks = scrape_page_chunks(url)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any readable text from that page.")

    vecs = vectorizer.transform(chunks)
    preds = model.predict(vecs)
    probas = model.predict_proba(vecs)

    idxs = [i for i in range(len(chunks)) if preds[i] == 1]
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
