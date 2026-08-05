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
from typing import List
import joblib
import os
from playwright.sync_api import sync_playwright

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

MODEL_PATH = "baseline_model.pkl"
VECTORIZER_PATH = "baseline_vectorizer.pkl"

model = None
vectorizer = None


@app.on_event("startup")
def load_model():
    """Load the model once when the server starts, not on every request."""
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(
            f"WARNING: {MODEL_PATH} or {VECTORIZER_PATH} not found in this folder. "
            f"Copy your trained .pkl files into backend/ before starting the server."
        )
        return
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")


class TextInput(BaseModel):
    text: str


class PredictionResult(BaseModel):
    is_dark_pattern: bool
    confidence: float
    label: str


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

    return PredictionResult(
        is_dark_pattern=bool(pred),
        confidence=round(confidence, 4),
        label="Dark Pattern" if pred == 1 else "Not Dark Pattern",
    )


# ---------------------------------------------------------------------------
# URL analysis: load the real page with a headless browser (so JS-rendered
# e-commerce sites work), pull out small "leaf" text chunks (buttons, badges,
# banners, labels), and classify each one separately.
# ---------------------------------------------------------------------------

def scrape_page_chunks(url: str, max_chunks: int = 300) -> List[str]:
    """Return a deduplicated list of short visible text chunks from a page."""
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

    flagged = [
        FlaggedChunk(text=chunks[i], confidence=round(float(probas[i][1]), 3))
        for i in range(len(chunks)) if preds[i] == 1
    ]
    flagged.sort(key=lambda c: -c.confidence)

    return URLAnalysisResult(
        url=url,
        chunks_scanned=len(chunks),
        dark_patterns_found=len(flagged),
        verdict="Dark patterns detected" if flagged else "No dark patterns detected",
        flagged=flagged[:40],
    )
