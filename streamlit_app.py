"""
Dark Pattern Detector - Streamlit app (for Streamlit Community Cloud).

Uses the same trained models as the FastAPI web app (models/*.pkl) and the
same CCPA mapping (configs/ccpa_mapping.tsv). Three modes:
  1. Single text   - classify one snippet
  2. Many snippets - classify one snippet per line, as a table
  3. Website URL   - fetch the page with plain HTTP (no headless browser, so it
                     runs on Streamlit Cloud) and classify its text lines

Run locally from the repo root:
    streamlit run streamlit_app.py
"""

import csv
import ipaddress
import json
import re
import socket
import string
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import joblib
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CCPA_MAPPING = ROOT / "configs" / "ccpa_mapping.tsv"

MAX_TEXT_CHARS = 5000
MAX_BATCH_LINES = 500
MAX_URL_CHARS = 2048
MAX_PAGE_BYTES = 5_000_000
MAX_REDIRECTS = 5
FETCH_TIMEOUT_S = 20

EXAMPLES = [
    "Only 2 left in stock - order soon!",
    "Hurry! Sale ends in 00:14:59",
    "1,248 people bought this in the last 24 hours",
    "No thanks, I don't want to save money",
    "Add to Cart",
    "Free delivery on orders above ₹499",
]

st.set_page_config(page_title="Dark Pattern Detector", page_icon="🕵️", layout="centered")


# ---------------------------------------------------------------------------
# Model loading. @st.cache_resource runs this once per server process, not on
# every click (Streamlit re-runs the whole script on each interaction).
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    art = {"threshold": 0.5, "mc_model": None, "mc_vectorizer": None, "ccpa": {}, "warnings": []}
    art["model"] = joblib.load(MODELS_DIR / "binary_model.pkl")
    art["vectorizer"] = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")
    art["trained_at"] = datetime.fromtimestamp(
        (MODELS_DIR / "binary_model.pkl").stat().st_mtime).strftime("%Y-%m-%d")

    try:
        value = float(json.loads((MODELS_DIR / "binary_threshold.json").read_text())["threshold"])
        if 0 < value < 1:
            art["threshold"] = value
    except Exception as e:
        art["warnings"].append(f"Threshold file unreadable ({e}); using 0.50.")

    try:
        art["mc_model"] = joblib.load(MODELS_DIR / "multiclass_model.pkl")
        art["mc_vectorizer"] = joblib.load(MODELS_DIR / "multiclass_vectorizer.pkl")
    except Exception as e:
        art["warnings"].append(f"Category model unavailable ({e}).")

    try:
        with open(CCPA_MAPPING, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                art["ccpa"].setdefault(row["mathur_category"], []).append(
                    {"category": row["ccpa_category"], "description": row["ccpa_description"]})
    except Exception as e:
        art["warnings"].append(f"CCPA mapping unavailable ({e}).")
    return art


def classify(texts, art):
    """Return one dict per text: P(dark), verdict, and (if dark) category + CCPA."""
    p_dark = art["model"].predict_proba(art["vectorizer"].transform(texts))[:, 1]
    results = [{"text": t, "p_dark": float(p), "dark": bool(p >= art["threshold"]),
                "category": None, "ccpa": []} for t, p in zip(texts, p_dark)]
    dark = [r for r in results if r["dark"]]
    if dark and art["mc_model"] is not None:
        cats = art["mc_model"].predict(art["mc_vectorizer"].transform([r["text"] for r in dark]))
        for r, c in zip(dark, cats):
            r["category"] = str(c)
            r["ccpa"] = art["ccpa"].get(str(c), [])
    return results


# ---------------------------------------------------------------------------
# URL scanning. Same junk filter as prepare_dataset.is_junk and the FastAPI
# backend (keep the three in sync), so inputs look like the training text.
# ---------------------------------------------------------------------------
def is_junk(text: str) -> bool:
    text = text.strip()
    if len(text) < 4 or len(text) > 300:
        return True
    if re.match(r'^[\d\s.,₹$%/\-:;]+$', text):
        return True
    if all(c in string.punctuation + ' \t' for c in text):
        return True
    if re.match(r'^[\d.\s]*[★☆⭐]+[\d.\s]*$', text):
        return True
    return len(set(text)) <= 1


def check_public(url: str):
    """Allow only http(s) URLs that resolve to public IPs. The server fetches
    the URL itself, so this stops it being pointed at internal addresses (SSRF)."""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError("Please enter a full URL starting with http:// or https://")
    host = urlparse(url).hostname
    if not host:
        raise ValueError("That URL has no site name in it.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        raise ValueError(f"Could not find the site '{host}'. Check the address.")
    if any(not ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses):
        raise ValueError("Only public websites can be scanned.")


def fetch_page_chunks(url: str, max_chunks: int = 1500):
    """Download the page HTML (re-checking every redirect hop) and return
    deduplicated, non-junk text lines plus button/link texts."""
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-IN,en;q=0.9",
    }
    for _ in range(MAX_REDIRECTS + 1):
        check_public(url)
        try:
            resp = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT_S,
                                allow_redirects=False, stream=True)
        except requests.Timeout:
            raise ValueError(f"The site took longer than {FETCH_TIMEOUT_S} s to respond.")
        except requests.RequestException as e:
            raise ValueError(f"Could not load the page ({type(e).__name__}).")
        if resp.is_redirect:
            url = urljoin(url, resp.headers.get("Location", ""))
            continue
        break
    else:
        raise ValueError("Too many redirects.")

    if resp.status_code in (401, 403, 429, 503):
        raise ValueError(f"The site blocked automated access (HTTP {resp.status_code}). "
                         "Paste the text into the 'Single text' or 'Many snippets' tab instead.")
    if resp.status_code >= 400:
        raise ValueError(f"The site returned an error page (HTTP {resp.status_code}).")
    content = resp.raw.read(MAX_PAGE_BYTES, decode_content=True)
    if not content.strip():  # e.g. Amazon answers bots with an empty HTTP 202
        raise ValueError(f"The site sent back an empty page (HTTP {resp.status_code}), which usually "
                         "means it blocks automated access. Paste the page text into the "
                         "'Many snippets' tab instead.")
    soup = BeautifulSoup(content, "html.parser")
    buttons = [(el.get_text(" ") or el.get("value") or "").strip()[:80]
               for el in soup.select("button, a, input[type=submit]")]
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    lines = soup.get_text("\n").split("\n") + buttons

    seen, chunks = set(), []
    for line in lines:
        t = re.sub(r"\s+", " ", line).strip()
        if t not in seen and not is_junk(t):
            seen.add(t)
            chunks.append(t)
    return chunks[:max_chunks]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def show_result(r):
    if r["dark"]:
        st.error(f"⚠️ **Dark pattern** (P(dark) = {r['p_dark']:.2f})")
        if r["category"]:
            st.markdown(f"**Category (Mathur et al.):** {r['category']}")
        for c in r["ccpa"]:
            st.markdown(f"**CCPA 2023 guideline:** {c['category']} — _{c['description']}_")
    else:
        st.success(f"✅ **Not a dark pattern** (P(dark) = {r['p_dark']:.2f})")
    st.progress(min(max(r["p_dark"], 0.0), 1.0), text="Probability of dark pattern")


def results_table(results):
    return pd.DataFrame([{
        "Text": r["text"],
        "P(dark)": round(r["p_dark"], 3),
        "Verdict": "Dark" if r["dark"] else "Not dark",
        "Category": r["category"] or "",
        "CCPA guideline": ", ".join(sorted({c["category"] for c in r["ccpa"]})),
    } for r in results])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🕵️ Dark Pattern Detector")
st.caption("Detects manipulative e-commerce text (urgency, scarcity, social proof, "
           "misdirection, …) and maps it to India's CCPA 2023 dark-pattern guidelines.")

try:
    art = load_artifacts()
except Exception as e:
    st.error(f"Could not load the trained model from `{MODELS_DIR}`: {e}")
    st.stop()

with st.sidebar:
    st.header("About the model")
    st.markdown(
        f"- Binary model: `{type(art['model']).__name__}`\n"
        f"- Category model: `{type(art['mc_model']).__name__ if art['mc_model'] is not None else 'n/a'}`\n"
        f"- Decision threshold: **{art['threshold']:.2f}**\n"
        f"- Trained: {art['trained_at']}\n"
        "- Features: word 1–2-gram + char 2–5-gram TF-IDF")
    st.markdown("Taxonomy: Mathur et al. (2019). Legal mapping: CCPA "
                "*Guidelines for Prevention and Regulation of Dark Patterns*, 2023.")
    for w in art["warnings"]:
        st.warning(w)

tab_text, tab_batch, tab_url = st.tabs(["Single text", "Many snippets", "Website URL"])

with tab_text:
    example = st.selectbox("Try an example (or type your own below)", ["—"] + EXAMPLES)
    text = st.text_area("Text from a shopping page", value="" if example == "—" else example,
                        max_chars=MAX_TEXT_CHARS, height=120)
    if st.button("Analyze", type="primary", key="analyze_text"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            show_result(classify([text.strip()], art)[0])

with tab_batch:
    raw = st.text_area("One snippet per line", height=200, value="\n".join(EXAMPLES),
                       max_chars=MAX_TEXT_CHARS * 10)
    if st.button("Analyze all", type="primary", key="analyze_batch"):
        lines = [l.strip() for l in raw.splitlines() if l.strip()][:MAX_BATCH_LINES]
        if not lines:
            st.warning("Enter at least one line.")
        else:
            results = classify(lines, art)
            n_dark = sum(r["dark"] for r in results)
            st.metric("Dark patterns found", f"{n_dark} / {len(results)}")
            df = results_table(results).sort_values("P(dark)", ascending=False)
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), "dark_pattern_results.csv")

with tab_url:
    st.info("Fetches the page's HTML directly (no browser), so sites that build their "
            "content with JavaScript or block bots may return little text. If that "
            "happens, copy the page text into the 'Many snippets' tab.")
    url = st.text_input("Product or checkout page URL", max_chars=MAX_URL_CHARS,
                        placeholder="https://www.amazon.in/dp/...")
    if st.button("Scan page", type="primary", key="analyze_url"):
        try:
            with st.spinner("Fetching and analysing the page…"):
                chunks = fetch_page_chunks(url.strip())
        except ValueError as e:
            st.error(str(e))
        else:
            if not chunks:
                st.warning("Could not extract any readable text from that page.")
            else:
                results = classify(chunks, art)
                flagged = sorted([r for r in results if r["dark"]], key=lambda r: -r["p_dark"])
                c1, c2 = st.columns(2)
                c1.metric("Text snippets scanned", len(results))
                c2.metric("Dark patterns found", len(flagged))
                if flagged:
                    st.dataframe(results_table(flagged[:100]), width="stretch", hide_index=True)
                else:
                    st.success("No dark patterns detected.")
                with st.expander("All scanned snippets"):
                    st.dataframe(results_table(results), width="stretch", hide_index=True)

st.caption("Research prototype. Predictions can be wrong, especially on Indian sites; "
           "see the project's gold-set evaluation.")
