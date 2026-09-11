"""
prepare_dataset.py — Dark Pattern Training Data Pipeline

Stages 1-5: Extract → Pseudo-label → Structural features → Merge → CCPA mapping
Designed for expandability: scrape a new site, rerun this script, done.
"""

import os
import re
import json
import string
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RAW_DATA_DIR = Path("data/raw")
ORIGINAL_DATASET = Path("ec-darkpattern/dataset/dataset.tsv")
BASELINE_MODEL_PATH = Path("ec-darkpattern/baseline_model.pkl")
BASELINE_VECTORIZER_PATH = Path("ec-darkpattern/baseline_vectorizer.pkl")
MANUAL_LABELS_PATH = Path("data/manual_labels.tsv")
CCPA_MAPPING_PATH = Path("configs/ccpa_mapping.tsv")

OUTPUT_DIR = Path("data/processed")
COMBINED_OUTPUT = OUTPUT_DIR / "combined_dataset.tsv"
NEEDS_REVIEW_OUTPUT = OUTPUT_DIR / "needs_review.tsv"
STATS_OUTPUT = OUTPUT_DIR / "pipeline_stats.txt"

# Filtering thresholds
MIN_SNIPPET_LENGTH = 4          # chars — drop anything shorter
BINARY_CONFIDENCE_THRESHOLD = 0.80
MULTICLASS_CONFIDENCE_THRESHOLD = 0.65  # lower bar: fewer categories, noisier

# Class balance: max ratio of scraped "Not Dark Pattern" to original "Not Dark Pattern"
MAX_BENIGN_RATIO = 2.0  # at most 2x the original count

# Rare category handling for multi-class
RARE_CATEGORY_MIN_SAMPLES = 15  # categories below this get merged into "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Extract & Clean Text Snippets
# ─────────────────────────────────────────────────────────────────────────────

def is_junk(text: str) -> bool:
    """Filter out non-informative text snippets."""
    if not text or not text.strip():
        return True
    text = text.strip()
    if len(text) < MIN_SNIPPET_LENGTH:
        return True
    # Pure numeric (prices, ratings, counts)
    if re.match(r'^[\d\s.,₹$%/\-:;]+$', text):
        return True
    # Pure symbols/punctuation
    if all(c in string.punctuation + ' \t' for c in text):
        return True
    # Star ratings like "★★★★☆" or "4.5 ★"
    if re.match(r'^[\d.\s]*[★☆⭐]+[\d.\s]*$', text):
        return True
    # Single repeated character
    if len(set(text.strip())) <= 1:
        return True
    return False


def extract_from_txt(txt_path: Path) -> list[str]:
    """Extract text lines from a .txt page body file."""
    try:
        content = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = content.split("\n")
    return [line.strip() for line in lines if not is_junk(line)]


def extract_from_json(json_path: Path) -> tuple[list[str], list[dict]]:
    """Extract clickable element texts + raw element metadata from a .json file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], []

    elements = data.get("clickable_elements", [])
    texts = []
    valid_elements = []
    for el in elements:
        t = (el.get("text") or "").strip()
        if not is_junk(t):
            texts.append(t)
            valid_elements.append(el)
    return texts, valid_elements


def parse_step_label(filename: str) -> str:
    """Extract step label from filename like 'step03_cart_20260805_152530'."""
    match = re.match(r'step\d+_(.+?)_\d{8}_\d{6}', filename)
    if match:
        return match.group(1)
    return "unknown"


def extract_all_snippets() -> pd.DataFrame:
    """Stage 1: Walk data/raw/, extract and clean all text snippets."""
    records = []
    seen_per_site = {}  # site -> set of texts (within-site dedup)

    if not RAW_DATA_DIR.exists():
        print(f"ERROR: {RAW_DATA_DIR} not found")
        return pd.DataFrame()

    sites = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    print(f"\n{'='*60}")
    print(f"STAGE 1: Extracting snippets from {len(sites)} sites")
    print(f"{'='*60}")

    for site_dir in sites:
        site_name = site_dir.name
        seen_per_site[site_name] = set()

        # Process each JSON file (has both element text + metadata)
        json_files = sorted(site_dir.glob("*.json"))
        for jf in json_files:
            step_label = parse_step_label(jf.stem)
            texts, elements = extract_from_json(jf)

            for i, txt in enumerate(texts):
                if txt in seen_per_site[site_name]:
                    continue
                seen_per_site[site_name].add(txt)

                el = elements[i] if i < len(elements) else {}
                records.append({
                    "text": txt,
                    "site": site_name,
                    "step_label": step_label,
                    "domain": "indian",
                    "source_file": jf.name,
                    "source_type": "json_element",
                    # Structural metadata (Stage 3 uses these)
                    "el_visible": el.get("visible"),
                    "el_width": el.get("width"),
                    "el_height": el.get("height"),
                    "el_x": el.get("x"),
                    "el_y": el.get("y"),
                    "el_tag": el.get("tag"),
                })

        # Process .txt files for page body text not captured in elements
        txt_files = sorted(site_dir.glob("*.txt"))
        for tf in txt_files:
            step_label = parse_step_label(tf.stem)
            lines = extract_from_txt(tf)

            for line in lines:
                if line in seen_per_site[site_name]:
                    continue
                seen_per_site[site_name].add(line)

                records.append({
                    "text": line,
                    "site": site_name,
                    "step_label": step_label,
                    "domain": "indian",
                    "source_file": tf.name,
                    "source_type": "txt_body",
                    "el_visible": None,
                    "el_width": None,
                    "el_height": None,
                    "el_x": None,
                    "el_y": None,
                    "el_tag": None,
                })

    df = pd.DataFrame(records)

    # Global dedup (across sites) — keep first occurrence
    before_global = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    after_global = len(df)

    print(f"  Raw records (within-site deduped): {before_global}")
    print(f"  After global dedup: {after_global} (removed {before_global - after_global} cross-site duplicates)")
    print(f"  Sites: {df['site'].nunique()}")
    print(f"  Step labels: {dict(df['step_label'].value_counts())}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Pseudo-Labeling (Binary + Multi-class)
# ─────────────────────────────────────────────────────────────────────────────

def train_interim_multiclass_model() -> tuple:
    """
    Train a throwaway multi-class classifier on dataset.tsv's Pattern Category.
    Returns (model, vectorizer, label_classes).
    Only used to pseudo-label dark-pattern snippets from Stage 2 binary pass.
    """
    print(f"\n{'='*60}")
    print("STAGE 2a: Training interim multi-class classifier on original dataset")
    print(f"{'='*60}")

    df = pd.read_csv(ORIGINAL_DATASET, sep="\t", encoding="utf-8")
    # Only train on dark pattern rows (label=1) — non-dark-patterns don't need a category
    dark_df = df[df["label"] == 1].copy()

    texts = dark_df["text"].tolist()
    labels = dark_df["Pattern Category"].tolist()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    # Use LogisticRegression — fast, gives predict_proba, good enough for pseudo-labels
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, labels)

    # Quick self-eval to sanity-check
    preds = model.predict(X)
    print(f"  Training accuracy (self-eval, not a real metric): "
          f"{(np.array(preds) == np.array(labels)).mean():.3f}")
    print(f"  Classes: {list(model.classes_)}")

    return model, vectorizer, model.classes_


def pseudo_label(scraped_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stage 2: Two-pass pseudo-labeling.
    Returns (accepted_df, needs_review_df).
    """
    print(f"\n{'='*60}")
    print(f"STAGE 2b: Binary pseudo-labeling ({len(scraped_df)} snippets)")
    print(f"{'='*60}")

    # ── Binary pass ──
    binary_model = joblib.load(BASELINE_MODEL_PATH)
    binary_vectorizer = joblib.load(BASELINE_VECTORIZER_PATH)

    X_binary = binary_vectorizer.transform(scraped_df["text"])
    binary_preds = binary_model.predict(X_binary)
    binary_proba = binary_model.predict_proba(X_binary)
    binary_confidence = np.max(binary_proba, axis=1)

    scraped_df = scraped_df.copy()
    scraped_df["label"] = binary_preds
    scraped_df["binary_confidence"] = binary_confidence

    # Stats
    n_dark = (binary_preds == 1).sum()
    n_benign = (binary_preds == 0).sum()
    print(f"  Binary predictions: {n_dark} dark, {n_benign} not-dark")
    print(f"  Confidence: mean={binary_confidence.mean():.3f}, "
          f"median={np.median(binary_confidence):.3f}, "
          f"min={binary_confidence.min():.3f}")

    # ── Multi-class pass (only on dark-pattern rows) ──
    print(f"\n{'='*60}")
    print(f"STAGE 2c: Multi-class pseudo-labeling ({n_dark} dark-pattern snippets)")
    print(f"{'='*60}")

    mc_model, mc_vectorizer, mc_classes = train_interim_multiclass_model()

    # Default all to "Not Dark Pattern"
    scraped_df["Pattern Category"] = "Not Dark Pattern"
    scraped_df["multiclass_confidence"] = 1.0  # benign rows get full confidence

    dark_mask = scraped_df["label"] == 1
    if dark_mask.sum() > 0:
        X_mc = mc_vectorizer.transform(scraped_df.loc[dark_mask, "text"])
        mc_preds = mc_model.predict(X_mc)
        mc_proba = mc_model.predict_proba(X_mc)
        mc_confidence = np.max(mc_proba, axis=1)

        scraped_df.loc[dark_mask, "Pattern Category"] = mc_preds
        scraped_df.loc[dark_mask, "multiclass_confidence"] = mc_confidence

        print(f"  Category distribution:")
        for cat, count in Counter(mc_preds).most_common():
            print(f"    {cat}: {count}")

    # ── Confidence-based split ──
    # Combined confidence = min of binary and multiclass confidence
    scraped_df["confidence"] = scraped_df[["binary_confidence", "multiclass_confidence"]].min(axis=1)

    accepted_mask = (
        (scraped_df["label"] == 0) & (scraped_df["binary_confidence"] >= BINARY_CONFIDENCE_THRESHOLD)
    ) | (
        (scraped_df["label"] == 1) & (scraped_df["binary_confidence"] >= BINARY_CONFIDENCE_THRESHOLD)
        & (scraped_df["multiclass_confidence"] >= MULTICLASS_CONFIDENCE_THRESHOLD)
    )

    accepted_df = scraped_df[accepted_mask].copy()
    review_df = scraped_df[~accepted_mask].copy()

    print(f"\n  Confidence split (binary threshold={BINARY_CONFIDENCE_THRESHOLD}, "
          f"multi-class threshold={MULTICLASS_CONFIDENCE_THRESHOLD}):")
    print(f"    Accepted: {len(accepted_df)} snippets")
    print(f"    Needs review: {len(review_df)} snippets")

    # Drop intermediate columns
    for col in ["binary_confidence", "multiclass_confidence"]:
        accepted_df.drop(columns=[col], inplace=True, errors="ignore")
        review_df.drop(columns=[col], inplace=True, errors="ignore")

    return accepted_df, review_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Structural Features (DOM proxy)
# ─────────────────────────────────────────────────────────────────────────────

# Step labels that indicate higher-risk pages for dark patterns
HIGH_RISK_STEPS = {"checkout", "payment", "cancel_step", "settings"}

def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3: Derive structural features from JSON element metadata.
    These are DOM-based proxies, NOT visual/pixel features from screenshots.
    """
    print(f"\n{'='*60}")
    print("STAGE 3: Adding structural features")
    print(f"{'='*60}")

    df = df.copy()

    # Feature: is the element hidden? (width=0 or height=0 or visible=False)
    df["feat_hidden"] = (
        (df["el_visible"] == False) |
        (df["el_width"] == 0) |
        (df["el_height"] == 0)
    ).astype(float)
    # NaN for rows without element data
    df.loc[df["el_visible"].isna(), "feat_hidden"] = np.nan

    # Feature: tiny element (area < 400px² — roughly a 20x20 button)
    df["feat_tiny"] = 0.0
    has_dims = df["el_width"].notna() & df["el_height"].notna()
    area = df.loc[has_dims, "el_width"] * df.loc[has_dims, "el_height"]
    df.loc[has_dims, "feat_tiny"] = (area < 400).astype(float)
    df.loc[~has_dims, "feat_tiny"] = np.nan

    # Feature: element buried at bottom (y > 5000px — below the fold)
    df["feat_bottom_buried"] = 0.0
    has_y = df["el_y"].notna()
    df.loc[has_y, "feat_bottom_buried"] = (df.loc[has_y, "el_y"] > 5000).astype(float)
    df.loc[~has_y, "feat_bottom_buried"] = np.nan

    # Feature: high-risk page step
    df["feat_high_risk_step"] = df["step_label"].isin(HIGH_RISK_STEPS).astype(float)

    # Feature: element tag (one-hot for BUTTON vs A vs INPUT)
    df["feat_is_button"] = (df["el_tag"] == "BUTTON").astype(float)
    df["feat_is_link"] = (df["el_tag"] == "A").astype(float)
    df.loc[df["el_tag"].isna(), ["feat_is_button", "feat_is_link"]] = np.nan

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    n_with_feats = df[feat_cols[0]].notna().sum()
    print(f"  Structural features added: {feat_cols}")
    print(f"  Rows with structural data: {n_with_feats} / {len(df)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Merge & Balance
# ─────────────────────────────────────────────────────────────────────────────

def apply_manual_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply manual label corrections from manual_labels.tsv."""
    if not MANUAL_LABELS_PATH.exists():
        return df

    overrides = pd.read_csv(MANUAL_LABELS_PATH, sep="\t", encoding="utf-8")
    if len(overrides) == 0:
        return df

    print(f"  Applying {len(overrides)} manual label overrides")
    override_map = dict(zip(overrides["text"], zip(overrides["label"], overrides["Pattern Category"])))

    for idx, row in df.iterrows():
        if row["text"] in override_map:
            label, cat = override_map[row["text"]]
            df.at[idx, "label"] = label
            df.at[idx, "Pattern Category"] = cat

    return df


def merge_and_balance(scraped_accepted: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 4: Merge original dataset with pseudo-labeled scraped data.
    Apply class balance controls.
    """
    print(f"\n{'='*60}")
    print("STAGE 4: Merge & balance")
    print(f"{'='*60}")

    # Load original dataset
    original = pd.read_csv(ORIGINAL_DATASET, sep="\t", encoding="utf-8")
    original["domain"] = "foreign"
    original["site"] = "ec_darkpattern"
    original["step_label"] = "unknown"
    original["confidence"] = 1.0  # ground truth
    original["source_type"] = "original"

    # Add empty structural feature columns for original rows
    feat_cols = [c for c in scraped_accepted.columns if c.startswith("feat_")]
    for col in feat_cols:
        if col not in original.columns:
            original[col] = np.nan

    # Drop raw element columns from scraped data (keep derived features only)
    el_cols = ["el_visible", "el_width", "el_height", "el_x", "el_y", "el_tag",
               "source_file"]
    scraped_clean = scraped_accepted.drop(columns=el_cols, errors="ignore")

    # Align columns
    keep_cols = ["text", "label", "Pattern Category", "domain", "site",
                 "step_label", "confidence", "source_type"] + feat_cols
    original_aligned = original[[c for c in keep_cols if c in original.columns]]
    scraped_aligned = scraped_clean[[c for c in keep_cols if c in scraped_clean.columns]]

    # ── Class balance control ──
    # Cap scraped benign samples
    n_original_benign = (original["label"] == 0).sum()
    max_scraped_benign = int(n_original_benign * MAX_BENIGN_RATIO)

    scraped_benign = scraped_aligned[scraped_aligned["label"] == 0]
    scraped_dark = scraped_aligned[scraped_aligned["label"] == 1]

    if len(scraped_benign) > max_scraped_benign:
        print(f"  Capping scraped benign: {len(scraped_benign)} -> {max_scraped_benign} "
              f"(max {MAX_BENIGN_RATIO}x original {n_original_benign})")
        scraped_benign = scraped_benign.sample(n=max_scraped_benign, random_state=42)

    scraped_balanced = pd.concat([scraped_benign, scraped_dark], ignore_index=True)

    # Merge
    combined = pd.concat([original_aligned, scraped_balanced], ignore_index=True)

    # Apply manual overrides
    combined = apply_manual_overrides(combined)

    print(f"\n  Final dataset:")
    print(f"    Total rows: {len(combined)}")
    print(f"    Original: {(combined['domain'] == 'foreign').sum()}")
    print(f"    Scraped (accepted): {(combined['domain'] == 'indian').sum()}")
    print(f"    Label distribution:")
    for label, count in combined["label"].value_counts().items():
        print(f"      {label}: {count}")
    print(f"    Category distribution:")
    for cat, count in combined["Pattern Category"].value_counts().items():
        print(f"      {cat}: {count}")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — CCPA Mapping
# ─────────────────────────────────────────────────────────────────────────────

def add_ccpa_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 5: Map Mathur categories to CCPA 2023 Guidelines categories.
    Applied only to domain='indian' rows, as the paper specifies.
    """
    print(f"\n{'='*60}")
    print("STAGE 5: CCPA mapping")
    print(f"{'='*60}")

    if not CCPA_MAPPING_PATH.exists():
        print(f"  WARNING: {CCPA_MAPPING_PATH} not found, skipping CCPA mapping")
        df["ccpa_category"] = None
        return df

    mapping = pd.read_csv(CCPA_MAPPING_PATH, sep="\t")
    # Build lookup: Mathur category → primary CCPA category (take first match)
    ccpa_lookup = {}
    for _, row in mapping.iterrows():
        mathur = row["mathur_category"]
        if mathur not in ccpa_lookup:
            ccpa_lookup[mathur] = row["ccpa_category"]

    df["ccpa_category"] = None

    indian_mask = df["domain"] == "indian"
    for idx in df[indian_mask].index:
        cat = df.at[idx, "Pattern Category"]
        if cat in ccpa_lookup:
            df.at[idx, "ccpa_category"] = ccpa_lookup[cat]

    n_mapped = df["ccpa_category"].notna().sum()
    print(f"  Mapped {n_mapped} Indian rows to CCPA categories")
    if n_mapped > 0:
        print(f"  CCPA distribution:")
        for cat, count in df["ccpa_category"].value_counts().items():
            print(f"    {cat}: {count}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DARK PATTERN DATA PIPELINE")
    print("=" * 60)

    # Ensure output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Extract
    scraped_df = extract_all_snippets()
    if scraped_df.empty:
        print("No scraped data found. Exiting.")
        return

    # Stage 2: Pseudo-label
    accepted_df, review_df = pseudo_label(scraped_df)

    # Save needs-review list
    review_df.to_csv(NEEDS_REVIEW_OUTPUT, sep="\t", index=False)
    print(f"\n  Saved {len(review_df)} low-confidence snippets to {NEEDS_REVIEW_OUTPUT}")

    # Stage 3: Structural features
    accepted_df = add_structural_features(accepted_df)

    # Stage 4: Merge & balance
    combined = merge_and_balance(accepted_df)

    # Stage 5: CCPA mapping
    combined = add_ccpa_mapping(combined)

    # ── Handle rare categories ──
    cat_counts = combined[combined["label"] == 1]["Pattern Category"].value_counts()
    rare_cats = [cat for cat, count in cat_counts.items() if count < RARE_CATEGORY_MIN_SAMPLES]
    if rare_cats:
        print(f"\n  NOTE: Rare categories (< {RARE_CATEGORY_MIN_SAMPLES} samples) "
              f"will be merged to 'Other' during multi-class training: {rare_cats}")
        # Don't modify the dataset — let train_model.py handle this at training time
        # Just flag it here for visibility

    # Save final dataset
    combined.to_csv(COMBINED_OUTPUT, sep="\t", index=False)
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Combined dataset: {COMBINED_OUTPUT} ({len(combined)} rows)")
    print(f"  Needs review:     {NEEDS_REVIEW_OUTPUT} ({len(review_df)} rows)")

    # Write stats summary
    stats = []
    stats.append(f"Pipeline run summary")
    stats.append(f"{'='*40}")
    stats.append(f"Sites processed: {scraped_df['site'].nunique()}")
    stats.append(f"Raw scraped snippets (post-dedup): {len(scraped_df)}")
    stats.append(f"Accepted (above confidence): {len(accepted_df)}")
    stats.append(f"Needs review (below confidence): {len(review_df)}")
    stats.append(f"Final combined rows: {len(combined)}")
    stats.append(f"  - Foreign (original): {(combined['domain'] == 'foreign').sum()}")
    stats.append(f"  - Indian (scraped): {(combined['domain'] == 'indian').sum()}")
    stats.append(f"Label balance: {dict(combined['label'].value_counts())}")
    stats.append(f"Category balance: {dict(combined['Pattern Category'].value_counts())}")
    if rare_cats:
        stats.append(f"Rare categories flagged: {rare_cats}")

    with open(STATS_OUTPUT, "w") as f:
        f.write("\n".join(stats))
    print(f"  Stats:            {STATS_OUTPUT}")


if __name__ == "__main__":
    main()
