# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Text-classification research project that detects e-commerce dark patterns (Mathur et al. taxonomy: Scarcity, Urgency, Social Proof, Misdirection, Obstruction, Sneaking, Forced Action) on Indian shopping sites, and maps them to India's CCPA 2023 dark-pattern guideline categories. It extends the `ec-darkpattern/` dataset (foreign sites, ground truth) with pseudo-labeled text scraped from Indian sites. All models are TF-IDF (1–2 grams, 5000 features) plus a classical classifier.

There is no test suite, linter, or build step. All scripts use paths relative to the repo root, so **run them from the repo root**.

## Commands

```bash
# Capture a new site (interactive: opens headed Chromium, you navigate and press Enter at each step).
# Uncomment/add the site in the `sites` dict at the bottom of scraper.py first.
python scraper.py

# Rebuild data/processed/combined_dataset.tsv from data/raw/
python prepare_dataset.py

# Train binary + multiclass models (runs prepare_dataset first if --prepare or the dataset is missing)
python train_model.py [--prepare]

# Explainability / error analysis (require the saved model to be linear, i.e. have coef_)
python explain_model.py          # global top features from LR coefficients
python explain_prediction.py     # per-text SHAP explanations on sample strings
python error_analysis_shap.py    # SHAP on binary model's FP/FN from the held-out split

# Web app (FastAPI backend + static HTML frontend)
cd dark-pattern-webapp/backend
pip install -r requirements.txt && playwright install chromium
uvicorn server:app --reload      # http://127.0.0.1:8000; then open frontend/index.html
```

Optional training deps are detected at import time: `xgboost`, `lightgbm`, `optuna` (without Optuna, models train with default hyperparameters). SHAP scripts need `shap`. There is no top-level requirements file.

## Git workflow

Commit and push regularly so no work is lost. This is standing authorization: you don't need to ask first.

- After each logical unit of work (a feature, a fix, a pipeline rerun, a retrain), commit it and push to `origin` (GitHub: `idfcLeo/dark-pattern-detector`). Don't let changes pile up uncommitted across tasks.
- Work on the current feature branch (currently `scrapeR`). Don't commit directly to `main`; it's updated by merging the feature branch.
- Commit messages: a short imperative subject (≤ ~72 chars) saying what changed and why, e.g. `Add XGBoost comparison, fix domain-eval leakage`. Add a body only when the change needs explaining. Use one commit per logical change; don't bundle unrelated edits.
- When a pipeline or training run changes `data/processed/` or `models/`, commit the regenerated artifacts together with the code change that produced them, and put the headline numbers (rows, F1) in the message.
- Stage specific files, not `git add -A`, and check `git status` first so stray files (venvs, `__pycache__`, screenshots you didn't mean to add, credentials) stay out.
- Never force-push or rewrite pushed history. If a push is rejected, `git pull --rebase` and resolve the conflicts. If a pre-commit hook fails, fix the cause.

## Architecture

**Data flow:** `scraper.py` → `data/raw/<Site>/` → `prepare_dataset.py` → `data/processed/combined_dataset.tsv` → `train_model.py` → `models/*.pkl` → consumed by the explain scripts and the web backend.

- **`scraper.py`** writes, per flow step, `stepNN_<label>_<YYYYMMDD_HHMMSS>.{png,txt,json}` into `data/raw/<site_name>/`. The step label (landing, product, cart, checkout, payment, settings, cancel_step) is parsed back out of the filename later, so keep that naming format. The `.json` holds clickable elements with bounding boxes.

- **`prepare_dataset.py`** (5 stages, constants at top of file):
  1. Extract lines from `.txt` and element texts from `.json`, filter junk, dedup within and then across sites. Scraped rows get `domain="indian"`.
  2. Pseudo-label: binary labels come from the **baseline** model `ec-darkpattern/baseline_model.pkl` (produced by `ec-darkpattern/readdata.py`), not from `models/`. Categories come from an interim LR trained on the original dataset's dark rows. Rows below the confidence thresholds go to `needs_review.tsv` instead of the dataset.
  3. Structural `feat_*` columns from element metadata/step label (NaN for original rows).
  4. Merge with `ec-darkpattern/dataset/dataset.tsv` (`domain="foreign"`), cap scraped benign rows at `MAX_BENIGN_RATIO`× original benign, then apply `data/manual_labels.tsv` overrides (exact text match → label + category).
  5. CCPA mapping from `configs/ccpa_mapping.tsv` (Indian rows only, first match per Mathur category).

- **`train_model.py`** trains LR, RF, and optionally LightGBM/XGBoost for each task with Optuna CV tuning, picks the best by test F1, and saves `models/{binary,multiclass}_{model,vectorizer}.pkl`, `model_comparison.json`, and `training_report.txt`. Metrics are reported separately for foreign / indian / combined test subsets. Rare categories (< `RARE_CATEGORY_MIN_SAMPLES`) are merged to "Other" at training time, not in the dataset. XGBoost needs integer-encoded labels, which is handled inside the script.

- **Split coupling:** `error_analysis_shap.py` rebuilds the binary test set with the same `test_size=0.2`, `random_state=42`, `stratify=label` split over `combined_dataset.tsv` after `dropna(subset=["text"])`. If you change how `train_model.py` loads or splits data, update `rebuild_test_set` to match, or the analysis will no longer run on held-out data.

- **Linear-only explainability:** the SHAP and coefficient scripts use `shap.LinearExplainer` / `coef_` and skip (with a warning) if model selection saved a tree model.

- **Web app** (`dark-pattern-webapp/`): `server.py` loads models from `<repo>/models` and the CCPA mapping from `<repo>/configs` (override the root with the `DP_ROOT` env var). The `.pkl` files inside `backend/` and the webapp README's instructions to copy baseline pickles there are outdated. Endpoints: `GET /health`, `POST /predict` (single text), `POST /analyze-url` (headless Playwright loads the page, takes leaf-element text chunks of 3–300 chars, classifies each). The multiclass model and CCPA tags are only added for texts predicted as dark. The frontend hard-codes `API_URL = "http://127.0.0.1:8000"`.

- `ec-darkpattern/` is the vendored upstream dataset/research repo (its own Poetry projects and a TS scraper). Treat it as read-only input except for `readdata.py` and the baseline pickles.
