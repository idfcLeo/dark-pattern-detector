# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Text-classification research project that detects e-commerce dark patterns (Mathur et al. taxonomy: Scarcity, Urgency, Social Proof, Misdirection, Obstruction, Sneaking, Forced Action) on Indian shopping sites, and maps them to India's CCPA 2023 dark-pattern guideline categories. It extends the `ec-darkpattern/` dataset (foreign sites, ground truth) with pseudo-labeled text scraped from Indian sites. Features are word 1–2-gram + char 2–5-gram TF-IDF (a `FeatureUnion`) feeding classical classifiers or ensembles of them.

There is no test suite, linter, or build step. All scripts use paths relative to the repo root, so **run them from the repo root**.

## Current phase (update as it changes)

**Phase 2: human labelling for Indian data.** Paused 2026-10-17, waiting on the user.

- Done:
  - Phase 1: leakage-free ensemble training with CV-based selection. Multiclass macro F1 went from 0.76 to 0.84, binary F1 stayed at 0.949. Both winners are SVC.
  - The 300-row gold set and `batch_01.tsv` have been generated, and the models were retrained without gold rows.
- In progress (the user does this in their own terminal): `label gold`, then `label data/active_learning/batch_01.tsv`.
- Next, once labels exist:
  1. Run `python active_learning.py status`.
  2. Run `python train_model.py --prepare` (about 17 min, run it in the background).
  3. Report the gold-set result: selected model vs. baseline pseudo-labeller, weighted P/R/F1. This is the project's headline novelty result.
  4. Commit, then `select` the next batch and repeat while the gold scores keep improving.
- Later options discussed:
  - Iterative self-training on `needs_review.tsv`.
  - A context-aware hybrid model (text + `feat_*` + journey step), evaluated on the gold set.
  - Inter-annotator agreement (a second labeller on about 50 gold rows, Cohen's kappa).

Labelling perspective agreed with the user: judge the text as an ordinary Indian shopper would see it, and label the type of claim, not whether it's true (Mathur et al. convention). A plain discount or rating is `0`; a discount with a deadline is Urgency. When several patterns appear, pick the most manipulative one. Keep `unsure` under about 10%.

## Commands

```bash
# Capture a new site (interactive: opens headed Chromium, you navigate and press Enter at each step).
# Uncomment/add the site in the `sites` dict at the bottom of scraper.py first.
python scraper.py

# Rebuild data/processed/combined_dataset.tsv from data/raw/
python prepare_dataset.py

# Train binary + multiclass models (runs prepare_dataset first if --prepare or the dataset is missing).
# Full run ≈ 15–20 min on CPU; --trials 0 skips Optuna (defaults) for a ~5 min smoke test.
python train_model.py [--prepare] [--trials N]

# Human labelling (interactive — the user runs `label` themselves; never fill labels in for them)
python active_learning.py gold                 # one-time stratified held-out gold test set
python active_learning.py label gold           # or: label data/active_learning/batch_NN.tsv
python active_learning.py select [--n 100]     # next active-learning batch from needs_review.tsv
python active_learning.py status

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

## Coding & learning instructions

The user is learning while building this project. Whenever you make changes, briefly explain:

- **WHAT:** what you changed or implemented.
- **WHY:** why the change was necessary and the reasoning behind your approach.
- **HOW:** how it works, including the relevant logic, functions, files, or technologies.
- **LEARNING:** the key concept the user should understand from this change.

Keep explanations concise, beginner-friendly, and technically accurate. Explain reasoning in terms of the problem, alternatives, and trade-offs.

Follow best practices, respect the existing project architecture, and avoid unnecessary changes. Never modify code without explaining the intended change. If something is unclear, ask before making risky assumptions.

**Goal:** help the user build working software while gradually improving their understanding of coding, system design, and problem-solving. Treat every code change as a small learning opportunity.

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

- **`train_model.py`** runs the same `train_task()` for both tasks. Candidates: LR, calibrated LinearSVC, RF, LightGBM, XGBoost (each Optuna-tuned with class weighting in the search space; RF/LGBM/XGB get a third of the trials and use median pruning), plus soft-voting and stacking ensembles of the top 3 by CV. Leakage rules the code depends on:
  - TF-IDF is fit per CV fold on that fold's training part (`make_cv_folds`) and finally on the training split only, never on test rows.
  - **Selection is by 5-fold CV on the training split** (binary: dark-class F1, multiclass: macro F1). Test metrics for every candidate are printed and saved but must not drive selection.
  - The multiclass train/test sets are the dark rows of the binary split, so neither model ever trains on a test row.

  Outputs: `models/{binary,multiclass}_{model,vectorizer,linear_model}.pkl`, `model_comparison.json`, `training_report.txt` (per-candidate CV vs. test table plus per-domain metrics). Saved models must stay plain sklearn objects because the web backend unpickles them without importing this repo. A winning XGBoost is wrapped in a one-member `VotingClassifier` so `predict()` returns the original labels. Rare categories (< `RARE_CATEGORY_MIN_SAMPLES`) are merged to "Other" at training time, not in the dataset.

- **Split coupling:** `error_analysis_shap.py` rebuilds the binary test set with the same `test_size=0.2`, `random_state=42`, `stratify=label` split over `combined_dataset.tsv` after `dropna(subset=["text"])`. If you change how `train_model.py` loads or splits data, update `rebuild_test_set` to match, or the analysis will no longer run on held-out data.

- **Explainability:** the SHAP and coefficient scripts load models through `explain_prediction.load_linear_model()`. If the deployed model isn't linear (an ensemble or SVC), it falls back to `{task}_linear_model.pkl` (the tuned LR on the same features) and says so. Feature names carry `word__` / `char__` prefixes from the `FeatureUnion`.

- **Human labels vs. pseudo-labels:** Indian rows are pseudo-labels from the baseline LR unless a human labelled them, so the Indian-domain *split* scores measure agreement with the baseline. Human ground truth lives in two places with different roles:
  - `data/gold/indian_gold_test.tsv` is **test only**. It's a stratified sample over bands of the binary model's P(dark), and each row stores `weight` = stratum_size / sampled so weighted metrics estimate the whole scraped pool. `prepare_dataset.py` drops gold texts from the scraped pool and from the final dataset. `train_model.py` scores the selected models and the baseline pseudo-labeller on it (`evaluate_gold`). Never tune or select on it. Regenerating it (`gold --force`) discards the user's labels.
  - `data/active_learning/batch_*.tsv` and `data/manual_labels.tsv` are **training** labels. `load_human_labels()` collects them, and `apply_manual_overrides()` replaces matching pseudo-labels or appends the rows as `source_type="human"` after benign capping.
  - In label files, blank `label` = not yet labelled and `unsure` = excluded. Read them with `read_label_file()` (`dtype=str`, `keep_default_na=False`) so blanks and texts like "NA" survive.

- **Web app** (`dark-pattern-webapp/`): `server.py` loads models from `<repo>/models` and the CCPA mapping from `<repo>/configs` (override the root with the `DP_ROOT` env var). The `.pkl` files inside `backend/` and the webapp README's instructions to copy baseline pickles there are outdated. Endpoints: `GET /health`, `POST /predict` (single text), `POST /analyze-url` (headless Playwright loads the page, takes leaf-element text chunks of 3–300 chars, classifies each). The multiclass model and CCPA tags are only added for texts predicted as dark. The frontend hard-codes `API_URL = "http://127.0.0.1:8000"`.

- `ec-darkpattern/` is the vendored upstream dataset/research repo (its own Poetry projects and a TS scraper). Treat it as read-only input except for `readdata.py` and the baseline pickles.
