"""
train_model.py — Dark Pattern Model Training

Trains both binary and multi-class models on the combined dataset.

Candidates per task: LR, calibrated LinearSVC, RF, LightGBM, XGBoost (each
Optuna-tuned), plus soft-voting and stacking ensembles of the top base models.

Evaluation protocol (no test-set leakage):
  - 80/20 stratified train/test split (same split error_analysis_shap.py rebuilds)
  - TF-IDF features are fit on training folds only, never on test data
  - Model selection uses 5-fold CV on the training set only
    (binary: F1 on the dark class, multi-class: macro F1)
  - The test set is touched once, for the final report
Metrics are also reported separately by domain (foreign vs. Indian).
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

import prepare_dataset

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    accuracy_score,
    precision_score,
    recall_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COMBINED_DATASET = Path("data/processed/combined_dataset.tsv")
MODELS_DIR = Path("models")
REPORT_PATH = Path("models/training_report.txt")

# Rare category threshold — merge into "Other" if below this
RARE_CATEGORY_MIN_SAMPLES = 15

# Training params
N_FOLDS = 5
N_OPTUNA_TRIALS = 30
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000       # word 1-2 grams
CHAR_TFIDF_MAX_FEATURES = 20000  # char 2-5 grams (catches "hurry!!", "5 left", typos)
ENSEMBLE_TOP_K = 3              # how many best base models go into the ensembles
SLOW_MODELS = {"rf", "lgbm", "xgb"}  # get a third of the trials (one XGB fit is ~10s)
# Binary decision threshold, picked to maximise F-beta on out-of-fold CV
# probabilities (never on test or gold). beta=2 weighs recall twice as much as
# precision: for a consumer-protection tool a missed dark pattern costs more
# than a false alarm. The web backend reads the saved threshold.
THRESHOLD_BETA = 2.0
THRESHOLD_PATH = MODELS_DIR / "binary_threshold.json"
# Deploy the CV-tuned threshold, or plain 0.50? The tuned value comes from
# mostly-foreign CV data and has scored worse on the Indian gold set (0.40 and
# 0.29 both lost to 0.50), so 0.50 ships. The tuned value is still computed,
# saved as "tuned_threshold" and scored on gold, so this can be revisited.
USE_TUNED_THRESHOLD = False
# Gold-set confidence intervals: stratified bootstrap resamples (rows drawn with
# replacement within each wave+stratum, keeping their sampling weights).
GOLD_BOOTSTRAP_REPS = 2000


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load the combined dataset and validate required columns."""
    df = pd.read_csv(COMBINED_DATASET, sep="\t", encoding="utf-8")

    required = ["text", "label", "Pattern Category", "domain"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    print(f"Loaded {len(df)} rows from {COMBINED_DATASET}")
    print(f"  Domains: {dict(df['domain'].value_counts())}")
    print(f"  Labels: {dict(df['label'].value_counts())}")
    return df


def handle_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rare categories (< RARE_CATEGORY_MIN_SAMPLES) into 'Other'.
    Only affects dark-pattern rows (label=1).
    """
    cat_counts = df[df["label"] == 1]["Pattern Category"].value_counts()
    rare = [cat for cat, count in cat_counts.items() if count < RARE_CATEGORY_MIN_SAMPLES]

    if rare:
        print(f"\n  Merging rare categories into 'Other': {rare}")
        print(f"    (each had < {RARE_CATEGORY_MIN_SAMPLES} samples)")
        df.loc[df["Pattern Category"].isin(rare), "Pattern Category"] = "Other"
    return df


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------

def build_vectorizer() -> FeatureUnion:
    """Word 1-2 grams + character 2-5 grams, both sublinear TF-IDF."""
    return FeatureUnion([
        ("word", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2),
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 max_features=CHAR_TFIDF_MAX_FEATURES, min_df=2,
                                 sublinear_tf=True)),
    ])


def make_cv_folds(texts: pd.Series, y_enc: np.ndarray) -> list:
    """
    Pre-vectorize each CV fold once, fitting TF-IDF on that fold's training part
    only, so tuning never sees validation-fold vocabulary. Reused by every trial.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for tr, va in skf.split(texts, y_enc):
        vec = build_vectorizer()
        X_tr = vec.fit_transform(texts.iloc[tr])
        X_va = vec.transform(texts.iloc[va])
        folds.append((X_tr, X_va, y_enc[tr], y_enc[va], va))
    return folds


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def available_model_types() -> list:
    types = ["lr", "svc", "rf"]
    if HAS_LIGHTGBM:
        types.append("lgbm")
    if HAS_XGBOOST:
        types.append("xgb")
    return types


def suggest_params(trial, model_type: str) -> dict:
    """Optuna search space per model type."""
    if model_type == "lr":
        return {"C": trial.suggest_float("C", 1e-2, 100.0, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if model_type == "svc":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if model_type == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 8, 64),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])}
    if model_type == "lgbm":
        return {"num_leaves": trial.suggest_int("num_leaves", 8, 128),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if model_type == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)}
    raise ValueError(model_type)


def build_model(model_type: str, params: dict):
    """Instantiate an (unfitted) estimator. Every estimator exposes predict_proba."""
    if model_type == "lr":
        return LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, **params)
    if model_type == "svc":
        # LinearSVC has no predict_proba; calibration adds it (needed by the
        # web backend's confidence score and by soft voting / stacking).
        return CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE, **params), cv=3)
    if model_type == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_type == "lgbm":
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)
    if model_type == "xgb":
        return xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", **params)
    raise ValueError(model_type)


def score(y_true, y_pred, n_classes: int) -> float:
    """Selection metric: F1 of the dark class (binary) or macro F1 (multi-class)."""
    return f1_score(y_true, y_pred, average="binary" if n_classes == 2 else "macro", zero_division=0)


def oof_proba(model_type: str, params: dict, folds: list, n_rows: int, n_classes: int) -> np.ndarray:
    """Out-of-fold class probabilities over the training set (labels integer-encoded)."""
    proba = np.zeros((n_rows, n_classes))
    for X_tr, X_va, y_tr, _, va in folds:
        model = build_model(model_type, params)
        model.fit(X_tr, y_tr)
        proba[va] = model.predict_proba(X_va)
    return proba


def tune(model_type: str, folds: list, y_enc: np.ndarray, n_classes: int, n_trials: int):
    """Optuna-tune one model type on the CV folds. Returns (best_params, oof_proba, cv_score)."""
    if model_type in SLOW_MODELS and n_trials > 0:
        n_trials = max(3, n_trials // 3)
    print(f"\n  Tuning {model_type.upper()} ({n_trials} trials x {N_FOLDS} folds)...", flush=True)

    if HAS_OPTUNA and n_trials > 0:
        def objective(trial):
            params = suggest_params(trial, model_type)
            scores = []
            for step, (X_tr, X_va, y_tr, y_va, _) in enumerate(folds):
                model = build_model(model_type, params)
                model.fit(X_tr, y_tr)
                scores.append(score(y_va, model.predict(X_va), n_classes))
                # Stop clearly-worse trials early instead of running all folds
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
    else:
        print("    (Optuna not available, using defaults)")
        best_params = {}

    proba = oof_proba(model_type, best_params, folds, len(y_enc), n_classes)
    cv = score(y_enc, proba.argmax(axis=1), n_classes)
    print(f"    CV score: {cv:.4f}   params: {best_params}")
    return best_params, proba, cv


# -----------------------------------------------------------------------------
# Test-set Evaluation
# -----------------------------------------------------------------------------

def test_metrics(model, X_test, y_test, n_classes: int) -> dict:
    preds = model.predict(X_test)
    avg = "binary" if n_classes == 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average=avg, zero_division=0)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, preds, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, preds, average=avg, zero_division=0)),
        "report": classification_report(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def evaluate_by_domain(model, vectorizer, df, label_col, n_classes: int) -> dict:
    """Evaluate a model separately for foreign, Indian, and combined test rows."""
    avg = "binary" if n_classes == 2 else "weighted"
    results = {}
    for domain_name in ["foreign", "indian", "combined"]:
        subset = df if domain_name == "combined" else df[df["domain"] == domain_name]
        if len(subset) < 5:
            print(f"    {domain_name}: too few samples ({len(subset)}), skipping")
            continue

        y_true = subset[label_col].values
        y_pred = model.predict(vectorizer.transform(subset["text"]))
        results[domain_name] = {
            "n_samples": len(subset),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        m = results[domain_name]
        print(f"    [{domain_name.upper()}] n={m['n_samples']}  acc={m['accuracy']:.4f}  "
              f"f1={m['f1']:.4f}  macro_f1={m['macro_f1']:.4f}")
    return results


# -----------------------------------------------------------------------------
# Per-task Training
# -----------------------------------------------------------------------------

def train_task(task_df: pd.DataFrame, label_col: str, train_idx, test_idx,
               task_name: str, file_prefix: str, n_trials: int) -> dict:
    """
    Tune all base models + build ensembles, select by CV on the training split,
    fit everything on the full training split, report on test, save the winner.
    """
    train_df = task_df.iloc[train_idx]
    test_df = task_df.iloc[test_idx]
    y_train = train_df[label_col].values
    y_test = test_df[label_col].values

    encoder = LabelEncoder().fit(y_train)
    y_train_enc = encoder.transform(y_train)
    n_classes = len(encoder.classes_)
    metric_name = "F1 (dark class)" if n_classes == 2 else "macro F1"

    print(f"\n  Train: {len(train_df)}  Test: {len(test_df)}  Classes: {list(encoder.classes_)}")
    print(f"  Selection metric: {N_FOLDS}-fold CV {metric_name} on the training split")

    folds = make_cv_folds(train_df["text"], y_train_enc)

    # ── Base models ──
    tuned = {}
    for mt in available_model_types():
        params, proba, cv = tune(mt, folds, y_train_enc, n_classes, n_trials)
        tuned[mt] = {"params": params, "oof": proba, "cv": cv}

    # ── Ensembles from the top-K base models (scored on the same OOF predictions) ──
    top = sorted(tuned, key=lambda m: tuned[m]["cv"], reverse=True)[:ENSEMBLE_TOP_K]
    print(f"\n  Ensemble members (top {ENSEMBLE_TOP_K} by CV): {[m.upper() for m in top]}")

    vote_proba = np.mean([tuned[m]["oof"] for m in top], axis=0)
    vote_cv = score(y_train_enc, vote_proba.argmax(axis=1), n_classes)

    meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    stack_features = np.hstack([tuned[m]["oof"] for m in top])
    meta_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    stack_proba = cross_val_predict(meta, stack_features, y_train_enc, cv=meta_cv, method="predict_proba")
    stack_cv = score(y_train_enc, stack_proba.argmax(axis=1), n_classes)
    oof = {mt: tuned[mt]["oof"] for mt in tuned}
    oof["voting"], oof["stacking"] = vote_proba, stack_proba

    print(f"    VOTING   CV score: {vote_cv:.4f}")
    print(f"    STACKING CV score: {stack_cv:.4f}")

    cv_scores = {mt: tuned[mt]["cv"] for mt in tuned}
    cv_scores["voting"] = vote_cv
    cv_scores["stacking"] = stack_cv

    # ── Fit every candidate on the full training split ──
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    members = [(m, build_model(m, tuned[m]["params"])) for m in top]
    candidates = {}
    for mt in tuned:
        if mt == "xgb":
            # XGBoost needs integer labels; a one-member VotingClassifier handles
            # the encoding internally so predict() returns the original labels.
            candidates[mt] = VotingClassifier([("xgb", build_model(mt, tuned[mt]["params"]))], voting="soft")
        else:
            candidates[mt] = build_model(mt, tuned[mt]["params"])
    candidates["voting"] = VotingClassifier([(n, clone(e)) for n, e in members], voting="soft")
    candidates["stacking"] = StackingClassifier(
        [(n, clone(e)) for n, e in members], final_estimator=clone(meta),
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
    )

    results = {}
    print(f"\n  {'model':10s} {'CV ' + metric_name:>18s} {'test F1':>9s} {'test macroF1':>13s} {'test acc':>9s}")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        m = test_metrics(model, X_test, y_test, n_classes)
        m["cv_score"] = float(cv_scores[name])
        m["params"] = tuned[name]["params"] if name in tuned else {"members": top}
        results[name] = m
        print(f"  {name:10s} {m['cv_score']:18.4f} {m['f1']:9.4f} {m['macro_f1']:13.4f} {m['accuracy']:9.4f}")

    # Selection is by CV only; the test numbers above are for reporting.
    best = max(results, key=lambda n: results[n]["cv_score"])
    print(f"\n  Selected: {best.upper()} (CV {metric_name}={results[best]['cv_score']:.4f})")
    print(f"\n  Test classification report:\n{results[best]['report']}")

    joblib.dump(candidates[best], MODELS_DIR / f"{file_prefix}_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / f"{file_prefix}_vectorizer.pkl")
    # Tuned LR on the same features, kept so the coefficient/SHAP scripts can
    # still explain predictions when the selected model is an ensemble.
    joblib.dump(candidates["lr"], MODELS_DIR / f"{file_prefix}_linear_model.pkl")
    print(f"  Saved: {MODELS_DIR / f'{file_prefix}_model.pkl'} (+ vectorizer, linear explainer model)")

    print(f"\n  Domain-separated test evaluation ({task_name}, {best.upper()}):")
    domain_results = evaluate_by_domain(candidates[best], vectorizer, test_df, label_col, n_classes)

    return {"best": best, "metric_name": metric_name, "results": results,
            "domains": domain_results, "ensemble_members": top,
            "model": candidates[best], "vectorizer": vectorizer,
            "oof_proba": oof[best], "y_train_enc": y_train_enc}


def tune_threshold(oof_proba: np.ndarray, y_enc: np.ndarray) -> dict:
    """Pick the dark-class threshold maximising F-beta on out-of-fold CV probabilities."""
    p_dark = oof_proba[:, 1]
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    fbeta = [fbeta_score(y_enc, (p_dark >= t).astype(int), beta=THRESHOLD_BETA, zero_division=0)
             for t in grid]
    best = int(np.argmax(fbeta))
    at_half = fbeta_score(y_enc, (p_dark >= 0.5).astype(int), beta=THRESHOLD_BETA, zero_division=0)
    result = {"threshold": float(grid[best]), "beta": THRESHOLD_BETA,
              "cv_fbeta_at_threshold": float(fbeta[best]), "cv_fbeta_at_0.5": float(at_half)}
    print(f"\n  Threshold (max CV F{THRESHOLD_BETA:g}): {result['threshold']:.2f} "
          f"(CV F{THRESHOLD_BETA:g} {at_half:.4f} at 0.50 -> {fbeta[best]:.4f})")
    return result


def predict_dark(model, X, threshold: float) -> np.ndarray:
    """Binary prediction at a custom threshold on P(dark)."""
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


# -----------------------------------------------------------------------------
# Gold Test Set (human-labelled Indian snippets, see active_learning.py)
# -----------------------------------------------------------------------------

def weighted_scores(y_true, y_pred, weights, average) -> dict:
    """Metrics with inverse-sampling weights = estimates for the full scraped pool."""
    kw = {"average": average, "zero_division": 0, "sample_weight": weights}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=weights)),
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
        "f1": float(f1_score(y_true, y_pred, **kw)),
    }


def gold_weights(gold: pd.DataFrame) -> np.ndarray:
    """
    Inverse-sampling weights, recomputed from the rows that are actually labelled.

    Each wave of the gold set (`active_learning.py gold`, then `gold --extend`)
    is its own stratified random sample, so inside one (wave, stratum) a labelled
    row stands for stratum_size / n_labelled_there rows of the scraped pool.
    Waves are then averaged in proportion to their labelled sizes. Recomputing,
    instead of trusting the stored `weight`, keeps the estimate right when rows
    are marked unsure or a new wave is only partly labelled. For a fully labelled
    single wave it equals the stored weight.
    """
    g = gold.assign(_wave=gold["wave"].replace("", "1") if "wave" in gold else "1")
    n_in_stratum = g.groupby(["_wave", "stratum"])["text"].transform("size")
    n_in_wave = g.groupby("_wave")["text"].transform("size")
    return (g["stratum_size"].astype(float) / n_in_stratum * n_in_wave / len(g)).values


def gold_groups(gold: pd.DataFrame) -> np.ndarray:
    wave = gold["wave"].replace("", "1") if "wave" in gold else pd.Series("1", index=gold.index)
    return (wave + "/" + gold["stratum"]).values


def weighted_prf(y, pred, w) -> tuple:
    """Weighted precision, recall, F1 for the dark class (fast, for the bootstrap)."""
    tp = w[(y == 1) & (pred == 1)].sum()
    fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap_gold(y, preds: dict, w, groups, reps: int = GOLD_BOOTSTRAP_REPS) -> dict:
    """
    Stratified bootstrap: redraw the gold set many times (with replacement,
    inside each wave+stratum) and recompute the metrics. The spread of those
    scores shows how much a result could move from sampling luck alone.
    All systems share each resample, so their differences are paired.
    Returns {system: array of shape (reps, 3) with P, R, F1}.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    members = [np.where(groups == g)[0] for g in np.unique(groups)]
    out = {name: np.empty((reps, 3)) for name in preds}
    for i in range(reps):
        idx = np.concatenate([rng.choice(m, size=len(m), replace=True) for m in members])
        for name, pred in preds.items():
            out[name][i] = weighted_prf(y[idx], pred[idx], w[idx])
    return out


def ci95(values) -> list:
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def evaluate_gold(binary: dict, multiclass: dict, rare_categories: list) -> dict:
    """
    Score the selected models — and the baseline that produced the Indian
    pseudo-labels — on the human-labelled gold set. Gold rows are never used
    for training or selection.

    Reports weighted P/R/F1 (estimates for the whole scraped pool) with 95%
    bootstrap confidence intervals, plain counts of dark rows caught and false
    alarms, and a paired comparison of the selected model against the baseline.
    """
    if not prepare_dataset.GOLD_TEST_PATH.exists():
        print(f"\n  No gold set ({prepare_dataset.GOLD_TEST_PATH}); "
              "create one with: python active_learning.py gold")
        return {}

    gold = prepare_dataset.read_label_file(prepare_dataset.GOLD_TEST_PATH)
    gold = gold[gold["label"].isin(["0", "1"])].reset_index(drop=True)
    if len(gold) == 0:
        print("\n  Gold set has no labels yet: python active_learning.py label gold")
        return {}

    y = gold["label"].astype(int).values
    w = gold_weights(gold)
    groups = gold_groups(gold)
    n_waves = len({g.split("/")[0] for g in groups})
    print(f"\n  Gold set: {len(gold)} labelled ({y.sum()} dark, {n_waves} wave(s)). "
          "Weighted = estimate for the whole scraped Indian pool.")
    print(f"  95% CIs from {GOLD_BOOTSTRAP_REPS} stratified bootstrap resamples.")

    # (name, model, vectorizer, threshold); None = the model's own predict()
    t = binary["threshold"]["threshold"]
    systems = [("selected @0.50", binary["model"], binary["vectorizer"], 0.5),
               (f"selected @{t:.2f} (CV-tuned)", binary["model"], binary["vectorizer"], t)]
    if prepare_dataset.BASELINE_MODEL_PATH.exists():
        systems.append(("baseline (pseudo-labeller)", joblib.load(prepare_dataset.BASELINE_MODEL_PATH),
                        joblib.load(prepare_dataset.BASELINE_VECTORIZER_PATH), None))

    preds = {}
    for name, model, vec, threshold in systems:
        X = vec.transform(gold["text"])
        preds[name] = (np.asarray(model.predict(X)).astype(int) if threshold is None
                       else predict_dark(model, X, threshold))
    boot = bootstrap_gold(y, preds, w, groups)

    results = {"n_labelled": int(len(gold)), "n_dark": int(y.sum()), "n_waves": n_waves,
               "bootstrap_reps": GOLD_BOOTSTRAP_REPS, "binary": {}}
    for name, pred in preds.items():
        counts = {"dark_caught": int(((y == 1) & (pred == 1)).sum()),
                  "dark_missed": int(((y == 1) & (pred == 0)).sum()),
                  "false_alarms": int(((y == 0) & (pred == 1)).sum())}
        results["binary"][name] = {
            "weighted": weighted_scores(y, pred, w, "binary"),
            "unweighted": weighted_scores(y, pred, None, "binary"),
            "ci95": {k: ci95(boot[name][:, i]) for i, k in enumerate(["precision", "recall", "f1"])},
            "counts": counts,
        }
        m, ci = results["binary"][name]["weighted"], results["binary"][name]["ci95"]
        print(f"    BINARY {name:28s} weighted P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} [95% CI {ci['f1'][0]:.3f}-{ci['f1'][1]:.3f}] acc={m['accuracy']:.3f}  "
              f"caught {counts['dark_caught']}/{int(y.sum())} dark, {counts['false_alarms']} false alarms")

    # Paired comparison: is the selected model really better than the baseline?
    base = "baseline (pseudo-labeller)"
    if base in boot:
        diff = boot["selected @0.50"][:, 2] - boot[base][:, 2]
        results["selected_vs_baseline_f1"] = {
            "diff": results["binary"]["selected @0.50"]["weighted"]["f1"]
                    - results["binary"][base]["weighted"]["f1"],
            "ci95": ci95(diff), "share_of_resamples_better": float((diff > 0).mean())}
        c = results["selected_vs_baseline_f1"]
        print(f"    selected @0.50 vs baseline: weighted F1 {c['diff']:+.3f} "
              f"[95% CI {c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f}], "
              f"better in {c['share_of_resamples_better']:.0%} of resamples")

    # Category accuracy on gold rows a human marked dark (same rare-merge as training)
    gold["_w"] = w
    dark = gold[gold["label"] == "1"].copy()
    classes = set(multiclass["model"].classes_)
    dark["Pattern Category"] = dark["Pattern Category"].where(
        ~dark["Pattern Category"].isin(rare_categories) | ("Other" not in classes), "Other")
    if len(dark):
        pred = multiclass["model"].predict(multiclass["vectorizer"].transform(dark["text"]))
        wd = dark["_w"].values
        results["multiclass"] = {
            "n": int(len(dark)),
            "weighted_accuracy": float(accuracy_score(dark["Pattern Category"], pred, sample_weight=wd)),
            "macro_f1": float(f1_score(dark["Pattern Category"], pred, average="macro", zero_division=0)),
            "report": classification_report(dark["Pattern Category"], pred, zero_division=0),
        }
        mc = results["multiclass"]
        print(f"    MULTI-CLASS on {mc['n']} human-marked dark rows: "
              f"weighted acc={mc['weighted_accuracy']:.3f} macroF1={mc['macro_f1']:.3f}")
    return results


def gold_report_section(gold: dict) -> list:
    if not gold:
        return ["\nGOLD TEST SET: not available yet (python active_learning.py gold, then label it)"]
    lines = [f"\nGOLD TEST SET (human-labelled Indian snippets): n={gold['n_labelled']}, "
             f"dark={gold['n_dark']}, waves={gold.get('n_waves', 1)}",
             "  Binary, weighted by inverse sampling probability (estimate for the full scraped pool).",
             f"  F1 CI = 95% stratified bootstrap ({gold.get('bootstrap_reps', '?')} resamples).",
             f"  {'system':30s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F1 95% CI':>13s} {'acc':>6s}"
             "   unweighted R / F1   caught / missed / false alarms"]
    for name, m in gold["binary"].items():
        wt, uw, ci, c = m["weighted"], m["unweighted"], m["ci95"]["f1"], m["counts"]
        lines.append(f"  {name:30s} {wt['precision']:6.3f} {wt['recall']:6.3f} {wt['f1']:6.3f} "
                     f"  {ci[0]:.3f}-{ci[1]:.3f} {wt['accuracy']:6.3f}   {uw['recall']:.3f} / {uw['f1']:.3f}"
                     f"       {c['dark_caught']} / {c['dark_missed']} / {c['false_alarms']}")
    if "selected_vs_baseline_f1" in gold:
        c = gold["selected_vs_baseline_f1"]
        lines.append(f"  selected @0.50 minus baseline, weighted F1: {c['diff']:+.3f} "
                     f"[95% CI {c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f}], "
                     f"better in {c['share_of_resamples_better']:.0%} of resamples")
    if "multiclass" in gold:
        mc = gold["multiclass"]
        lines.append(f"\n  Multi-class on {mc['n']} human-marked dark rows: weighted acc="
                     f"{mc['weighted_accuracy']:.3f}, macroF1={mc['macro_f1']:.3f}\n{mc['report']}")
    return lines


def report_section(title: str, task: dict) -> list:
    best, res = task["best"], task["results"]
    lines = [f"\n{title} — selected: {best.upper()}",
             f"  Selection: {N_FOLDS}-fold CV {task['metric_name']} on training split "
             f"(ensembles use top {ENSEMBLE_TOP_K}: {[m.upper() for m in task['ensemble_members']]})",
             f"\n  {'model':10s} {'CV':>8s} {'test F1':>9s} {'macroF1':>9s} {'acc':>8s}"]
    for name, m in sorted(res.items(), key=lambda kv: kv[1]["cv_score"], reverse=True):
        marker = "  <-- selected" if name == best else ""
        lines.append(f"  {name:10s} {m['cv_score']:8.4f} {m['f1']:9.4f} {m['macro_f1']:9.4f} "
                     f"{m['accuracy']:8.4f}{marker}")
    lines.append(f"\n  Test classification report ({best.upper()}):\n{res[best]['report']}")
    for domain, m in task["domains"].items():
        lines.append(f"  Domain [{domain}]: F1={m['f1']:.4f}, macroF1={m['macro_f1']:.4f}, "
                     f"Acc={m['accuracy']:.4f}, n={m['n_samples']}")
    return lines


# -----------------------------------------------------------------------------
# Main Training Pipeline
# -----------------------------------------------------------------------------

def gold_only():
    """
    Re-score the saved models on the gold set without retraining, e.g. after
    labelling more gold rows. Refuses if any gold text is still in the training
    dataset: the saved models may have trained on it (new gold rows only leave
    the training data on the next `train_model.py --prepare`).
    """
    df = load_dataset()
    leaked = prepare_dataset.load_gold_texts() & set(df["text"])
    if leaked:
        sys.exit(f"\n{len(leaked)} gold texts are still in {COMBINED_DATASET}, so the saved models "
                 "may have trained on them. Run python train_model.py --prepare first.")
    th = json.loads(THRESHOLD_PATH.read_text()) if THRESHOLD_PATH.exists() else {}
    binary = {"model": joblib.load(MODELS_DIR / "binary_model.pkl"),
              "vectorizer": joblib.load(MODELS_DIR / "binary_vectorizer.pkl"),
              "threshold": {"threshold": th.get("tuned_threshold", th.get("threshold", 0.5))}}
    multiclass = {"model": joblib.load(MODELS_DIR / "multiclass_model.pkl"),
                  "vectorizer": joblib.load(MODELS_DIR / "multiclass_vectorizer.pkl")}
    labels = df["label"].astype(int)
    rare = [cat for cat, n in df.loc[labels == 1, "Pattern Category"].value_counts().items()
            if n < RARE_CATEGORY_MIN_SAMPLES]
    print(f"\n{'=' * 60}\nGOLD TEST SET (saved models, no retraining)\n{'=' * 60}")
    evaluate_gold(binary, multiclass, rare)


def main():
    parser = argparse.ArgumentParser(description="Dark Pattern Model Training")
    parser.add_argument("--prepare", action="store_true",
                        help="Run data preparation pipeline (prepare_dataset.py) before "
                             "training to update dataset from data/raw")
    parser.add_argument("--trials", type=int, default=N_OPTUNA_TRIALS,
                        help=f"Optuna trials per model type (default {N_OPTUNA_TRIALS})")
    parser.add_argument("--gold-only", action="store_true",
                        help="score the saved models on the gold set and exit (no training, seconds)")
    args = parser.parse_args()

    if args.gold_only:
        gold_only()
        return

    print("=" * 60)
    print("DARK PATTERN MODEL TRAINING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.prepare or not COMBINED_DATASET.exists():
        print("\n[INFO] Running data preparation pipeline on data/raw...")
        prepare_dataset.main()
        print("\n[INFO] Data preparation complete. Proceeding to training...\n")

    df = load_dataset()

    # One stratified split shared by both tasks: the multi-class model is trained
    # and tested on the dark-pattern rows of the binary train/test sets, so no
    # test row is ever seen in training by either model.
    # error_analysis_shap.py rebuilds this exact split — keep them in sync.
    y_binary = df["label"].values.astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_binary
    )
    df = df.reset_index(drop=True)
    df["label"] = y_binary

    # =========================================================================
    # BINARY MODEL
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("BINARY MODEL (dark pattern vs. not)")
    print(f"{'=' * 60}")
    binary = train_task(df, "label", train_idx, test_idx, "Binary", "binary", args.trials)
    binary["threshold"] = tune_threshold(binary["oof_proba"], binary["y_train_enc"])
    tuned = binary["threshold"]["threshold"]
    deployed = tuned if USE_TUNED_THRESHOLD else 0.5
    # "threshold" is what the web backend reads; the tuned value is kept for reference.
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({**binary["threshold"], "threshold": deployed, "tuned_threshold": tuned}, f, indent=2)
    print(f"  Deployed threshold: {deployed:.2f} (USE_TUNED_THRESHOLD={USE_TUNED_THRESHOLD})")

    # =========================================================================
    # MULTI-CLASS MODEL
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("MULTI-CLASS MODEL (category prediction, dark-pattern rows only)")
    print(f"{'=' * 60}")

    rare_categories = [cat for cat, count in df[df["label"] == 1]["Pattern Category"].value_counts().items()
                       if count < RARE_CATEGORY_MIN_SAMPLES]
    mc_df = handle_rare_categories(df.copy())
    is_dark = (mc_df["label"] == 1).values
    test_mask = np.zeros(len(mc_df), dtype=bool)
    test_mask[test_idx] = True
    dark_df = mc_df[is_dark].reset_index(drop=True)
    dark_test_mask = test_mask[is_dark]

    print("\n  Category distribution (after rare-merge):")
    for cat, count in dark_df["Pattern Category"].value_counts().items():
        print(f"    {cat}: {count}")

    multiclass = train_task(dark_df, "Pattern Category",
                            np.where(~dark_test_mask)[0], np.where(dark_test_mask)[0],
                            "Multi-class", "multiclass", args.trials)

    # =========================================================================
    # GOLD TEST SET
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("GOLD TEST SET (human-labelled Indian snippets)")
    print(f"{'=' * 60}")
    gold = evaluate_gold(binary, multiclass, rare_categories)

    # =========================================================================
    # Save Report + Comparison
    # =========================================================================
    report_lines = [f"Training Report — {datetime.now().isoformat()}", "=" * 60]
    report_lines += report_section("BINARY MODEL", binary)
    th = binary["threshold"]
    report_lines.append(f"  Tuned threshold: {th['threshold']:.2f} (max CV F{th['beta']:g}: "
                        f"{th['cv_fbeta_at_0.5']:.4f} at 0.50 -> {th['cv_fbeta_at_threshold']:.4f}); "
                        f"deployed: {th['threshold'] if USE_TUNED_THRESHOLD else 0.5:.2f}")
    report_lines += report_section("MULTI-CLASS MODEL", multiclass)
    report_lines.append(f"  Rare categories merged to 'Other': {rare_categories}")
    report_lines += gold_report_section(gold)

    comparison = {
        "protocol": {
            "selection": f"{N_FOLDS}-fold CV on training split; test used only for reporting",
            "features": f"word 1-2gram TF-IDF ({TFIDF_MAX_FEATURES}) + char_wb 2-5gram TF-IDF "
                        f"({CHAR_TFIDF_MAX_FEATURES}), fit on training data only",
            "optuna_trials_per_model": args.trials,
            "optuna_trials_slow_models": max(3, args.trials // 3) if args.trials > 0 else 0,
        },
    }
    for key, task in [("binary", binary), ("multiclass", multiclass)]:
        comparison[key] = {
            "selected": task["best"],
            "selection_metric": task["metric_name"],
            "ensemble_members": task["ensemble_members"],
            "candidates": {n: {k: v for k, v in m.items() if k != "report"}
                           for n, m in task["results"].items()},
            "domains": task["domains"],
        }
    comparison["binary"]["threshold"] = binary["threshold"]
    comparison["gold"] = {k: v for k, v in gold.items() if k != "multiclass"}
    if "multiclass" in gold:
        comparison["gold"]["multiclass"] = {k: v for k, v in gold["multiclass"].items() if k != "report"}

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Binary model:      {MODELS_DIR / 'binary_model.pkl'} ({binary['best'].upper()})")
    print(f"  Multi-class model: {MODELS_DIR / 'multiclass_model.pkl'} ({multiclass['best'].upper()})")
    print(f"  Report:            {REPORT_PATH}")
    print(f"  Comparison:        {MODELS_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
