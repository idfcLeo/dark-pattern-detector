"""
train_model.py — Dark Pattern Model Training

Trains both binary and multi-class models on the combined dataset.
Evaluates with domain-separated metrics (foreign vs. Indian).
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

import prepare_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)

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
N_OPTUNA_TRIALS = 50
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000


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

    # Drop rows with missing text
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
    dark = df[df["label"] == 1]
    cat_counts = dark["Pattern Category"].value_counts()

    rare = [cat for cat, count in cat_counts.items() if count < RARE_CATEGORY_MIN_SAMPLES]

    if rare:
        print(f"\n  Merging rare categories into 'Other': {rare}")
        print(f"    (each had < {RARE_CATEGORY_MIN_SAMPLES} samples)")
        df.loc[df["Pattern Category"].isin(rare), "Pattern Category"] = "Other"

    return df


# -----------------------------------------------------------------------------
# Model Training with Optuna
# -----------------------------------------------------------------------------

def create_objective(X, y, model_type="lr", n_folds=N_FOLDS, scoring="f1_weighted"):
    """Create an Optuna objective function for hyperparameter tuning."""

    def objective(trial):
        if model_type == "lr":
            C = trial.suggest_float("C", 1e-4, 10.0, log=True)
            model = LogisticRegression(
                C=C, max_iter=1000, random_state=RANDOM_STATE
            )
        elif model_type == "rf":
            model = RandomForestClassifier(
                max_depth=trial.suggest_int("max_depth", 2, 32),
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                random_state=RANDOM_STATE
            )
        elif model_type == "lgbm" and HAS_LIGHTGBM:
            model = lgb.LGBMClassifier(
                num_leaves=trial.suggest_int("num_leaves", 8, 128),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                random_state=RANDOM_STATE,
                verbose=-1
            )
        else:
            return 0.0

        # Determine appropriate fold count
        if isinstance(y[0], (int, np.integer)):
            min_class_count = int(np.bincount(y).min())
        else:
            min_class_count = int(pd.Series(y).value_counts().min())
        actual_folds = min(n_folds, min_class_count)
        if actual_folds < 2:
            actual_folds = 2

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)

        try:
            scores = cross_validate(model, X, y, cv=skf, scoring=scoring)
            return scores["test_score"].mean()
        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.0

    return objective


def train_single_model(X_train, y_train, X_test, y_test, model_type, task_name):
    """Train a single model with Optuna tuning, return best model + metrics."""
    print(f"\n  Training {model_type.upper()} for {task_name}...")

    scoring = "f1" if len(np.unique(y_train)) == 2 else "f1_weighted"

    if HAS_OPTUNA:
        study = optuna.create_study(direction="maximize")
        objective = create_objective(X_train, y_train, model_type=model_type, scoring=scoring)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
        print(f"    Best CV {scoring}: {best_score:.4f}")
        print(f"    Best params: {best_params}")
    else:
        best_params = {}
        print("    (Optuna not available, using defaults)")

    # Retrain on full training set with best params
    if model_type == "lr":
        model = LogisticRegression(
            C=best_params.get("C", 1.0), max_iter=1000, random_state=RANDOM_STATE
        )
    elif model_type == "rf":
        model = RandomForestClassifier(
            max_depth=best_params.get("max_depth", 10),
            n_estimators=best_params.get("n_estimators", 200),
            min_samples_split=best_params.get("min_samples_split", 5),
            min_samples_leaf=best_params.get("min_samples_leaf", 2),
            random_state=RANDOM_STATE
        )
    elif model_type == "lgbm" and HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            num_leaves=best_params.get("num_leaves", 31),
            learning_rate=best_params.get("learning_rate", 0.1),
            n_estimators=best_params.get("n_estimators", 200),
            min_child_samples=best_params.get("min_child_samples", 20),
            reg_alpha=best_params.get("reg_alpha", 0.1),
            reg_lambda=best_params.get("reg_lambda", 0.1),
            random_state=RANDOM_STATE,
            verbose=-1
        )
    else:
        return None, {}

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="weighted" if len(np.unique(y_train)) > 2 else "binary"),
        "precision": precision_score(y_test, preds, average="weighted" if len(np.unique(y_train)) > 2 else "binary", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted" if len(np.unique(y_train)) > 2 else "binary", zero_division=0),
        "report": classification_report(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "best_params": best_params,
    }

    print(f"    Test accuracy: {metrics['accuracy']:.4f}")
    print(f"    Test F1: {metrics['f1']:.4f}")

    return model, metrics


# -----------------------------------------------------------------------------
# Domain-Separated Evaluation
# -----------------------------------------------------------------------------

def evaluate_by_domain(model, vectorizer, df, label_col, task_name):
    """
    Evaluate model separately for foreign, Indian, and combined domains.
    This is the domain-split evaluation the paper requires.
    """
    print(f"\n  {'-'*50}")
    print(f"  Domain-separated evaluation: {task_name}")
    print(f"  {'-'*50}")

    results = {}

    for domain_name in ["foreign", "indian", "combined"]:
        if domain_name == "combined":
            subset = df
        else:
            subset = df[df["domain"] == domain_name]

        if len(subset) < 5:
            print(f"    {domain_name}: too few samples ({len(subset)}), skipping")
            continue

        X = vectorizer.transform(subset["text"])
        y_true = subset[label_col].values
        y_pred = model.predict(X)

        avg = "weighted" if len(np.unique(y_true)) > 2 else "binary"
        metrics = {
            "n_samples": len(subset),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        }
        results[domain_name] = metrics

        print(f"\n    [{domain_name.upper()}] (n={metrics['n_samples']})")
        print(f"      Accuracy:  {metrics['accuracy']:.4f}")
        print(f"      F1:        {metrics['f1']:.4f}")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall:    {metrics['recall']:.4f}")

    return results


# -----------------------------------------------------------------------------
# Main Training Pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dark Pattern Model Training")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run data preparation pipeline (prepare_dataset.py) before training to update dataset from data/raw",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DARK PATTERN MODEL TRAINING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.prepare or not COMBINED_DATASET.exists():
        print("\n[INFO] Running data preparation pipeline on data/raw...")
        prepare_dataset.main()
        print("\n[INFO] Data preparation complete. Proceeding to training...\n")

    # Load data
    df = load_dataset()

    report_lines = []
    report_lines.append(f"Training Report — {datetime.now().isoformat()}")
    report_lines.append("=" * 60)

    # =======================================================================
    # BINARY MODEL
    # =======================================================================

    print(f"\n{'='*60}")
    print("BINARY MODEL (dark pattern vs. not)")
    print(f"{'='*60}")

    # Vectorize
    binary_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_all = binary_vectorizer.fit_transform(df["text"])
    y_binary = df["label"].values.astype(int)

    # Train/test split (stratified, preserving domain distribution)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_binary
    )

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]

    # Train models
    model_types = ["lr", "rf"]
    if HAS_LIGHTGBM:
        model_types.append("lgbm")

    best_binary_model = None
    best_binary_f1 = -1
    best_binary_type = None
    all_binary_metrics = {}

    for mt in model_types:
        model, metrics = train_single_model(X_train, y_train, X_test, y_test, mt, "binary")
        if model is not None:
            all_binary_metrics[mt] = metrics
            if metrics["f1"] > best_binary_f1:
                best_binary_f1 = metrics["f1"]
                best_binary_model = model
                best_binary_type = mt

    if best_binary_model is not None:
        print(f"\n  Best binary model: {best_binary_type.upper()} (F1={best_binary_f1:.4f})")
        print(f"\n  Classification Report:")
        print(all_binary_metrics[best_binary_type]["report"])

        # Save
        joblib.dump(best_binary_model, MODELS_DIR / "binary_model.pkl")
        joblib.dump(binary_vectorizer, MODELS_DIR / "binary_vectorizer.pkl")
        print(f"  Saved: {MODELS_DIR / 'binary_model.pkl'}")

        # Domain-separated evaluation
        binary_domain_results = evaluate_by_domain(
            best_binary_model, binary_vectorizer, df, "label", "Binary"
        )

        report_lines.append(f"\nBINARY MODEL ({best_binary_type.upper()})")
        report_lines.append(f"  Best F1: {best_binary_f1:.4f}")
        report_lines.append(f"  Classification Report:\n{all_binary_metrics[best_binary_type]['report']}")
        for domain, metrics in binary_domain_results.items():
            report_lines.append(f"  Domain [{domain}]: F1={metrics['f1']:.4f}, "
                              f"Acc={metrics['accuracy']:.4f}, n={metrics['n_samples']}")

    # =======================================================================
    # MULTI-CLASS MODEL (on dark-pattern rows only)
    # =======================================================================

    print(f"\n{'='*60}")
    print("MULTI-CLASS MODEL (category prediction, dark-pattern rows only)")
    print(f"{'='*60}")

    dark_df = df[df["label"] == 1].copy()
    dark_df = handle_rare_categories(dark_df)

    # Check minimum class size for stratified CV
    cat_counts = dark_df["Pattern Category"].value_counts()
    print(f"\n  Category distribution (after rare-merge):")
    for cat, count in cat_counts.items():
        print(f"    {cat}: {count}")

    # Categories with enough data
    min_count = cat_counts.min()
    if min_count < 2:
        print(f"  WARNING: Category with only {min_count} sample(s) — may cause CV issues")

    # Vectorize
    mc_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_dark = mc_vectorizer.fit_transform(dark_df["text"])
    y_mc = dark_df["Pattern Category"].values

    # Train/test split
    train_idx_mc, test_idx_mc = train_test_split(
        np.arange(len(dark_df)), test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_mc
    )

    X_train_mc, X_test_mc = X_dark[train_idx_mc], X_dark[test_idx_mc]
    y_train_mc, y_test_mc = y_mc[train_idx_mc], y_mc[test_idx_mc]

    best_mc_model = None
    best_mc_f1 = -1
    best_mc_type = None
    all_mc_metrics = {}

    for mt in model_types:
        model, metrics = train_single_model(X_train_mc, y_train_mc, X_test_mc, y_test_mc, mt, "multi-class")
        if model is not None:
            all_mc_metrics[mt] = metrics
            if metrics["f1"] > best_mc_f1:
                best_mc_f1 = metrics["f1"]
                best_mc_model = model
                best_mc_type = mt

    if best_mc_model is not None:
        print(f"\n  Best multi-class model: {best_mc_type.upper()} (F1={best_mc_f1:.4f})")
        print(f"\n  Classification Report:")
        print(all_mc_metrics[best_mc_type]["report"])

        # Save
        joblib.dump(best_mc_model, MODELS_DIR / "multiclass_model.pkl")
        joblib.dump(mc_vectorizer, MODELS_DIR / "multiclass_vectorizer.pkl")
        print(f"  Saved: {MODELS_DIR / 'multiclass_model.pkl'}")

        # Domain-separated evaluation
        mc_domain_results = evaluate_by_domain(
            best_mc_model, mc_vectorizer, dark_df, "Pattern Category", "Multi-class"
        )

        report_lines.append(f"\nMULTI-CLASS MODEL ({best_mc_type.upper()})")
        report_lines.append(f"  Best F1 (weighted): {best_mc_f1:.4f}")
        report_lines.append(f"  Classification Report:\n{all_mc_metrics[best_mc_type]['report']}")
        report_lines.append(f"  Rare categories merged to 'Other': "
                          f"{[cat for cat, count in df[df['label']==1]['Pattern Category'].value_counts().items() if count < RARE_CATEGORY_MIN_SAMPLES]}")
        for domain, metrics in mc_domain_results.items():
            report_lines.append(f"  Domain [{domain}]: F1={metrics['f1']:.4f}, "
                              f"Acc={metrics['accuracy']:.4f}, n={metrics['n_samples']}")

    # =======================================================================
    # Save report
    # =======================================================================

    # Save all model comparison metrics
    comparison = {}
    for mt in model_types:
        comparison[mt] = {
            "binary": {k: v for k, v in all_binary_metrics.get(mt, {}).items()
                      if k not in ["report", "confusion_matrix"]}
            if mt in all_binary_metrics else None,
            "multiclass": {k: v for k, v in all_mc_metrics.get(mt, {}).items()
                          if k not in ["report", "confusion_matrix"]}
            if mt in all_mc_metrics else None,
        }

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Binary model:      {MODELS_DIR / 'binary_model.pkl'}")
    print(f"  Multi-class model: {MODELS_DIR / 'multiclass_model.pkl'}")
    print(f"  Report:            {REPORT_PATH}")
    print(f"  Comparison:        {MODELS_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
