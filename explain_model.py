"""
explain_model.py — Global Feature Importance for Dark Pattern Models

Loads the trained binary and multi-class models with their TF-IDF
vectorizers and extracts the words with the strongest positive/negative
influence on each class, directly from LogisticRegression coefficients.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODELS_DIR = Path("models")
TOP_N = 20

def _is_pure_stopword(feature_name, stopwords=ENGLISH_STOP_WORDS):
    """True if every word in this feature (unigram or bigram) is a stopword.
    A bigram like 'no thanks' is kept if at least one word carries meaning."""
    return all(w in stopwords for w in feature_name.split())

def top_features_binary(model, vectorizer, top_n=TOP_N):
    """
    Binary LogisticRegression coef_ has shape (1, n_features).
    Positive weight -> pushes toward class 1 (dark pattern).
    Negative weight -> pushes toward class 0 (not dark pattern).
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    ranked_dark = np.argsort(coefs)[::-1]
    ranked_benign = np.argsort(coefs)

    top_dark_idx = [
        idx for idx in ranked_dark
        if not _is_pure_stopword(feature_names[idx])
    ][:top_n]

    top_benign_idx = [
        idx for idx in ranked_benign
        if not _is_pure_stopword(feature_names[idx])
    ][:top_n]

    print("\n  Top words -> Dark Pattern (class 1):")
    for idx in top_dark_idx:
        print(f"    {feature_names[idx]:30s}  {coefs[idx]:+.4f}")

    print("\n  Top words -> Not Dark Pattern (class 0):")
    for idx in top_benign_idx:
        print(f"    {feature_names[idx]:30s}  {coefs[idx]:+.4f}")


def top_features_multiclass(model, vectorizer, top_n=TOP_N):
    """
    Multi-class LogisticRegression coef_ has shape (n_classes, n_features),
    one row per class in the order given by model.classes_.
    """
    feature_names = vectorizer.get_feature_names_out()

    for class_idx, class_name in enumerate(model.classes_):
        coefs = model.coef_[class_idx]
        ranked = np.argsort(coefs)[::-1]
        top_idx = [
            idx for idx in ranked
            if not _is_pure_stopword(feature_names[idx])
        ][:top_n]

        print(f"\n  Top words -> {class_name}:")
        for idx in top_idx:
            print(f"    {feature_names[idx]:30s}  {coefs[idx]:+.4f}")


def main():
    print("=" * 60)
    print("GLOBAL FEATURE IMPORTANCE (Logistic Regression coefficients)")
    print("=" * 60)

    binary_model = joblib.load(MODELS_DIR / "binary_model.pkl")
    binary_vectorizer = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")

    if not hasattr(binary_model, "coef_"):
        print(
            f"\n[WARN] Saved binary model is a {type(binary_model).__name__}, "
            "not linear — coefficients unavailable. This method only "
            "applies to LogisticRegression; a tree-based winner needs "
            "SHAP's TreeExplainer instead."
        )
    else:
        print("\nBINARY MODEL")
        top_features_binary(binary_model, binary_vectorizer)

    mc_model = joblib.load(MODELS_DIR / "multiclass_model.pkl")
    mc_vectorizer = joblib.load(MODELS_DIR / "multiclass_vectorizer.pkl")

    if not hasattr(mc_model, "coef_"):
        print(
            f"\n[WARN] Saved multi-class model is a {type(mc_model).__name__}, "
            "not linear — coefficients unavailable."
        )
    else:
        print("\nMULTI-CLASS MODEL")
        top_features_multiclass(mc_model, mc_vectorizer)


if __name__ == "__main__":
    main()