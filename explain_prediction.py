"""
explain_prediction.py — Local (per-prediction) explanations using SHAP.

For one piece of text, shows which specific words pushed the model's
binary prediction and, if flagged as a dark pattern, which words drove
the predicted CCPA-mappable category.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

MODELS_DIR = Path("models")
COMBINED_DATASET = Path("data/processed/combined_dataset.tsv")
BACKGROUND_SAMPLE_SIZE = 100
TOP_N = 10


def load_background(vectorizer, sample_size=BACKGROUND_SAMPLE_SIZE):
    """
    SHAP's LinearExplainer measures each word's contribution against a
    'typical' baseline rather than against zero. We use a random sample
    of existing text, vectorized the same way, as that baseline.
    """
    df = pd.read_csv(COMBINED_DATASET, sep="\t", encoding="utf-8")
    df = df.dropna(subset=["text"])
    sample = df["text"].sample(n=min(sample_size, len(df)), random_state=42)
    return vectorizer.transform(sample)


def explain_binary(text, model, vectorizer, explainer, top_n=TOP_N):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    shap_values = explainer(X)
    values = shap_values.values[0]

    present_idx = X.nonzero()[1]
    contributions = sorted(
        ((feature_names[i], values[i]) for i in present_idx),
        key=lambda pair: pair[1],
        reverse=True,
    )

    pred_label = model.predict(X)[0]
    pred_name = "Dark Pattern" if pred_label == 1 else "Not Dark Pattern"

    print(f"\n  Predicted: {pred_name}")
    print("  Words actually present, ranked by contribution:")
    for word, val in contributions[:top_n]:
        direction = "toward Dark Pattern" if val > 0 else "toward Not Dark Pattern"
        print(f"    {word:25s} {val:+.4f}  ({direction})")

    return contributions


def explain_multiclass(text, model, vectorizer, explainer, top_n=TOP_N):
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    pred_label = model.predict(X)[0]
    class_idx = list(model.classes_).index(pred_label)

    shap_values = explainer(X)
    raw = shap_values.values

    # Shape differs across SHAP versions for multi-output linear models;
    # print it once so we can confirm which branch actually applies here.
    print(f"  [debug] raw SHAP values shape: {raw.shape}")

    if raw.ndim == 3:
        values = raw[0, :, class_idx]
    else:
        values = raw[0]

    present_idx = X.nonzero()[1]
    contributions = sorted(
        ((feature_names[i], values[i]) for i in present_idx),
        key=lambda pair: pair[1],
        reverse=True,
    )

    print(f"\n  Predicted category: {pred_label}")
    print("  Words actually present, ranked by contribution:")
    for word, val in contributions[:top_n]:
        print(f"    {word:25s} {val:+.4f}")

    return contributions


def main():
    binary_model = joblib.load(MODELS_DIR / "binary_model.pkl")
    binary_vectorizer = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")

    mc_model = joblib.load(MODELS_DIR / "multiclass_model.pkl")
    mc_vectorizer = joblib.load(MODELS_DIR / "multiclass_vectorizer.pkl")

    sample_texts = [
        "Only 2 left in stock, order now!",
        "Hurry! Sale ends in 10 minutes.",
        "No thanks, I don't want to save money.",
    ]

    if hasattr(binary_model, "coef_"):
        print("=" * 60)
        print("BINARY MODEL — LOCAL EXPLANATIONS")
        print("=" * 60)
        background = load_background(binary_vectorizer)
        explainer = shap.LinearExplainer(binary_model, background)
        for text in sample_texts:
            print(f'\nText: "{text}"')
            explain_binary(text, binary_model, binary_vectorizer, explainer)
    else:
        print(f"[WARN] Binary model is {type(binary_model).__name__}, not linear — skipping.")

    if hasattr(mc_model, "coef_"):
        print("\n" + "=" * 60)
        print("MULTI-CLASS MODEL — LOCAL EXPLANATIONS")
        print("=" * 60)
        background = load_background(mc_vectorizer)
        explainer = shap.LinearExplainer(mc_model, background)
        for text in sample_texts:
            print(f'\nText: "{text}"')
            explain_multiclass(text, mc_model, mc_vectorizer, explainer)
    else:
        print(f"[WARN] Multi-class model is {type(mc_model).__name__}, not linear — skipping.")


if __name__ == "__main__":
    main()