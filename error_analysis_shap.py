"""
error_analysis_shap.py — SHAP-based diagnosis of the binary model's
current false positives and false negatives.

Reproduces the exact train/test split train_model.py used, finds
misclassified examples, and explains each one with SHAP to check
whether the two failure modes reported in the paper (missed numeric
urgency cues, false positives on stock-status language) still hold
after Optuna tuning and the combined-dataset retrain.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split

from explain_prediction import load_background  # reuse existing background logic

MODELS_DIR = Path("models")
COMBINED_DATASET = Path("data/processed/combined_dataset.tsv")
RANDOM_STATE = 42
TOP_N_WORDS = 8
MAX_EXAMPLES_PER_TYPE = 10  # cap printed output; change if you want more


def rebuild_test_set(vectorizer):
    """
    Must match train_model.py's split exactly: same random_state,
    same stratify column, same test_size — otherwise this isn't
    actually the model's held-out set.
    """
    df = pd.read_csv(COMBINED_DATASET, sep="\t", encoding="utf-8")
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    y_binary = df["label"].values.astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_binary,
    )

    test_df = df.iloc[test_idx].reset_index(drop=True)
    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"].values

    return test_df, X_test, y_test


def explain_one(text, model, vectorizer, explainer, top_n=TOP_N_WORDS):
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
    return contributions[:top_n]


def main():
    model = joblib.load(MODELS_DIR / "binary_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")

    if not hasattr(model, "coef_"):
        print(f"[WARN] Binary model is {type(model).__name__}, not linear — SHAP LinearExplainer doesn't apply. Stopping.")
        return

    test_df, X_test, y_test = rebuild_test_set(vectorizer)
    preds = model.predict(X_test)

    test_df["true_label"] = y_test
    test_df["pred_label"] = preds

    false_negatives = test_df[(test_df["true_label"] == 1) & (test_df["pred_label"] == 0)]
    false_positives = test_df[(test_df["true_label"] == 0) & (test_df["pred_label"] == 1)]

    print(f"Total test set: {len(test_df)}")
    print(f"False negatives (missed dark patterns): {len(false_negatives)}")
    print(f"False positives (wrongly flagged): {len(false_positives)}")

    background = load_background(vectorizer)
    explainer = shap.LinearExplainer(model, background)

    print("\n" + "=" * 60)
    print(f"FALSE NEGATIVES (showing up to {MAX_EXAMPLES_PER_TYPE})")
    print("=" * 60)
    for _, row in false_negatives.head(MAX_EXAMPLES_PER_TYPE).iterrows():
        print(f'\nText: "{row["text"]}"  [domain={row["domain"]}]')
        for word, val in explain_one(row["text"], model, vectorizer, explainer):
            print(f"    {word:25s} {val:+.4f}")

    print("\n" + "=" * 60)
    print(f"FALSE POSITIVES (showing up to {MAX_EXAMPLES_PER_TYPE})")
    print("=" * 60)
    for _, row in false_positives.head(MAX_EXAMPLES_PER_TYPE).iterrows():
        print(f'\nText: "{row["text"]}"  [domain={row["domain"]}]')
        for word, val in explain_one(row["text"], model, vectorizer, explainer):
            print(f"    {word:25s} {val:+.4f}")


if __name__ == "__main__":
    main()