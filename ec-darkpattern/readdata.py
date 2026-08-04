import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load data FIRST
df = pd.read_csv('dataset/dataset.tsv', sep='\t')

# 2. Then split it
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 3. Then vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Then train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# 5. Then evaluate
preds = clf.predict(X_test_tfidf)
print(classification_report(y_test, preds))

import joblib

joblib.dump(clf, 'baseline_model.pkl')
joblib.dump(vectorizer, 'baseline_vectorizer.pkl')
print("Model and vectorizer saved.")



results = pd.DataFrame({
    'text': X_test,
    'actual': y_test,
    'predicted': preds
})

false_negatives = results[(results['actual'] == 1) & (results['predicted'] == 0)]
false_positives = results[(results['actual'] == 0) & (results['predicted'] == 1)]

print(f"\n--- False Negatives ({len(false_negatives)}) — real dark patterns it missed ---")
print(false_negatives['text'].to_string(index=False))

print(f"\n--- False Positives ({len(false_positives)}) — flagged wrongly ---")
print(false_positives['text'].to_string(index=False))