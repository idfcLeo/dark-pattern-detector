"""
Fine-tune a Transformer (DistilRoBERTa / RoBERTa / BERT) for dark-pattern detection.

Experimental companion to train_model.py. It does NOT replace the deployed
TF-IDF models in models/*.pkl (the web backend still loads those); it trains a
Transformer on the same data and split and scores it next to them, so you can
decide whether it is worth deploying.

Leakage rules are the same as train_model.py:
  - Same stratified 80/20 split (TEST_SIZE, RANDOM_STATE, stratify=label).
  - The epoch is chosen on a validation slice of the TRAINING split only.
  - The gold set is only scored, never used for training or selection.

Usage (from the repo root):
    python train_transformer.py                         # binary, distilroberta-base
    python train_transformer.py --model roberta-base    # bigger, slower
    python train_transformer.py --model bert-base-uncased
    python train_transformer.py --task multiclass       # category model on dark rows
    python train_transformer.py --epochs 1 --max-train 300   # quick smoke test

Output: models/transformer_<task>/ (Hugging Face save_pretrained format plus
metrics.json). These folders are git-ignored: they are hundreds of MB.

Needs: pip install torch "transformers>=4.44,<5"
CPU works (distilroberta, binary, 3 epochs: roughly 15-30 min); a GPU is much faster.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import prepare_dataset
import train_model as tm

DEFAULT_MODEL = "distilroberta-base"
MAX_LENGTH = 64          # snippets are short; 64 tokens covers ~99% and keeps CPU training fast
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
VAL_SIZE = 0.1           # share of the training split held out to pick the best epoch


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(texts, labels, tokenizer, batch_size, shuffle, rng=None):
    """Yield tokenized batches. Padding is per batch (dynamic), which is faster than padding to MAX_LENGTH."""
    order = np.arange(len(texts))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor(labels[idx], dtype=torch.long)
        yield enc


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device) -> np.ndarray:
    model.eval()
    out = []
    for enc in batches(list(texts), None, tokenizer, EVAL_BATCH_SIZE, shuffle=False):
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out)


def task_score(y_true, proba, n_classes) -> float:
    """Same selection metric as train_model.py: dark-class F1 (binary) or macro F1."""
    return tm.score(y_true, proba.argmax(axis=1), n_classes)


def train(model_name, texts, y_enc, n_classes, epochs, device, seed):
    """
    Fine-tune on (texts, y_enc) minus a stratified validation slice; return the
    model from the epoch with the best validation score.
    """
    tr, va = train_test_split(np.arange(len(texts)), test_size=VAL_SIZE,
                              random_state=seed, stratify=y_enc)
    tr_texts, va_texts = [texts[i] for i in tr], [texts[i] for i in va]
    y_tr, y_va = y_enc[tr], y_enc[va]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes).to(device)

    # Class weights (inverse frequency) so the rarer class is not ignored,
    # the Transformer equivalent of class_weight="balanced".
    counts = np.bincount(y_tr, minlength=n_classes)
    class_w = torch.tensor(len(y_tr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = epochs * int(np.ceil(len(tr) / BATCH_SIZE))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * steps), steps)

    rng = np.random.default_rng(seed)
    best_score, best_state, best_epoch, history = -1.0, None, 0, []
    print(f"  Fine-tuning {model_name}: train {len(tr)}, validation {len(va)}, "
          f"{epochs} epoch(s), {steps} steps, device={device}")
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for enc in batches(tr_texts, y_tr, tokenizer, BATCH_SIZE, shuffle=True, rng=rng):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc.pop("labels")
            loss = loss_fn(model(**enc).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(labels)
            n += len(labels)
        val = task_score(y_va, predict_proba(model, tokenizer, va_texts, device), n_classes)
        history.append({"epoch": epoch, "train_loss": total / n, "val_score": val})
        print(f"    epoch {epoch}: loss {total / n:.4f}  validation score {val:.4f}  ({time.time() - t0:.0f}s)")
        if val > best_score:
            best_score, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch} (validation score {best_score:.4f})")
    return model, tokenizer, {"best_epoch": best_epoch, "val_score": best_score, "history": history}


def test_metrics(y_true, pred, n_classes) -> dict:
    return {"f1": float(tm.score(y_true, pred, n_classes)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "n_samples": int(len(y_true))}


# -----------------------------------------------------------------------------
# Gold set: Transformer vs. the deployed TF-IDF model vs. the baseline, same rows
# -----------------------------------------------------------------------------

def gold_binary(model, tokenizer, device) -> dict:
    if not prepare_dataset.GOLD_TEST_PATH.exists():
        print("  No gold set; skipping.")
        return {}
    gold = prepare_dataset.read_label_file(prepare_dataset.GOLD_TEST_PATH)
    gold = gold[gold["label"].isin(["0", "1"])].reset_index(drop=True)
    if not len(gold):
        return {}
    y, w, groups = gold["label"].astype(int).values, tm.gold_weights(gold), tm.gold_groups(gold)

    preds = {"transformer @0.50": (predict_proba(model, tokenizer, gold["text"], device)[:, 1] >= 0.5).astype(int)}
    for name, mpath, vpath in [
        ("tfidf deployed @0.50", tm.MODELS_DIR / "binary_model.pkl", tm.MODELS_DIR / "binary_vectorizer.pkl"),
        ("baseline (pseudo-labeller)", prepare_dataset.BASELINE_MODEL_PATH, prepare_dataset.BASELINE_VECTORIZER_PATH),
    ]:
        if Path(mpath).exists():
            X = joblib.load(vpath).transform(gold["text"])
            m = joblib.load(mpath)
            preds[name] = (tm.predict_dark(m, X, 0.5) if hasattr(m, "predict_proba")
                           else np.asarray(m.predict(X)).astype(int))

    boot = tm.bootstrap_gold(y, preds, w, groups)
    print(f"\n  Gold set: {len(gold)} labelled ({y.sum()} dark). Weighted = estimate for the scraped pool.")
    results = {"n_labelled": int(len(gold)), "n_dark": int(y.sum()), "systems": {}}
    for name, pred in preds.items():
        c = {"dark_caught": int(((y == 1) & (pred == 1)).sum()),
             "dark_missed": int(((y == 1) & (pred == 0)).sum()),
             "false_alarms": int(((y == 0) & (pred == 1)).sum())}
        wt = tm.weighted_scores(y, pred, w, "binary")
        ci = tm.ci95(boot[name][:, 2])
        results["systems"][name] = {"weighted": wt, "unweighted": tm.weighted_scores(y, pred, None, "binary"),
                                    "f1_ci95": ci, "counts": c}
        print(f"    {name:28s} weighted P={wt['precision']:.3f} R={wt['recall']:.3f} F1={wt['f1']:.3f} "
              f"[{ci[0]:.3f}-{ci[1]:.3f}]  caught {c['dark_caught']}/{int(y.sum())}, "
              f"{c['false_alarms']} false alarms")

    # Paired differences: same bootstrap resamples, so luck largely cancels out.
    for other in [n for n in preds if n != "transformer @0.50"]:
        diff = boot["transformer @0.50"][:, 2] - boot[other][:, 2]
        d = (results["systems"]["transformer @0.50"]["weighted"]["f1"]
             - results["systems"][other]["weighted"]["f1"])
        results[f"transformer_vs_{other}"] = {"diff": d, "ci95": tm.ci95(diff),
                                              "share_better": float((diff > 0).mean())}
        print(f"    transformer minus {other}: F1 {d:+.3f} [{tm.ci95(diff)[0]:+.3f} to "
              f"{tm.ci95(diff)[1]:+.3f}], better in {(diff > 0).mean():.0%} of resamples")
    return results


def gold_multiclass(model, tokenizer, encoder, rare, device) -> dict:
    if not prepare_dataset.GOLD_TEST_PATH.exists():
        return {}
    gold = prepare_dataset.read_label_file(prepare_dataset.GOLD_TEST_PATH)
    gold = gold[gold["label"].isin(["0", "1"])].reset_index(drop=True)
    w = tm.gold_weights(gold)
    dark_mask = (gold["label"] == "1").values
    dark = gold[dark_mask]
    if not len(dark):
        return {}
    truth = dark["Pattern Category"].where(
        ~dark["Pattern Category"].isin(rare) | ("Other" not in encoder.classes_), "Other")
    pred = encoder.inverse_transform(predict_proba(model, tokenizer, dark["text"], device).argmax(axis=1))
    res = {"n": int(len(dark)),
           "weighted_accuracy": float(accuracy_score(truth, pred, sample_weight=w[dark_mask])),
           "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0))}
    print(f"\n  Gold multi-class on {res['n']} human-marked dark rows: "
          f"weighted acc={res['weighted_accuracy']:.3f} macroF1={res['macro_f1']:.3f}")
    return res


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Fine-tune a Transformer dark-pattern classifier")
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Hugging Face model id, e.g. distilroberta-base, roberta-base, bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-train", type=int, default=0, help="subsample the training split (smoke tests)")
    p.add_argument("--seed", type=int, default=tm.RANDOM_STATE)
    args = p.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = tm.load_dataset()
    leaked = prepare_dataset.load_gold_texts() & set(df["text"])
    if leaked:
        sys.exit(f"{len(leaked)} gold texts are in the training data; run python train_model.py --prepare first.")

    # Identical split to train_model.py, so test numbers are comparable.
    y_binary = df["label"].values.astype(int)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=tm.TEST_SIZE,
                                           random_state=tm.RANDOM_STATE, stratify=y_binary)
    df = df.reset_index(drop=True)
    df["label"] = y_binary

    rare = []
    if args.task == "binary":
        label_col, task_df, tr_idx, te_idx = "label", df, train_idx, test_idx
    else:
        labels_before = df.loc[df["label"] == 1, "Pattern Category"].value_counts()
        rare = [c for c, n in labels_before.items() if n < tm.RARE_CATEGORY_MIN_SAMPLES]
        df = tm.handle_rare_categories(df)
        dark = df["label"] == 1
        tr_idx = np.intersect1d(train_idx, np.where(dark)[0])
        te_idx = np.intersect1d(test_idx, np.where(dark)[0])
        label_col, task_df = "Pattern Category", df

    train_df, test_df = task_df.iloc[tr_idx], task_df.iloc[te_idx]
    if args.max_train and args.max_train < len(train_df):
        train_df = train_df.sample(n=args.max_train, random_state=args.seed)
    encoder = LabelEncoder().fit(train_df[label_col])
    n_classes = len(encoder.classes_)
    print(f"\n{'=' * 60}\nTRANSFORMER {args.task.upper()} ({args.model})\n{'=' * 60}")
    print(f"  Train {len(train_df)}  Test {len(test_df)}  Classes {list(encoder.classes_)}")

    t0 = time.time()
    model, tokenizer, fit_info = train(args.model, train_df["text"].tolist(),
                                       encoder.transform(train_df[label_col]), n_classes,
                                       args.epochs, device, args.seed)

    # Held-out test split (categories unseen in a subsampled train are dropped).
    test_df = test_df[test_df[label_col].isin(encoder.classes_)]
    y_test = encoder.transform(test_df[label_col])
    pred = predict_proba(model, tokenizer, test_df["text"], device).argmax(axis=1)
    test = test_metrics(y_test, pred, n_classes)
    print(f"\n  Test: F1={test['f1']:.4f} macroF1={test['macro_f1']:.4f} acc={test['accuracy']:.4f}")
    print(classification_report(y_test, pred, target_names=[str(c) for c in encoder.classes_], zero_division=0))
    domains = {}
    for dom in sorted(test_df["domain"].unique()):
        m = (test_df["domain"] == dom).values
        if m.sum():
            domains[dom] = test_metrics(y_test[m], pred[m], n_classes)
            print(f"  Domain [{dom}]: F1={domains[dom]['f1']:.4f} acc={domains[dom]['accuracy']:.4f} "
                  f"n={domains[dom]['n_samples']}")

    gold = (gold_binary(model, tokenizer, device) if args.task == "binary"
            else gold_multiclass(model, tokenizer, encoder, rare, device))

    out_dir = tm.MODELS_DIR / f"transformer_{args.task}"
    model.config.id2label = {i: str(c) for i, c in enumerate(encoder.classes_)}
    model.config.label2id = {str(c): i for i, c in enumerate(encoder.classes_)}
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    metrics = {"model": args.model, "task": args.task, "epochs": args.epochs, "max_length": MAX_LENGTH,
               "learning_rate": LEARNING_RATE, "n_train": int(len(train_df)),
               "minutes": round((time.time() - t0) / 60, 1), **fit_info,
               "test": test, "domains": domains, "gold": gold}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\n  Saved {out_dir}/ (model, tokenizer, metrics.json) in {metrics['minutes']} min")


if __name__ == "__main__":
    main()
