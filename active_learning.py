"""
active_learning.py — Human-in-the-loop labelling for Indian e-commerce snippets

Every Indian row in combined_dataset.tsv is a pseudo-label from the baseline
model, so Indian-domain test scores only measure agreement with that baseline.
This script builds the human-labelled data that fixes it:

  gold    Draw a stratified random sample of scraped Indian snippets as a
          held-out GOLD TEST SET (data/gold/indian_gold_test.tsv). Strata are
          bands of the current binary model's P(dark); each row stores its
          inverse-sampling weight so train_model.py can report unbiased
          estimates for the whole scraped population. Gold texts are excluded
          from all training data by prepare_dataset.py.
  select  Pick the next ACTIVE-LEARNING batch from the unlabelled pool
          (needs_review.tsv): the snippets the current model is least sure
          about, de-duplicated and spread across sites
          (data/active_learning/batch_NN.tsv). Labelled batches are trained on.
  label   Interactive terminal labeller for a gold or batch file. Saves after
          every answer, so it can be quit and resumed. The model's guess is
          never shown, to keep labels unbiased.
  status  Labelling progress per file.

Workflow:
  python active_learning.py gold            # once
  python active_learning.py label gold
  python active_learning.py select          # then label it, retrain, repeat:
  python active_learning.py label data/active_learning/batch_01.tsv
  python train_model.py --prepare
"""

import argparse
import contextlib
import io
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import prepare_dataset
from prepare_dataset import (
    GOLD_TEST_PATH,
    ACTIVE_LEARNING_DIR,
    NEEDS_REVIEW_OUTPUT,
    read_label_file,
    load_gold_texts,
    load_human_labels,
)

MODELS_DIR = Path("models")
RANDOM_STATE = 42

# Mathur et al. categories, numbered for the labeller (0 = not a dark pattern)
CATEGORIES = [
    ("Scarcity", "low stock / high demand claims: 'Only 2 left', 'Selling fast'"),
    ("Social Proof", "activity or testimonial pressure: '120 people bought this today'"),
    ("Urgency", "deadlines and timers: 'Offer ends in 10:00', 'Today only'"),
    ("Misdirection", "confirmshaming, trick wording, pressured upsell: 'No thanks, I hate saving'"),
    ("Obstruction", "making cancel/opt-out hard: extra steps, 'Are you sure? You'll lose...'"),
    ("Sneaking", "hidden costs, items or subscriptions added without clear consent"),
    ("Forced Action", "must sign up / share data / install app to continue"),
]

# Gold strata over the current model's P(dark): (name, low, high, share of n)
GOLD_STRATA = [
    ("likely_dark", 0.50, 1.01, 0.40),
    ("borderline", 0.15, 0.50, 0.30),
    ("likely_benign", 0.00, 0.15, 0.30),
]

LABEL_COLUMNS = ["text", "site", "step_label", "label", "Pattern Category"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_binary():
    model = joblib.load(MODELS_DIR / "binary_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "binary_vectorizer.pkl")
    return model, vectorizer


def dark_proba(model, vectorizer, texts) -> np.ndarray:
    return model.predict_proba(vectorizer.transform(texts))[:, 1]


def scraped_pool() -> pd.DataFrame:
    """All deduplicated, junk-filtered scraped Indian snippets (Stage 1 output)."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = prepare_dataset.extract_all_snippets()
    return df[["text", "site", "step_label"]].reset_index(drop=True)


def batch_paths() -> list:
    return sorted(ACTIVE_LEARNING_DIR.glob("batch_*.tsv"))


def resolve_path(name: str) -> Path:
    return GOLD_TEST_PATH if name == "gold" else Path(name)


# -----------------------------------------------------------------------------
# gold
# -----------------------------------------------------------------------------

def cmd_gold(args):
    if GOLD_TEST_PATH.exists():
        existing = read_label_file(GOLD_TEST_PATH)
        if (existing["label"] != "").any() and not args.force:
            sys.exit(f"{GOLD_TEST_PATH} already has labels. Refusing to replace it "
                     "(pass --force to discard them).")

    model, vectorizer = load_binary()
    pool = scraped_pool()
    pool["p_dark"] = dark_proba(model, vectorizer, pool["text"])
    print(f"Scraped pool: {len(pool)} snippets from {pool['site'].nunique()} sites")

    parts = []
    for name, low, high, share in GOLD_STRATA:
        stratum = pool[(pool["p_dark"] >= low) & (pool["p_dark"] < high)]
        take = min(len(stratum), int(round(args.n * share)))
        if take == 0:
            continue
        sample = stratum.sample(n=take, random_state=RANDOM_STATE).copy()
        sample["stratum"] = name
        sample["stratum_size"] = len(stratum)
        sample["weight"] = len(stratum) / take
        parts.append(sample)
        print(f"  {name:14s} P(dark) in [{low:.2f}, {high:.2f}): {len(stratum):6d} in pool -> {take} sampled")

    gold = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)  # shuffle: labeller can't infer stratum
    gold["label"] = ""
    gold["Pattern Category"] = ""
    gold = gold[LABEL_COLUMNS + ["stratum", "stratum_size", "weight"]]

    GOLD_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(GOLD_TEST_PATH, sep="\t", index=False)
    print(f"\nWrote {len(gold)} gold rows to {GOLD_TEST_PATH}")
    print("Next: python active_learning.py label gold")
    print("Then rerun python train_model.py --prepare so these texts leave the training data.")


# -----------------------------------------------------------------------------
# select
# -----------------------------------------------------------------------------

def cmd_select(args):
    if not NEEDS_REVIEW_OUTPUT.exists():
        sys.exit(f"{NEEDS_REVIEW_OUTPUT} not found. Run python prepare_dataset.py first.")

    pending = set()
    for p in batch_paths():
        b = read_label_file(p)
        pending |= set(b.loc[b["label"] == "", "text"])
    if pending and not args.allow_pending:
        sys.exit(f"{len(pending)} rows in existing batches are still unlabelled. Label them "
                 "first (or pass --allow-pending).")

    pool = read_label_file(NEEDS_REVIEW_OUTPUT)[["text", "site", "step_label"]]
    exclude = load_gold_texts() | set(load_human_labels()["text"]) | pending
    for p in batch_paths():
        exclude |= set(read_label_file(p)["text"])  # includes rows marked unsure
    pool = pool[~pool["text"].isin(exclude)].drop_duplicates(subset=["text"]).reset_index(drop=True)
    if pool.empty:
        sys.exit("Unlabelled pool is empty.")

    model, vectorizer = load_binary()
    p_dark = dark_proba(model, vectorizer, pool["text"])
    pool["uncertainty"] = 1 - np.abs(2 * p_dark - 1)  # 1 at P=0.5, 0 when certain

    # Most uncertain first, then greedily skip near-duplicates and cap rows per
    # site so one site's repeated template can't fill the batch.
    candidates = pool.sort_values("uncertainty", ascending=False).head(args.n * 10).reset_index(drop=True)
    X = vectorizer.transform(candidates["text"])
    per_site_cap = max(3, args.n // 10)
    picked, site_counts = [], {}
    for i in range(len(candidates)):
        site = candidates.at[i, "site"]
        if site_counts.get(site, 0) >= per_site_cap:
            continue
        if picked and cosine_similarity(X[i], X[picked]).max() > args.max_similarity:
            continue
        picked.append(i)
        site_counts[site] = site_counts.get(site, 0) + 1
        if len(picked) == args.n:
            break

    batch = candidates.iloc[picked].copy()
    batch["label"] = ""
    batch["Pattern Category"] = ""

    ACTIVE_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    out = ACTIVE_LEARNING_DIR / f"batch_{len(batch_paths()) + 1:02d}.tsv"
    batch[LABEL_COLUMNS].to_csv(out, sep="\t", index=False)
    print(f"Pool: {len(pool)} unlabelled snippets. Selected {len(batch)} from {len(site_counts)} sites "
          f"(mean uncertainty {batch['uncertainty'].mean():.3f} vs pool {pool['uncertainty'].mean():.3f})")
    print(f"Wrote {out}\nNext: python active_learning.py label {out.as_posix()}")


# -----------------------------------------------------------------------------
# label
# -----------------------------------------------------------------------------

def print_guide():
    print("\nKeys:  0 = NOT a dark pattern")
    for i, (name, hint) in enumerate(CATEGORIES, start=1):
        print(f"       {i} = {name:13s} {hint}")
    print("       u = unsure (excluded)   b = back (redo previous)   q = save & quit   ? = this help\n")


def cmd_label(args):
    path = resolve_path(args.file)
    if not path.exists():
        sys.exit(f"{path} not found.")
    df = read_label_file(path)

    todo = list(df.index[df["label"] == ""])
    if not todo:
        print(f"All {len(df)} rows in {path} are labelled.")
        return

    print(f"Labelling {path}: {len(todo)} of {len(df)} rows left. Judge the text as a shopper would see it.")
    print_guide()

    history = []
    pos = 0
    while pos < len(todo):
        idx = todo[pos]
        row = df.loc[idx]
        print(f"[{len(df) - len(todo) + pos + 1}/{len(df)}] site={row['site']}  page={row['step_label']}")
        print(f"    \"{row['text']}\"")
        answer = input("  > ").strip().lower()

        if answer == "q":
            break
        if answer == "?":
            print_guide()
            continue
        if answer == "b":
            if history:
                pos = history.pop()
                df.loc[todo[pos], ["label", "Pattern Category"]] = ""
                df.to_csv(path, sep="\t", index=False)
            continue
        if answer == "u":
            df.loc[idx, ["label", "Pattern Category"]] = ["unsure", ""]
        elif answer == "0":
            df.loc[idx, ["label", "Pattern Category"]] = ["0", "Not Dark Pattern"]
        elif answer.isdigit() and 1 <= int(answer) <= len(CATEGORIES):
            df.loc[idx, ["label", "Pattern Category"]] = ["1", CATEGORIES[int(answer) - 1][0]]
        else:
            print("  ? unrecognised key")
            continue

        df.to_csv(path, sep="\t", index=False)  # save every answer
        history.append(pos)
        pos += 1

    done = (df["label"] != "").sum()
    print(f"\nSaved {path}: {done}/{len(df)} labelled.")


# -----------------------------------------------------------------------------
# status
# -----------------------------------------------------------------------------

def cmd_status(args):
    paths = ([GOLD_TEST_PATH] if GOLD_TEST_PATH.exists() else []) + batch_paths()
    if not paths:
        print("No gold set or batches yet. Start with: python active_learning.py gold")
        return
    print(f"{'file':42s} {'rows':>5s} {'done':>5s} {'dark':>5s} {'unsure':>7s}")
    for p in paths:
        d = read_label_file(p)
        print(f"{p.as_posix():42s} {len(d):5d} {(d['label'] != '').sum():5d} "
              f"{(d['label'] == '1').sum():5d} {(d['label'] == 'unsure').sum():7d}")
    print(f"\nHuman labels available for training: {len(load_human_labels())}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ₹ etc. on Windows consoles

    parser = argparse.ArgumentParser(description="Human-in-the-loop labelling for Indian snippets")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gold", help="create the stratified held-out gold test set")
    p.add_argument("--n", type=int, default=300, help="gold set size (default 300)")
    p.add_argument("--force", action="store_true", help="replace a gold set that already has labels")
    p.set_defaults(func=cmd_gold)

    p = sub.add_parser("select", help="pick the next active-learning batch")
    p.add_argument("--n", type=int, default=100, help="batch size (default 100)")
    p.add_argument("--max-similarity", type=float, default=0.8,
                   help="skip candidates with cosine similarity above this to an already-picked one")
    p.add_argument("--allow-pending", action="store_true", help="select even if a batch is unfinished")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("label", help="label a file interactively ('gold' or a batch path)")
    p.add_argument("file")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("status", help="labelling progress")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
