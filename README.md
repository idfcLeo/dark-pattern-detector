# Dark Pattern Detector

A text-based detector for dark patterns in e-commerce experiences. The project
collects text from shopping flows, prepares a training dataset, trains binary
and category classifiers, and exposes the trained models through a small web
application.

The detector can classify either a pasted snippet or the visible microcopy
extracted from a product URL. For flagged text, it can also return a dark
pattern category and related CCPA consumer-protection context.

## What is included

- **Collection** — Playwright-assisted capture of landing, product, cart,
  checkout, payment, settings, and cancellation-flow pages.
- **Dataset preparation** — cleaning, pseudo-labelling, structural features,
  human-label overrides, class balancing, and CCPA category mapping.
- **Training** — binary (dark pattern / not) and multi-class classifiers with
  cross-validation, held-out testing, per-domain metrics, and a human-labelled
  Indian gold-set evaluation.
- **Active learning** — terminal workflow for selecting and labelling the
  most useful Indian e-commerce snippets.
- **Web app** — FastAPI API plus a lightweight browser frontend for text and
  URL analysis.

## Repository layout

```text
.
|-- data/
|   |-- raw/                 # captured page text, screenshots, and metadata
|   |-- active_learning/     # human-labelling batches
|   `-- gold/                # held-out human-labelled evaluation set
|-- models/                  # trained classifiers, vectorizers, and reports
|-- configs/ccpa_mapping.tsv # category-to-CCPA reference mapping
|-- dark-pattern-webapp/
|   |-- backend/             # FastAPI service
|   `-- frontend/            # static browser UI
|-- scraper.py               # interactive shopping-flow capture
|-- prepare_dataset.py       # data preparation pipeline
|-- active_learning.py       # labelling workflow
`-- train_model.py           # model training and evaluation
```

`ec-darkpattern/` contains the upstream dataset and baseline artifacts used as
an input to this project. See its own README for attribution and licensing.

## Quick start: run the web app

The repository already includes trained artifacts under `models/`, so training
is not required to try the application.

```powershell
cd dark-pattern-webapp/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
uvicorn server:app --reload
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate` instead. The API will run at
<http://127.0.0.1:8000>; `GET /health` confirms whether the model loaded.

Then open `dark-pattern-webapp/frontend/index.html` in a browser (or serve it
with any static-file server). The frontend supports:

- **Text analysis** via `POST /predict`
- **URL analysis** via `POST /analyze-url`, which renders JavaScript pages in
  Chromium, extracts visible text chunks, and evaluates them one at a time

### API example

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict `
  -ContentType 'application/json' `
  -Body '{"text":"Only 3 left — buy now!"}'
```

## Rebuild the dataset and models

Use Python 3.10+ and install the training dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn joblib playwright
pip install optuna lightgbm xgboost  # optional: enables tuned candidate models
playwright install chromium
```

Run the pipeline from the repository root:

```powershell
# Optionally capture new e-commerce flows after adding sites in scraper.py
python scraper.py

# Extract, clean, pseudo-label, merge, and write data/processed/ outputs
python prepare_dataset.py

# Train binary and multi-class models; write models/ artifacts and report
python train_model.py
```

The training script uses an 80/20 stratified split and selects models using
five-fold cross-validation on the training partition. It writes the deployed
binary model and vectorizer, multi-class artifacts, a decision-threshold file,
and `models/training_report.txt`.

## Human labelling and active learning

The Indian e-commerce data can be improved through a small terminal labelling
loop. Labels in the gold set are reserved for evaluation and are not used for
training.

```powershell
# See available labels and batch progress
python active_learning.py status

# Label the held-out gold set interactively
python active_learning.py label gold

# Select a new batch, then label it and retrain
python active_learning.py select
python active_learning.py label data/active_learning/batch_05.tsv
python prepare_dataset.py
python train_model.py
```

## Current model results

The latest report (`models/training_report.txt`) selected logistic regression
for binary detection. On its held-out combined test set it achieved **0.898
F1 for the dark-pattern class** and **0.951 accuracy**. The stacked
multi-class model achieved **0.859 macro F1** on its held-out test set.

Performance on the small human-labelled Indian gold set is lower, which is an
important limitation: the distribution differs substantially from the
upstream data. Treat URL-analysis results as decision support, not a legal or
compliance determination.

## Limitations

- E-commerce sites may block headless browsers or require authentication.
- URL extraction uses visible text chunks; patterns expressed solely through
  layout, imagery, timing, or multi-element interactions may be missed.
- Some rare pattern categories are merged into `Other` during training.
- Dataset and domain shift can reduce accuracy, especially on new Indian sites.

## Acknowledgements

The project builds on the text-based e-commerce dark-pattern dataset and
baseline work in `ec-darkpattern/`, based on prior work by Mathur et al. and
Yada et al. Please review the upstream repository's README and license before
redistributing its data or artifacts.
