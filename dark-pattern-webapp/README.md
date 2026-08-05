# Dark Pattern Detector — Web App

Two separate pieces that talk to each other over HTTP:

```
dark-pattern-webapp/
├── backend/
│   ├── server.py          # FastAPI server, loads model, exposes /predict
│   ├── requirements.txt
│   ├── baseline_model.pkl        <- YOU add this (copy from your training folder)
│   └── baseline_vectorizer.pkl   <- YOU add this
└── frontend/
    └── index.html          # website UI, calls the backend
```

## 1. Set up the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
playwright install chromium
```

The last line downloads a headless Chromium browser that the server uses to
load product pages exactly like a real browser would (needed because most
e-commerce sites render their content with JavaScript).

Copy your two trained model files into the `backend/` folder:
- `baseline_model.pkl`
- `baseline_vectorizer.pkl`

(These are the ones you created earlier with `joblib.dump(...)` in `readdata.py`.)

Start the server:

```bash
uvicorn server:app --reload
```

You should see `Model and vectorizer loaded successfully.` in the terminal.
Leave this terminal running — visit `http://127.0.0.1:8000` in a browser, you
should see `{"status":"ok",...}`.

## 2. Open the frontend

Just double-click `frontend/index.html` to open it in your browser, or serve
it with a simple local server (VS Code's "Live Server" extension works well
too — right-click `index.html` → "Open with Live Server").

The page will show "Backend connected · model ready" at the top if everything
is wired up correctly. Paste some text and click Analyze.

## Common issues

- **"Backend not reachable"** → the `uvicorn` server in step 1 isn't running,
  or you closed that terminal.
- **CORS error in browser console** → already handled in `server.py` via
  `CORSMiddleware`. If you still see it, make sure you didn't remove that
  block.
- **"Model not loaded" (503 error)** → the `.pkl` files aren't in `backend/`.
  Check the filenames match exactly.

## How the URL analysis works

1. You paste a product link in the frontend and click "Analyze link".
2. The backend opens the page in a headless Chromium browser (via Playwright),
   waits ~1.5s for JS content to render, then pulls out every small piece of
   visible text (buttons, badges, banners, labels — anything with no nested
   elements, 3-300 characters).
3. Each text piece is run through your trained classifier individually.
4. The frontend shows a verdict + a list of the specific flagged lines with
   confidence scores, so you can see exactly what tripped the detector.

## Known limitations (worth mentioning in your report)

- **Bot detection / login walls**: some sites block headless browsers or
  require login before showing checkout/cancellation flows. Those pages will
  return "could not extract text" or empty results.
- **Binary label only**: right now every flagged chunk just says "dark
  pattern", not which category (Urgency, Scarcity, etc.) — because your
  baseline model was trained for binary classification. If you also train a
  multi-class model on `Pattern Category`, you can add category labels to
  each flagged item.
- **Chunking is heuristic**: "leaf DOM elements" is a good approximation for
  microcopy but isn't perfect — some real dark patterns span multiple
  elements (e.g. a countdown built from several `<span>` tags) and may get
  split up or missed. Mention this as a known limitation / future work.

## Next steps once this works

1. **Deploy it** so it's not just `localhost` — cheapest options: backend on
   Render/Railway (free tier), frontend on Vercel/Netlify/GitHub Pages.
2. **Add category prediction** — train a second, multi-class model on
   `Pattern Category` and return it alongside the binary flag.
3. **Highlight in context** — instead of a flat list, show flagged lines
   overlaid on a screenshot of the actual page (you already capture
   screenshots in your Playwright scraper).
