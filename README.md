# Visual Product Search — Streamlit Demo

Interactive demo for the VR Final Project pipeline:
**Upload → YOLO detect & crop → Confirm crop → Visual search → Top-K results**.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Where to plug in your models

Both heavy steps are isolated stubs in `app.py`:

- `run_yolo_detection(image)` — replace with your trained YOLO inference.
  Return a list of `Detection(bbox, label, confidence, crop)`.
- `run_visual_search(crop, top_k)` — replace with your embedding model +
  nearest-neighbour search over your product index. Return a list of
  `SearchResult(product_id, title, category, price, score, image)`.

The UI, session state, the **Confirm crop / Re-crop** interactivity, and the
top-K grid with similarity scores all work out of the box with mock data.
