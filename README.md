# Visual Product Search Engine

An end-to-end multimodal retrieval pipeline for the VR Final Project. This project implements a query-by-image product search system using a fine-tuned CLIP model, BLIP image-text matching, YOLO object detection, and Hierarchical Navigable Small World (HNSW) indexing.

---

## 🚀 Features
- **Interactive Streamlit App:** An easy-to-use web UI that accepts an image, crops it using YOLO, searches an HNSW index, re-ranks the results using semantic text matching, and displays the top results.
- **Batch Evaluation Script:** Automates the complete ablation study required for the project, comparing Base Vision, Base Fusion, and Fine-Tuned Fusion configurations across different alpha weightings.
- **Offline Index Builder:** Automatically processes pre-computed `.npy` embeddings into exact HNSW FAISS indices for blazing fast online retrieval.

---

## 🛠️ Setup & Installation

### 1. Virtual Environment (Recommended)
It is highly recommended to run this project in a Python virtual environment to prevent dependency conflicts.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install all required machine learning, retrieval, and UI libraries:

```bash
pip install -r requirements.txt
```

---

## 📁 Required Data & Model Files

Before running the application or evaluation scripts, ensure you have the following files located in your `data/` directory.

**Model Weights:**
- `best_fashion_clip.pth` (Fine-tuned CLIP weights from your notebook)

**Base Embeddings:**
- `gallery_image_embeddings_512.npy`
- `gallery_text_embeddings.npy`

**Fine-Tuned Embeddings:**
- `optimized_gallery_image_embeddings.npy`
- `optimized_gallery_text_embeddings.npy`

**Metadata & Ground Truth:**
- `gallery_paths.npy`
- `gallery_item_ids.npy`
- `gallery_captions.npy`
- `query_df.csv` (For batch evaluation)

**Images Folder:**
- You must have an `Img/` folder in the root directory containing the DeepFashion gallery images (e.g. `Img/img/WOMEN/...`).

**YOLOv8 Weights:**
- `yolov8n_fashion.pt`: Automatically downloaded when you first run the app, handling clothing-specific object detection.

---

## 🏃‍♂️ How to Run

### Step 1: Build Offline HNSW Indices
Before doing any searching or evaluation, you must build the FAISS HNSW graph indices for all of your ablation configurations.

```bash
python build_indices.py
```
*This will read your `.npy` embedding files, dynamically fuse them using different alpha weights (default 0.7 and 0.5), and save them as `.index` files in your `data/` folder.*

### Step 2: Run Batch Evaluation (Ablation Study)
To run the automated pipeline across your `query_df.csv` and generate the `Recall@K`, `NDCG@K`, and `mAP@K` metrics for your final report:

1. Open `evaluate.py`.
2. Edit **Line 14**: `SEEDS = [101, 102, 103]` to match your team's actual roll numbers.
3. Edit **Lines 100-101**: Adjust `ALPHA_1` and `ALPHA_2` if you prefer different fusion weights.
4. Run the script:

```bash
python evaluate.py
```

### Step 3: Run the Streamlit Application
To interact with the final production model via the web application:

```bash
streamlit run app.py
```
*The app will automatically open in your web browser. It uses the `best_fashion_clip.pth` and the `data/optimized_fusion_hnsw_faiss.index` generated in Step 1.*

---

## 🧠 Architecture Overview

### Offline Indexing Pipeline
1. **Base Generation:** Generating image and text embeddings for the gallery using CLIP.
2. **Fine-Tuning:** Training CLIP using contrastive loss to pull identical `item_id`s together.
3. **Fusion:** Blending the fine-tuned image embedding and BLIP caption embedding via an alpha/beta weighted sum.
4. **HNSW:** Storing the normalized fusion vectors into a FAISS Hierarchical Navigable Small World index for low-latency retrieval.

### Online Query Pipeline
1. **Localization:** `yolov8n_fashion.pt` identifies and crops the primary clothing item from the uploaded image.
2. **Query Encoding:** The cropped query is passed through the fine-tuned CLIP Vision Encoder to generate a 512-dimensional embedding.
3. **Retrieval:** The system quickly searches the FAISS HNSW index using cosine similarity (Inner Product) to return the Top-K candidates.
4. **Semantic Re-Ranking:** The `Salesforce/blip-itm-base-coco` model computes an Image-Text Matching (ITM) probability score between the query crop and the pre-computed BLIP captions of the retrieved candidates, sorting the final list by maximum semantic relevance.
