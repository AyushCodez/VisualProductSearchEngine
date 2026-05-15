"""
Visual Product Search — Streamlit Demo App
------------------------------------------
End-to-end demo skeleton for the VR Final Project.

Pipeline:
    1. Upload image
    2. Run YOLO detection -> show cropped product(s)
    3. Pause for user to Confirm / Re-crop
    4. Run visual search (embedding + similarity)
    5. Show top-K results with metadata & similarity scores

Notes:
    - Detection + search are wrapped behind small functions so you can plug in
      your trained YOLO model and embedding/index code later. Stubs return
      mock data so the UI works end-to-end today.

Run:
    pip install streamlit pillow numpy
    streamlit run app.py
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval
import faiss
import os

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Visual Product Search")
st.caption(
    "Upload an image → detect product with YOLO → confirm crop → retrieve similar products."
)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
DEFAULTS = {
    "uploaded_image": None,
    "detections": None,        # list[ (bbox, crop_pil) ]
    "selected_idx": 0,
    "crop_confirmed": False,
    "results": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_pipeline():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ------------------------------------------------------------------
# Model Caching
# ------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n_fashion.pt")

@st.cache_resource
def load_retrieval_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load fine-tuned CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    try:
        clip_model.load_state_dict(torch.load("data/best_fashion_clip.pth", map_location=device))
    except Exception as e:
        print(f"Warning: Could not load fine-tuned CLIP weights. Using base model. Error: {e}")
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load BLIP ITM
    blip_itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    blip_itm_model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    blip_itm_model.eval()
    
    return clip_model, clip_processor, blip_itm_model, blip_itm_processor, device

@st.cache_resource
def load_retrieval_data():
    index = faiss.read_index("data/optimized_fusion_hnsw_faiss.index")
    gallery_paths = np.load("data/gallery_paths.npy", allow_pickle=True)
    gallery_item_ids = np.load("data/gallery_item_ids.npy", allow_pickle=True)
    gallery_captions = np.load("data/gallery_captions.npy", allow_pickle=True)
    return index, gallery_paths, gallery_item_ids, gallery_captions

yolo_model = load_yolo_model()
clip_model, clip_processor, blip_itm_model, blip_itm_processor, device = load_retrieval_models()
fusion_index, gallery_paths, gallery_item_ids, gallery_captions = load_retrieval_data()

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2)
    label: str
    confidence: float
    crop: Image.Image


def run_yolo_detection(image: Image.Image) -> List[Detection]:
    """
    Run real YOLO inference.
    Returns a list of Detection objects with crops.
    """
    results = yolo_model(image)
    detections = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        label = yolo_model.names[cls_id]
        
        # Crop the image using integer coordinates
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
        detections.append(Detection((x1, y1, x2, y2), label, conf, crop))
        
    # Always provide a fallback "Full Image" option
    w, h = image.size
    detections.append(Detection((0, 0, w, h), "full_image", 1.0, image))
        
    return detections


def draw_boxes(image: Image.Image, detections: List[Detection]) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, d in enumerate(detections):
        draw.rectangle(d.bbox, outline=(255, 64, 64), width=4)
        draw.text((d.bbox[0] + 6, d.bbox[1] + 6),
                  f"{i+1}. {d.label} {d.confidence:.2f}",
                  fill=(255, 255, 255))
    return img


@dataclass
class SearchResult:
    product_id: str
    title: str
    category: str
    price: str
    score: float
    image: Image.Image


def run_visual_search(crop: Image.Image, top_k: int = 8) -> List[SearchResult]:
    """
    Real multimodal retrieval pipeline with ITM re-ranking.
    """
    # 1. CLIP Image Embedding (Query is only image)
    inputs = clip_processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        image_features = clip_model.visual_projection(vision_outputs.pooler_output)
    image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
    query_emb = image_features.cpu().numpy()[0].astype(np.float32).reshape(1, -1)
    
    # 2. Candidate Retrieval (fetch top-15 from FAISS)
    search_k = 15
    scores, indices = fusion_index.search(query_emb, search_k)
    
    candidates = []
    for i in range(search_k):
        idx = indices[0][i]
        faiss_score = scores[0][i]
        candidates.append({
            "idx": idx,
            "faiss_score": float(faiss_score),
            "item_id": gallery_item_ids[idx],
            "rel_path": gallery_paths[idx],
            "caption": str(gallery_captions[idx])
        })
        
    # 3. Semantic Re-ranking (BLIP ITM)
    for cand in candidates:
        itm_inputs = blip_itm_processor(images=crop, text=cand["caption"], return_tensors="pt").to(
            device, torch.float16 if device == "cuda" else torch.float32
        )
        with torch.no_grad():
            itm_scores = blip_itm_model(**itm_inputs)[0]
            # itm_scores usually outputs logits for [negative, positive]
            itm_prob = torch.nn.functional.softmax(itm_scores, dim=1)[0][1].item()
        cand["itm_score"] = itm_prob
        
    # Sort candidates by ITM score descending
    candidates.sort(key=lambda x: x["itm_score"], reverse=True)
    
    # 4. Build Final Top-K Results
    results = []
    for i in range(min(top_k, len(candidates))):
        cand = candidates[i]
        full_path = os.path.join("Img", cand["rel_path"])
        
        try:
            res_image = Image.open(full_path).convert("RGB")
        except Exception:
            res_image = Image.new("RGB", (256, 256), color=(200, 200, 200))
            
        results.append(SearchResult(
            product_id=cand["item_id"],
            title=cand["caption"].title()[:40] + "...",
            category=cand["rel_path"].split("/")[1] if "/" in cand["rel_path"] else "Unknown",
            price=f"${random.randint(15, 250)}.99",
            score=cand["itm_score"],
            image=res_image,
        ))
        
    return results


# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K results", 1, 20, 8)
    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        reset_pipeline()
        st.rerun()

# ------------------------------------------------------------------
# Step 1 — Upload
# ------------------------------------------------------------------
st.subheader("1) Upload an image")
file = st.file_uploader("Choose a product image", type=["jpg", "jpeg", "png", "webp"])

if file is not None:
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    if st.session_state.uploaded_image is None or \
       img.tobytes() != st.session_state.uploaded_image.tobytes():
        st.session_state.uploaded_image = img
        st.session_state.detections = None
        st.session_state.crop_confirmed = False
        st.session_state.results = None

if st.session_state.uploaded_image is None:
    st.info("Upload an image to begin.")
    st.stop()

st.image(st.session_state.uploaded_image, caption="Uploaded image", width=400)

# ------------------------------------------------------------------
# Step 2 — Detection + crop
# ------------------------------------------------------------------
st.subheader("2) Detect & crop")

if st.session_state.detections is None:
    if st.button("Run YOLO detection", type="primary"):
        with st.spinner("Detecting…"):
            st.session_state.detections = run_yolo_detection(
                st.session_state.uploaded_image
            )
        st.rerun()
    st.stop()

dets = st.session_state.detections
col1, col2 = st.columns(2)
with col1:
    st.image(draw_boxes(st.session_state.uploaded_image, dets),
             caption="Detections", use_container_width=True)
with col2:
    options = [f"{i+1}. {d.label} ({d.confidence:.2f})" for i, d in enumerate(dets)]
    idx = st.radio("Select a detection", range(len(options)),
                   format_func=lambda i: options[i],
                   index=st.session_state.selected_idx)
    st.session_state.selected_idx = idx
    st.image(dets[idx].crop, caption="Cropped product", use_container_width=True)

# ------------------------------------------------------------------
# Step 3 — Confirm crop (interactive pause)
# ------------------------------------------------------------------
st.subheader("3) Confirm the crop")

c1, c2, _ = st.columns([1, 1, 4])
with c1:
    if st.button("✅ Confirm crop", type="primary",
                 disabled=st.session_state.crop_confirmed):
        st.session_state.crop_confirmed = True
        st.session_state.results = None
        st.rerun()
with c2:
    if st.button("🔁 Re-crop"):
        st.session_state.detections = None
        st.session_state.crop_confirmed = False
        st.session_state.results = None
        st.rerun()

if not st.session_state.crop_confirmed:
    st.warning("Please confirm the crop to run the visual search.")
    st.stop()

# ------------------------------------------------------------------
# Step 4 + 5 — Search + results
# ------------------------------------------------------------------
st.subheader(f"4) Top-{top_k} similar products")

if st.session_state.results is None:
    with st.spinner("Searching index…"):
        st.session_state.results = run_visual_search(
            dets[st.session_state.selected_idx].crop, top_k=top_k
        )

results = st.session_state.results
cols_per_row = 4
for row_start in range(0, len(results), cols_per_row):
    row = results[row_start:row_start + cols_per_row]
    cols = st.columns(len(row))
    for col, r in zip(cols, row):
        with col:
            st.image(r.image, use_container_width=True)
            st.markdown(f"**{r.title}**")
            st.caption(f"{r.product_id} · {r.category}")
            st.write(f"💲 {r.price}")
            st.progress(min(max(r.score, 0.0), 1.0),
                        text=f"Similarity: {r.score:.3f}")

st.success("Visual Product Search complete!")
