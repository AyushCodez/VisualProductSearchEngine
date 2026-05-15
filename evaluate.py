import os
import torch
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval
from ultralytics import YOLO

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
SEEDS = [546, 577, 607] 
TOP_K_LIST = [5, 10, 15]

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(retrieved_ids, true_id, k_list):
    metrics = {}
    for k in k_list:
        retrieved_k = retrieved_ids[:k]
        
        # Recall@K
        recall = 1 if true_id in retrieved_k else 0
        metrics[f'Recall@{k}'] = recall
        
        # NDCG@K
        dcg = 0
        for i, pred_id in enumerate(retrieved_k):
            if pred_id == true_id:
                dcg += 1 / np.log2(i + 2)
        idcg = 1 / np.log2(2) # Ideal is match at rank 1
        ndcg = dcg / idcg
        metrics[f'NDCG@{k}'] = ndcg
        
        # mAP@K
        ap = 0
        hits = 0
        for i, pred_id in enumerate(retrieved_k):
            if pred_id == true_id:
                hits += 1
                ap += hits / (i + 1)
        metrics[f'mAP@{k}'] = ap
        
    return metrics

# Cache for YOLO crops to avoid re-running object detection
YOLO_CACHE = {}

def get_yolo_crop(yolo_model, image_path):
    if image_path in YOLO_CACHE:
        return YOLO_CACHE[image_path]
        
    full_path = os.path.join("Img", image_path)
    try:
        image = Image.open(full_path).convert("RGB")
    except Exception:
        return None
        
    results = yolo_model(image, verbose=False)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    else:
        crop = image # fallback
        
    YOLO_CACHE[image_path] = crop
    return crop

def main():
    print(f"Running on {device.upper()}")
    print("Loading query dataset...")
    query_df = pd.read_csv("data/query_df.csv")
    
    # Load Models
    print("Loading Base Models...")
    yolo_model = YOLO("yolov8n_fashion.pt")
    
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    base_clip_model.eval()
    
    finetuned_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    finetuned_clip_model.load_state_dict(torch.load("data/best_fashion_clip.pth", map_location=device))
    finetuned_clip_model.to(device)
    finetuned_clip_model.eval()
    
    blip_itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    blip_itm_model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    blip_itm_model.eval()
    
    # Load Gallery Metadata
    print("Loading Gallery Metadata...")
    gallery_item_ids = np.load("data/gallery_item_ids.npy", allow_pickle=True)
    gallery_captions = np.load("data/gallery_captions.npy", allow_pickle=True)
    
    configs = [
        {"name": "A - Vision-only Base CLIP", "model": base_clip_model, "index_path": "data/index_config_A.index"},
        {"name": "B1 - Frozen Base CLIP + Frozen BLIP (a=0.7)", "model": base_clip_model, "index_path": "data/index_config_B_alpha_0.7.index"},
        {"name": "B2 - Frozen Base CLIP + Frozen BLIP (a=0.5)", "model": base_clip_model, "index_path": "data/index_config_B_alpha_0.5.index"},
        {"name": "C1 - Fine-tuned CLIP + Frozen BLIP (a=0.7)", "model": finetuned_clip_model, "index_path": "data/index_config_C_alpha_0.7.index"},
        {"name": "C2 - Fine-tuned CLIP + Frozen BLIP (a=0.5)", "model": finetuned_clip_model, "index_path": "data/index_config_C_alpha_0.5.index"},
    ]
    
    
    final_results = {}
    
    for seed in SEEDS:
        print(f"\n{'='*50}\nEvaluating Seed: {seed}\n{'='*50}")
        # Setting random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Sample queries to speed up evaluation (or use all)
        # For full evaluation, you may want to use all queries.
        # We limit to 100 here for demo script sanity. Adjust as needed.
        eval_queries = query_df.sample(min(100, len(query_df)), random_state=seed)
        
        for config in configs:
            print(f"\n--- Running Config: {config['name']} ---")
            
            # Load the pre-computed HNSW index
            index = faiss.read_index(config['index_path'])
            active_model = config['model']
            
            all_metrics = []
            
            for _, row in tqdm(eval_queries.iterrows(), total=len(eval_queries), desc="Queries"):
                true_id = row['item_id']
                img_path = row['image_path']
                
                # 1. YOLO Crop
                crop = get_yolo_crop(yolo_model, img_path)
                if crop is None:
                    continue
                    
                # 2. Query Encoding (Visual Only)
                inputs = clip_processor(images=crop, return_tensors="pt").to(device)
                with torch.no_grad():
                    vision_outputs = active_model.vision_model(pixel_values=inputs["pixel_values"])
                    image_features = active_model.visual_projection(vision_outputs.pooler_output)
                image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
                query_emb = image_features.cpu().numpy()[0].astype(np.float32).reshape(1, -1)
                
                # 3. Candidate Retrieval (fetch top-30 for re-ranking)
                search_k = 30
                scores, indices = index.search(query_emb, search_k)
                
                candidates = []
                for i in range(search_k):
                    idx = indices[0][i]
                    candidates.append({
                        "item_id": gallery_item_ids[idx],
                        "caption": str(gallery_captions[idx])
                    })
                
                # 4. Semantic Re-ranking (Batched for speed)
                texts = [cand["caption"] for cand in candidates]
                itm_inputs = blip_itm_processor(images=[crop]*len(texts), text=texts, return_tensors="pt", padding=True).to(
                    device, torch.float16 if device == "cuda" else torch.float32
                )
                with torch.no_grad():
                    itm_scores = blip_itm_model(**itm_inputs)[0]
                    itm_probs = torch.nn.functional.softmax(itm_scores, dim=1)[:, 1].cpu().tolist()
                
                for cand, prob in zip(candidates, itm_probs):
                    cand["itm_score"] = prob
                    
                candidates.sort(key=lambda x: x["itm_score"], reverse=True)
                retrieved_ids = [c["item_id"] for c in candidates]
                
                # 5. Compute Metrics
                metrics = compute_metrics(retrieved_ids, true_id, TOP_K_LIST)
                all_metrics.append(metrics)
                
            # Aggregate for this config
            agg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
            final_results.setdefault(config['name'], []).append(agg)
            print(f"Results for {config['name']}:")
            for k, v in agg.items():
                print(f"  {k}: {v:.4f}")

    print(f"\n\n{'='*50}\nFINAL ABLATION STUDY RESULTS\n{'='*50}")
    for config_name, runs in final_results.items():
        print(f"\nConfiguration: {config_name}")
        metrics_keys = runs[0].keys()
        for k in metrics_keys:
            vals = [r[k] for r in runs]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()
