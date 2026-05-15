import faiss
import numpy as np

def build_and_save_index(embeddings, output_path):
    print(f"Building HNSW index for {output_path}...")
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, output_path)
    print("Done!")

def main():
    ALPHA_1 = 0.7
    ALPHA_2 = 0.5
    
    print("Loading Base Embeddings...")
    base_img_emb = np.load("data/gallery_image_embeddings_512.npy")
    base_txt_emb = np.load("data/gallery_text_embeddings.npy")
    
    print("Loading Fine-tuned Embeddings...")
    opt_img_emb = np.load("data/optimized_gallery_image_embeddings.npy")
    opt_txt_emb = np.load("data/optimized_gallery_text_embeddings.npy")
    
    # Config A
    build_and_save_index(base_img_emb, "data/index_config_A.index")
    
    # Config B1
    fusion_B1 = (ALPHA_1 * base_img_emb + (1 - ALPHA_1) * base_txt_emb)
    fusion_B1 = fusion_B1 / np.linalg.norm(fusion_B1, axis=1, keepdims=True)
    build_and_save_index(fusion_B1, f"data/index_config_B_alpha_{ALPHA_1}.index")
    
    # Config B2
    fusion_B2 = (ALPHA_2 * base_img_emb + (1 - ALPHA_2) * base_txt_emb)
    fusion_B2 = fusion_B2 / np.linalg.norm(fusion_B2, axis=1, keepdims=True)
    build_and_save_index(fusion_B2, f"data/index_config_B_alpha_{ALPHA_2}.index")
    
    # Config C1
    fusion_C1 = (ALPHA_1 * opt_img_emb + (1 - ALPHA_1) * opt_txt_emb)
    fusion_C1 = fusion_C1 / np.linalg.norm(fusion_C1, axis=1, keepdims=True)
    build_and_save_index(fusion_C1, f"data/index_config_C_alpha_{ALPHA_1}.index")
    
    # Config C2
    fusion_C2 = (ALPHA_2 * opt_img_emb + (1 - ALPHA_2) * opt_txt_emb)
    fusion_C2 = fusion_C2 / np.linalg.norm(fusion_C2, axis=1, keepdims=True)
    build_and_save_index(fusion_C2, f"data/index_config_C_alpha_{ALPHA_2}.index")

if __name__ == "__main__":
    main()
