# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.preprocessing import StandardScaler
# from typing import List, Tuple
# import scipy.cluster.hierarchy as shc
# from scipy.cluster.hierarchy import dendrogram, linkage
# import imagehash  
# import matplotlib.pyplot as plt
# from extract_features import extract_features, log_feature_summary
# from preprocess_logos import preprocess_logos_from_folder

# FeatureData = Tuple[str, np.ndarray, np.ndarray]
# # (filename, dct_vector, phash)

# def compute_phash(image: Image.Image) -> str:
#     """
#     Compute phash for deduplication, returning a hexadecimal string.
#     """
#     return str(imagehash.phash(image))

# # CNN Feature Extraction (Embeddings)
# def extract_cnn_embedding(image: Image.Image, model_name="resnet18"):
#     model = models.__dict__[model_name](pretrained=True)
#     model.eval()
#     embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

#     img_tensor = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         embedding = embedding_model(img_tensor)

#     return embedding.squeeze().numpy()

# def extract_features(image_paths: List[str]) -> List[Tuple[str, np.ndarray, str]]:
#     """
#     Extract CNN embeddings + phash for each image.
    
#     :param image_paths: list of file paths to images
#     :return: list of (filename, cnn_vector, phash)
#     """
#     features = []
#     for path in image_paths:
#         try:
#             img = Image.open(path).convert("RGB")
#             cnn_vector = extract_cnn_embedding(img)
#             phash_str = compute_phash(img)
#             features.append((path, cnn_vector, phash_str))  # same structure
#         except Exception as e:
#             print(f"[WARN] Could not process image {path}: {e}")
#     return features


# # 2. Clustering CNN Embeddings
# def cluster_images_cnn(image_paths: List[str], linkage="ward", distance_threshold=6):
#     print("[INFO] Extracting CNN embeddings...")
#     embeddings = []
#     for path in image_paths:
#         img = Image.open(path).convert("RGB")
#         embedding = extract_cnn_embedding(img)
#         embeddings.append(embedding)

#     embeddings = np.array(embeddings)

#     # Normalize embeddings
#     scaler = StandardScaler()
#     embeddings_scaled = scaler.fit_transform(embeddings)

#     # Clustering
#     print("[INFO] Clustering images...")
#     clustering_model = AgglomerativeClustering(
#         linkage=linkage,
#         distance_threshold=distance_threshold,
#         n_clusters=None
#     )
#     labels = clustering_model.fit_predict(embeddings_scaled)

#     return labels



# def plot_clusters_dendrogram(clusters, linkage_method='ward', title="Clusters Dendrogram", output_path=None):
#     # Flatten the cluster data to get vectors
#     vectors = np.vstack([f[1] for cluster in clusters for f in cluster])
    
#     # Perform the same scaling if you want consistent distances
#     # (But typically you'd do the same scaling used in cluster_features)
#     # For a quick plot, do:
#     scaler = StandardScaler()
#     vectors_scaled = scaler.fit_transform(vectors)
    
#     # Linkage matrix
#     Z = linkage(vectors_scaled, method=linkage_method)
    
#     # Plot
#     plt.figure(figsize=(10, 7))
#     dendrogram(Z)
#     plt.title(title)
#     plt.xlabel("Sample index")
#     plt.ylabel("Distance")
    
#     if output_path:
#         plt.savefig(output_path)
#         print(f"[INFO] Dendrogram saved to {output_path}")
#     plt.show()
#     plt.close()

# def deduplicate_features(features: List[Tuple[str, np.ndarray, str]], threshold=0.9) -> List[Tuple[str, np.ndarray, str]]:
#     """
#     Deduplicate by phash similarity. A similarity >= threshold means duplicates.
#     phash1, phash2 -> compute Hamming distance, convert to similarity.
#     """
#     def phash_similarity(phash1: str, phash2: str) -> float:
#         # Convert hex to binary
#         bin1 = bin(int(phash1, 16))[2:].zfill(64)
#         bin2 = bin(int(phash2, 16))[2:].zfill(64)
#         hamming_dist = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
#         return 1 - (hamming_dist / 64.0)

#     deduped = []
#     for feat in features:
#         filename, cnn_vec, phash_str = feat
#         # Check if it's a near-duplicate of any we already kept
#         if any(phash_similarity(phash_str, kept[2]) >= threshold for kept in deduped):
#             # It's a duplicate, skip
#             continue
#         deduped.append(feat)
#     return deduped


# def plot_clusters_dendrogram(clusters, linkage_method='ward', title="Clusters Dendrogram", output_path=None):
#     # Flatten the cluster data to get vectors
#     vectors = np.vstack([f[1] for cluster in clusters for f in cluster])
    
#     # Perform the same scaling if you want consistent distances
#     # (But typically you'd do the same scaling used in cluster_features)
#     # For a quick plot, do:
#     scaler = StandardScaler()
#     vectors_scaled = scaler.fit_transform(vectors)
    
#     # Linkage matrix
#     Z = linkage(vectors_scaled, method=linkage_method)
    
#     # Plot
#     plt.figure(figsize=(10, 7))
#     dendrogram(Z)
#     plt.title(title)
#     plt.xlabel("Sample index")
#     plt.ylabel("Distance")
    
#     if output_path:
#         plt.savefig(output_path)
#         print(f"[INFO] Dendrogram saved to {output_path}")
#     plt.show()
#     plt.close()
    
    
# if __name__ == "__main__":
    
# # Load and prepare the CNN only once at the top
#     resnet = models.resnet18(pretrained=True)
#     resnet.eval()

#     # We'll remove the last classification layer to get embeddings:
#     cnn_embedding_model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

#     # Define the transform for ImageNet-based pretrained models
#     cnn_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         ),
#     ])
        
#     folder = "data/logos/test"
        
#     # STEP 1) Gather image paths from your folder
#         # Suppose 'preprocess_logos_from_folder(folder)' returns a list of valid .png filepaths
#     images = preprocess_logos_from_folder(folder)  # or however you gather them
        
#     print(f"Step 1) Found {len(images)} images to process.")
#     # STEP 2) Extract CNN features
#     features = extract_features(images)  # now uses CNN + phash
#     print(f"Step 2) Extracted {len(features)} CNN+phash features.")

#         # STEP 3) Deduplicate (Optional)
#     features = deduplicate_features(features, threshold=0.9)
#     print(f"Step 3) Deduplicated to {len(features)} unique features.")

#         # STEP 4) Cluster the features
#     clusters = cluster_images_cnn(features, threshold=6.0)  
#     print(f"[INFO] Found {len(clusters)} clusters with threshold=6.0")

#         # STEP 5) Save or visualize
#         # e.g., save_cluster_csv(clusters) or something similar...
#         # e.g., save_cluster_folders(clusters) if you have that function.

#         # STEP 6) (Optional) Plot a dendrogram from final clusters
#     plot_clusters_dendrogram(
#         clusters,
#         linkage_method='ward',
#         title="Clusters Dendrogram (CNN)",
#         output_path="data/clusters_dendrogram_cnn.png"
#     )

#     print("[DONE] All steps completed.")    

