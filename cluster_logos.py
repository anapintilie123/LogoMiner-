import csv
import os
import numpy as np
import imagehash
from PIL import Image
from typing import List, Tuple, Dict
from extract_features import extract_features, log_feature_summary
from preprocess_logos import preprocess_logos_from_folder
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# ----------------------------
# TYPE DEFINITIONS
# ----------------------------
FeatureData = Tuple[str, np.ndarray,np.ndarray]  # (filename 0, dct_vector 1, phash 2)

#FeatureData = Tuple[str, imagehash.ImageHash, np.ndarray]  # (filename, phash, color_hist)
Cluster = List[FeatureData]

def dct_bands_to_vector(f: FeatureData) -> np.ndarray:
    return f[1].astype(np.float32)


def hamming(phash1: str, phash2: str) -> float:
    """Calculate similarity between two hex-formatted perceptual hashes."""
    
    # Ensure they are strings
    phash1 = str(phash1).strip()
    phash2 = str(phash2).strip()

    bin1 = bin(int(phash1, 16))[2:].zfill(64)
    bin2 = bin(int(phash2, 16))[2:].zfill(64)
    
    # Calculate Hamming distance
    hamming_dist = sum(bit1 != bit2 for bit1, bit2 in zip(bin1, bin2))
    
    # Normalize similarity score (1 = identical, 0 = completely different)
    similarity = 1 - (hamming_dist / 64)
    
    return similarity

def deduplicate_features(features :List[FeatureData] ,dct_threshold: float = 0.90) -> List[FeatureData]:
    deduped = []
    
    for feature in features:
        is_duplicate = False
        for kept in deduped:
            if hamming(feature[2], kept[2]) > dct_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(feature)
    return deduped

def cluster_features(features: List[FeatureData], threshold=0.75) -> List[Cluster]:
    print(f"[INFO] Converting {len(features)} features to combined vectors...")
    
    # After all features collected
    vectors = np.array([f[1] for f in features])  # f[1] = full_vector
    
    #we should normalise the vectors
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)
    
    with open("data/debug/debug_normalised_features.txt", "a") as f_debug:
        # Write shape and a small sample
        f_debug.write("Shape of vectors: " + str(vectors.shape) + "\n")
        f_debug.write("Number of elements: " + str(vectors.size) + "\n\n")

        # Option 1: Write only the first few rows
        f_debug.write("First 5 rows of vectors:\n")
        f_debug.write(str(vectors[:5]) + "\n")
        
    #pca
    n_components = min(10, vectors_scaled.shape[0], vectors_scaled.shape[1])
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(vectors_scaled)

    # Rebuild FeatureData with reduced vectors
    features = [(f[0], reduced_vector, f[2]) for f, reduced_vector in zip(features, reduced_vectors)]

    
    ## Dendrogram visualization (Ward linkage)
    plt.figure(figsize=(10, 7))  
    plt.title("ward Dendrogram")  
    shc.dendrogram(shc.linkage(reduced_vectors, method='ward'))
    plt.show()
    plt.savefig("data/dendrogram_pca.png")
    plt.close()

    print("[INFO] Running Agglomerative Clustering...")
    model = AgglomerativeClustering(
        metric="euclidean",
        linkage="ward", #or average
        distance_threshold=9.5,
        #distance_threshold=0.65,
        n_clusters=None
    )
    labels = model.fit_predict(reduced_vectors)

    clusters_dict: Dict[int, Cluster] = {}
    for label, feature in zip(labels, features):
        clusters_dict.setdefault(label, []).append(feature)

    return list(clusters_dict.values())

# ----------------------------
# OUTPUT
# ----------------------------
def save_cluster_csv(clusters: List[Cluster], output_file="data/clustered_logos.csv") -> None:
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "domain", "cluster_id"])

        for cluster_id, cluster in enumerate(clusters):
            for filename, *_ in cluster:
                domain = filename.split('.')[0].replace('_', '.').strip()
                writer.writerow([filename, domain, cluster_id])

def save_cluster_folders(clusters: List[Cluster], output_dir="data/logo_clusters") -> None:
    os.makedirs(output_dir, exist_ok=True)

    for idx, cluster in enumerate(clusters):
        folder = os.path.join(output_dir, f"cluster_{idx}")
        os.makedirs(folder, exist_ok=True)

        for filename, *_ in cluster:
            src_path = os.path.join("data/logos", filename)
            dst_path = os.path.join(folder, filename)
            if os.path.exists(src_path):
                try:
                    Image.open(src_path).save(dst_path)
                except Exception as e:
                    print(f"[Warning] Failed to copy {filename}: {e}")




def plot_clusters_dendrogram(clusters: List[Cluster], linkage_method='average', title="Clusters Dendrogram", output_path=None):
    """
    Plots dendrogram from clustered FeatureData.

    :param clusters: List of clusters (where each cluster is a list of FeatureData tuples).
    :param linkage_method: Linkage method to use ('ward', 'average', etc.).
    :param title: Plot title.
    :param output_path: Path to save plot (optional).
    """

    # Correctly extract numeric vectors from clusters explicitly
    vectors = np.vstack([feature[1] for cluster in clusters for feature in cluster])

    # Compute linkage matrix correctly
    Z = linkage(vectors, method=linkage_method)

    # Plot dendrogram explicitly
    plt.figure(figsize=(12, 7))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Distance")

    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Dendrogram saved to {output_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    folder = "data/logos"
    images = preprocess_logos_from_folder(folder) # png, vg, ico-> pngs
    print(f"Step 3) Preprocessed {len(images)} logos")
    features = extract_features(images) #tuple(domain , csv , downoad histograms)
    print(f"Step 4) Extracted {len(features)} features")

    log_feature_summary(features)  # log again if needed
    
    features = deduplicate_features(features)   
    print(f"Deduplicated {len(features)} features - {len(images) - len(features)} duplicates removed")
    
    

    clusters = cluster_features(features, threshold=0.6)
    print(f"[INFO] Found {len(clusters)} clusters")

    save_cluster_csv(clusters)
    save_cluster_folders(clusters)
    print("[DONE] Step 5) Clustering complete. CSV and folders saved.")
    
    # Plot dendrogram to compare with PCA dendrogram
    plot_clusters_dendrogram(clusters, linkage_method='ward', title="Clusters Dendrogram", output_path="data/clusters_dendrogram.png")
