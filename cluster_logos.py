import csv
import os
import numpy as np
import imagehash
from PIL import Image
from typing import List, Tuple, Dict
from extract_features import extract_features, log_feature_summary
from extract_features2 import extract_deep_features
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

def hamming(phash1: np.ndarray, phash2: np.ndarray) -> int:
    if phash1.shape != phash2.shape:
        raise ValueError(f"phash shapes differ: {phash1.shape} vs {phash2.shape}")
    return int((phash1 != phash2).sum())

def deduplicate_features(
    features: List[FeatureData],
    similarity_threshold: float = 0.90
) -> List[FeatureData]:
    """
    Keep only one representative among near-duplicates.
    `similarity_threshold` is fraction of matching bits (0–1).
    """
    deduped = []
    hash_len = None

    for fd in features:
        _, _feat_vec, ph = fd
        if hash_len is None:
            hash_len = ph.size  # e.g. 64
        is_dup = False

        for kept in deduped:
            ph2 = kept[2]
            # compute normalized hamming
            dist = hamming(ph, ph2)
            similarity = 1.0 - (dist / hash_len)
            if similarity >= similarity_threshold:
                is_dup = True
                break

        if not is_dup:
            deduped.append(fd)

    return deduped



# ----------------------------------------------------------
# Combine DCT and pHash into a single matrix
# Scale features to zero mean/unit variance
# Reduce dimensionality with PCA
# ----------------------------------------------------------
def cluster_features(features: List[FeatureData], variance_threshold=0.90) -> List[Cluster]:
    print(f"[INFO] Converting {len(features)} features to combined vectors...")

    #############################################################################
    #combine dct and phash into a single matrix
    X = np.stack([
        np.concatenate((fd[1], fd[2]))
        for fd in features
    ], axis=0)  # shape = (n_samples, d1 + d2)
    
    #standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    X_reduced = pca.fit_transform(X_scaled)
    
    ###########################################################################
    
    ## Dendrogram visualization (Ward linkage)
    plt.figure(figsize=(10, 7))  
    plt.title("ward Dendrogram")  
    shc.dendrogram(shc.linkage(X_reduced, method='ward'))
    plt.savefig("data/dendrogram_pca.png")
    plt.show()
    
    plt.close()

    print("[INFO] Running Agglomerative Clustering...")
    model = AgglomerativeClustering(
        metric="euclidean",
        linkage="average", #or average
        distance_threshold=5,
        #distance_threshold=0.65,
        n_clusters=None
    )
    labels = model.fit_predict(X_reduced)

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
    features = extract_deep_features(images) #tuple(domain , csv , downoad histograms)
    #features = extract_features_from_paths(images)
    print(f"Step 4) Extracted {len(features)} features")

    log_feature_summary(features)  # log again if needed
    
    features = deduplicate_features(features)   
    print(f"Deduplicated {len(features)} features - {len(images) - len(features)} duplicates removed")

    clusters = cluster_features(features)
    print(f"[INFO] Found {len(clusters)} clusters")

    save_cluster_csv(clusters)
    save_cluster_folders(clusters)
    print("[DONE] Step 5) Clustering complete. CSV and folders saved.")
    
    # Plot dendrogram to compare with PCA dendrogram
    #plot_clusters_dendrogram(clusters, linkage_method='ward', title="Clusters Dendrogram", output_path="data/clusters_dendrogram.png")
