# extract_features.py

import imagehash
import numpy as np
from PIL import Image, ImageOps
from typing import List, Tuple
import scipy.fftpack
from sklearn.decomposition import PCA

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import List
from skimage.feature import hog

# New DCT-only feature data type
FeatureData = Tuple[str, np.ndarray ,np.ndarray]  # (filename, dct_vector,phash)

def extract_hog_features(img: Image.Image, resize_dim=(128, 128)) -> np.ndarray:
    gray = img.convert('L').resize(resize_dim)
    gray_arr = np.asarray(gray)
    features, _ = hog(
        gray_arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def compute_dct_bands(gray_image: Image.Image, dct_size=32) -> np.ndarray:
    img_resized = gray_image.resize((dct_size, dct_size))
    img_array = np.asarray(img_resized).astype(np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(img_array, axis=0, norm="ortho"), axis=1, norm="ortho")
    
    # Larger, more informative bands:
    flat_8x8 = dct[:8, :8].flatten()
    flat_16x16 = dct[8:24, 8:24].flatten()  # mid-frequency
    flat_32x32 = dct[24:56, 24:56].flatten()  # higher frequency detail
    
    with open("data/debug/debug_features.txt", "a") as f_debug:
        f_debug.write("Flattened DCT bands:\n")
        f_debug.write("8x8 : " + str(flat_8x8) + "\n")
        f_debug.write("16x16 : " + str(flat_16x16) + "\n")
        f_debug.write("32x32 : " + str(flat_32x32) + "\n")

    # Concatenate bands
    band_vector = np.concatenate([flat_8x8, flat_16x16, flat_32x32]).astype(np.float32)
    
    with open("data/debug/debug_features.txt", "a") as f_debug:
        f_debug.write("PCA DCT bands:\n")
        f_debug.write("PCA : " + str(band_vector) + "\n")
    
    return band_vector


def compute_dct_vector(gray_image: Image.Image, dct_size: int = 16) -> np.ndarray:
    """
    Compute a flat vector of top-left DCT coefficients from a grayscale image.
    - Resize to (dct_size, dct_size)
    - Apply 2D Discrete Cosine Transform
    - Flatten top-left (e.g. 16x16) low-frequency block
    """
    return  imagehash.phash(gray_image)



def extract_features(image_data: List[Tuple[str, Image.Image, Image.Image]]) -> Tuple[List[FeatureData]]:
    """
    Extract only DCT-based structure features (no color, no phash).
    Returns:
        Tuple containing:
            - List of (filename, dct_vector) feature data
            - List of (filename, phash) perceptual hash data
    """
    features = []

    for filename, rgb, gray in image_data:
        with open("data/debug/debug_features.txt", "a") as f_debug:
            f_debug.write("Extracting features...for filename : " + filename + "\n")
        dct_vector = compute_dct_bands(gray)
        phash = compute_dct_vector(gray)
        hog_vec = extract_hog_features(gray)
        color_hist = extract_color_histogram(rgb, bins=32)
        with open("data/debug/debug_features.txt", "a") as f_debug:
            f_debug.write("phash : " + str(phash) + "\n")
        combined = np.concatenate([dct_vector, hog_vec, color_hist])
        
        with open("data/debug/debug_features.txt", "a") as f_debug:
            f_debug.write("combined : " + str(combined) + "\n")
        
        features.append((filename, combined, phash)) 
        
    return features


def log_feature_summary(features: List[FeatureData], txt_file="data/debug/debug_features2.txt", csv_file="data/features.csv") -> None:
    with open(txt_file, "w") as f_txt, open(csv_file, "w", newline="") as f_csv:
        f_csv.write("filename,domain,dct_bands,phash\n")

        for filename, dct_vec,phash in features:
            domain = filename.split('.')[0].replace('_', '.').strip()
            

            f_txt.write(f"--- {filename} ---\n")
            f_txt.write(f"Domain         : {domain}\n")
            f_txt.write(f"DCT bands    : {dct_vec}\n")
            f_txt.write(f"phash         : {phash}\n")
            f_txt.write(f"Top 5 Coeffs   : {np.round(dct_vec[:5], 2)}\n")
            f_txt.write("\n")

            f_csv.write(f"{filename},{domain},{dct_vec}\n")

def extract_color_histogram(img: Image.Image, bins=32) -> np.ndarray:
    # Convert to RGB to handle color
    img_rgb = img.convert('RGB')
    # Split channels
    r, g, b = img_rgb.split()

    # Numpy arrays
    r_arr = np.asarray(r)
    g_arr = np.asarray(g)
    b_arr = np.asarray(b)

    # Histograms
    r_hist, _ = np.histogram(r_arr, bins=bins, range=(0, 256))
    g_hist, _ = np.histogram(g_arr, bins=bins, range=(0, 256))
    b_hist, _ = np.histogram(b_arr, bins=bins, range=(0, 256))

    # Concatenate and normalize
    hist = np.concatenate([r_hist, g_hist, b_hist]).astype(np.float32)
    hist /= (hist.sum() + 1e-7)  # normalize
    return hist

# No __main__ block here – handled in cluster_logos.py
