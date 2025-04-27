#!/usr/bin/env python3
#https://medium.com/data-science/fast-feature-engineering-in-python-image-data-5d3a8a7bf616
"""
deep_feature_extractor.py

Use a pretrained ResNet34 (minus its final classifier) to extract 512‑dim feature vectors
from any list of PIL images. Suitable for clustering or retrieval tasks.

Entry point:
    extract_deep_features(image_data: List[Tuple[str, PIL.Image.Image, PIL.Image.Image]]) -> List[Tuple[str, np.ndarray]]

Each tuple is (filename, feature_vector).
"""
import imagehash
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet34_Weights
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple

FeatureData = Tuple[str, np.ndarray ,np.ndarray]

def get_resnet_extractor(device: Optional[torch.device] = None) -> nn.Module:
    # Load ResNet34 with ImageNet weights
    weights = ResNet34_Weights.DEFAULT
    backbone = models.resnet34(weights=weights)
    # Drop the final fully-connected (classification) layer
    modules = list(backbone.children())[:-1]
    extractor = nn.Sequential(*modules)
    extractor.eval()
    if device:
        extractor.to(device)
    return extractor

# 2) Preprocessing pipeline matching ResNet34 expectations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=ResNet34_Weights.DEFAULT.transforms().mean,
                         std=ResNet34_Weights.DEFAULT.transforms().std),
])

def compute_phash_vec(gray_image: Image.Image, dct_size: int = 16) -> np.ndarray:
    """
    Compute a flat vector of top-left DCT coefficients from a grayscale image.
    - Resize to (dct_size, dct_size)
    - Apply 2D Discrete Cosine Transform
    - Flatten top-left (e.g. 16x16) low-frequency block
    """
    h: imagehash.ImageHash = imagehash.phash(gray_image)
    # h.hash is a 2D boolean array; flatten to 1D of 0/1 ints
    return h.hash.flatten().astype(np.uint8)

# 3) Main extraction function
def extract_deep_features(
    image_data: List[Tuple[str, Image.Image, Image.Image]],
    device: torch.device = None
) -> Tuple[List[FeatureData]]:
    """
    Args:
        image_data: list of (filename, rgb_image, gray_image)
        device: CPU or CUDA device for model & tensors
    Returns:
        List of (filename, feature_vector) where feature_vector is 512-dim numpy array
    """
    extractor = get_resnet_extractor(device)
    results: Tuple[List[FeatureData]] = []
    # We only need rgb_image for ResNet; gray channel ignored here
    with torch.no_grad():
        for filename, rgb_img, gray_img in image_data:
            # Preprocess and move to device
            inp = preprocess(rgb_img).unsqueeze(0)
            if device:
                inp = inp.to(device)
            # Forward pass
            feats = extractor(inp)                 # shape [1, 512, 1, 1]
            feats = feats.squeeze().cpu().numpy()  # [512]
            hash_vec = compute_phash_vec(gray_img)
            results.append((filename, feats.astype(np.float32),hash_vec.astype(np.uint8)))
    return results
