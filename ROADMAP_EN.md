# Scraping Pipeline Architecture

I started with a **simple, sequential script**, but it became slow and complicated to handle exceptions (redirects, bot-detection) once I reached hundreds of domains. I then moved to an **asynchronous approach** using `asyncio` plus a thread pool to run multiple HTTP requests in parallel, making it easier to handle redirection (HTTP->HTTPS, subdomains) and custom headers to avoid bot detection.

## Image Preprocessing

- **Diverse formats**: `.svg`, `.ico`, `.png` – some were large, some had transparency.
- **Solution**: Convert everything to a unified format (RGBA, 128×128). If the image is transparent, it’s composited on a white background.
- **cairosvg** is used for `.svg`; ICO files are opened with Pillow (PIL).
- **Goal**: Have consistent dimensions and color channels, simplifying the next steps of feature extraction.

## Feature Extraction

- **Earlier attempts**: 
  - Combining DCT + color histograms with various weights (e.g., 70% DCT, 30% histogram) didn’t yield consistent results.
  - Using only DCT caused issues with color-similar logos.  
- **Current approach**: A combined vector of DCT (8×8, 16×16, 32×32 blocks), HOG (for shapes/contours), a color histogram (R, G, B), and pHash (for near-duplicate detection). The result is normalized via `StandardScaler` to avoid DCT overshadowing other features.

## Hierarchical Clustering

- **Why not K-Means or DBSCAN**:
  - K-Means requires a fixed number of clusters upfront; I didn’t know the right k.
  - DBSCAN is highly sensitive to parameters (eps, min_samples) and can label many logos as noise.
- **Agglomerative Clustering**:
  - I can set a `distance_threshold` to “cut” the dendrogram at a flexible level.
  - The hierarchical structure is more intuitive to interpret. 
  - Tested both Ward and Average linkage, with Ward creating more compact clusters.

## Deduplication & Final Outputs

- **Deduplication**: pHash is used to quickly identify near-identical logos. If similarity > 0.9, they’re treated as duplicates.
- **CSV & Folders**: 
  - A CSV file `(filename, domain, cluster_id)` lets me see which domain belongs to which cluster.
  - Each cluster has its own folder so I can visually inspect if the grouping is accurate.

---

# Debugging and Results Analysis

- **Debug files**: 
  - Include DCT values, color histograms, and HOG details (for diagnosing any anomalies).
  - Dendrogram images (`dendrogram_pca.png`, `clusters_dendrogram.png`) show how logos merge step by step, validating that similar forms/colors unite first.
- **Outcome**:
  - A CSV for referencing `(filename, domain, cluster_id)`.
  - Cluster folders for direct visual checks.
- **Interpretation**:
  - If several similarly-colored logos end up together, it confirms the color histogram and DCT captured a common pattern.
  - If something was supposed to be “close” but ended up far, the debug files let me check HOG or pHash usage.  
- **Conclusion**: 
  - The pipeline effectively separates near-identical images and groups logos by frequency, color, and contours. 
  - Hierarchical clustering is flexible, letting me cut at various distances for finer or broader grouping, all while maintaining speed (async scraping) and robust feature extraction (DCT, HOG, color histogram, pHash).
