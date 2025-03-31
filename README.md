# Logo Clustering Pipeline

This repository contains a set of scripts that:
1. **Scrape** websites to discover potential logo URLs.
2. **Preprocess** the downloaded logos (resize, convert to a unified format).
3. **Extract features** (DCT, HOG, color histograms, pHash).
4. **Perform hierarchical clustering** on the resulting feature vectors.
5. **Generate** outputs (CSV listings, cluster folders, debug logs, dendrogram images) for analysis.

---

## Installation

1. **Clone this repository**:

   ```bash
   git clone (https://github.com/anapintilie123/LogoMiner-.git)
   cd logo_clustering_pipeline

2. **Create and activate a virtual environment (optional but recommended)**:
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

3. **Install dependencies**:

pip install -r requirements.txt
*Or if you prefer to keep a curated list, manually edit requirements.txt so it only includes the packages you actually need.

4. **Recommended Script Order**

1. **`logo_scraper.py`** – Scrapes each domain to gather potential logo URLs (stored in a CSV).
2. **`download_logos.py`** – Reads that CSV, downloads each logo file into `data/logos/`.
3. **`cluster_logos.py`** – Loads/preprocesses the logos, extracts features, and performs hierarchical clustering (final results go to CSV + cluster folders).


5. **Python version**

This pipeline has been tested on Python 3.9+ (though 3.7 or 3.8 may also work).

You can check your version with python --version.\

**Notes**
Performance: Large-scale scraping can take a while. The asynchronous approach should help keep it responsive, but be mindful of your system’s network and I/O limits.

Extensibility: You can easily add more feature extraction methods (e.g., CNN embeddings), or switch from hierarchical clustering to another algorithm if needed.

