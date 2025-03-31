"""

Steps:
1. Read the CSV of domains/logos that succeeded.
2. Download each logo.
3. Validate filetype for SVG/ICO.
4. Save logos to disk.
"""

import os
import requests
import pandas as pd
from urllib.parse import urlparse

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
OUTPUT_CSV = "data/output.csv"
LOGO_DIR = "data/logos"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
TIMEOUT = 10  # seconds


# ----------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------
def is_valid_svg_content(content: str) -> bool:
    """
    Checks if the provided text content appears to be an SVG
    by searching for `<svg` and `<?xml` in the first ~100 chars.
    """
    return ('<svg' in content.lower() and '<?xml' in content.lower()[:100])


def is_valid_ico_content(content: bytes) -> bool:
    """
    Checks if content is a valid ICO by verifying first 4 bytes.
    ICO signature: 0x00, 0x00, 0x01, 0x00
    """
    return content[:4] == b'\x00\x00\x01\x00'

# ----------------------------------------------------------
# Helper to get file extension from URL path
# ----------------------------------------------------------
def get_file_extension(url: str) -> str:
    """
    Parses the path from the URL, extracts the file extension.
    If no extension found, defaults to `.png`.
    """
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else ".png"


def main():
    """
    Main function:
    1. Reads successful logo rows from CSV.
    2. Downloads each logo.
    3. Validates format if SVG/ICO.
    4. Saves to `data/logos/`.
    """
    # Create logo directory if not exists
    os.makedirs(LOGO_DIR, exist_ok=True)

    # Read CSV data
    df = pd.read_csv(OUTPUT_CSV)

    # Filter only successful statuses with non-null logo_url
    df = df[df["status"] == "success"]
    df = df.dropna(subset=["logo_url", "domain"])

    # Download each logo row
    for _, row in df.iterrows():
        domain = row["domain"]
        logo_url = row["logo_url"]
        ext = get_file_extension(logo_url)
        filename = f"{domain.replace('.', '_')}{ext}"
        filepath = os.path.join(LOGO_DIR, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            continue

        try:
            response = requests.get(logo_url, timeout=TIMEOUT, headers=HEADERS)
            if response.status_code == 200:
                # Validate content if .svg or .ico
                if ext == ".svg":
                    if not is_valid_svg_content(response.text):
                        print(f"⚠️ Skipped invalid SVG: {logo_url}")
                        continue
                elif ext == ".ico":
                    if not is_valid_ico_content(response.content):
                        print(f"⚠️ Skipped invalid ICO: {logo_url}")
                        continue

                # Save file
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"✅ Saved: {filepath}")
            else:
                print(f"⚠️ Failed [HTTP {response.status_code}]: {logo_url}")
        except Exception as e:
            print(f"❌ Error for {domain}: {e}")

    downloaded_count = len(os.listdir(LOGO_DIR))
    print(f"Step 2) Downloaded {downloaded_count} logos")


if __name__ == "__main__":
    main()
