# preprocess_logos.py

import os
from PIL import Image, ImageOps
from typing import List, Tuple
from io import BytesIO
import cairosvg

# Define the standard size to which all logos will be resized
STANDARD_SIZE = (128, 128)

# Data type: (filename, RGB image, Grayscale image)
ImageData = Tuple[str, Image.Image, Image.Image]

def open_image_by_type(path: str) -> Image.Image:
    """
    Opens image based on its file extension:
    - .svg files are converted to PNG using cairosvg
    - .ico files are handled with PIL and largest size is chosen
    - Other formats are opened normally
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".svg":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip().lower()
            if not ("<svg" in first_line or "<?xml" in first_line):
                raise ValueError("Not a valid SVG file: missing <svg> tag or XML header")
        png_data = cairosvg.svg2png(url=path)
        return Image.open(BytesIO(png_data)).convert("RGBA")

    elif ext == ".ico": 
        try:
            img = Image.open(path)
            img = img.convert("RGBA")  # Convert before trying to use
            return img
        except Exception as e:
            raise ValueError(f"ICO file could not be processed: {e}")

    else:
        return Image.open(path).convert("RGBA")


def preprocess_logos_from_folder(folder_path: str) -> List[ImageData]:
    """
    Loads and preprocesses logo images from a folder into memory.
    Each image is:
    - Opened and converted to RGBA (to handle transparency)
    - Alpha composited onto a white background
    - Converted to RGB (for color features)
    - Resized to a standard size
    - Grayscaled (for structure features)

    Returns:
        A list of tuples (filename, RGB image, Grayscale image)
    """
    preprocessed_images: List[ImageData] = []
    total_files = 0
    failed_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip hidden files or non-images
        if filename.startswith(".") or not os.path.isfile(file_path):
            continue
        total_files += 1

        try:
            image = open_image_by_type(file_path)

            # Place image on white background
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")

            # Resize and create grayscale version
            rgb_resized = image.resize(STANDARD_SIZE)
            gray_resized = ImageOps.grayscale(rgb_resized)

            # Store in memory
            preprocessed_images.append((filename, rgb_resized, gray_resized))
            ##rgb_resized.save(f"data/debug/{filename}_processed.png")


        except Exception as e:
            failed_files += 1
            print(f"[Warning] Failed to process {filename}: {e}")

    print(f"Successfully processed {len(preprocessed_images)} of {total_files} image files. ({failed_files} failed)")
    return preprocessed_images




# Example usage (you can call this in another script or test it directly):
if __name__ == "__main__":
    folder = "data/logos"
    images = preprocess_logos_from_folder(folder)
    print(f"Loaded and preprocessed {len(images)} images into memory.")
