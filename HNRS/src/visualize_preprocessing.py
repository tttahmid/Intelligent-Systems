# HNRS/src/invert_specific_images.py
import cv2
import os

# List of specific files to invert
FILES_TO_INVERT = [
    "single_number_1.jpg",
    "single_number_2.jpg",
    "single_number_3.jpg"
]

INPUT_DIR = "HNRS/data/test_digits"
OUTPUT_DIR = "HNRS/data/test_digits"  # overwrite in same folder

def main():
    for filename in FILES_TO_INVERT:
        path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(path):
            print(f"File not found: {filename}")
            continue

        # Load in grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {filename}")
            continue

        # Invert colors
        inverted = 255 - img

        # Save back (overwrite original)
        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, inverted)
        print(f"Inverted and saved: {out_path}")

if __name__ == "__main__":
    main()
