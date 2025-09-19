import os
import cv2

# Input and output directories (relative to project root: D:\Intelligent System\HNRS)
INPUT_DIR = "data/test_digits_conventional"
OUTPUT_DIR = "data/test_digits_inverted"

def main():
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                in_path = os.path.join(root, file)

                # Preserve subfolder structure (digit folders)
                rel_path = os.path.relpath(root, INPUT_DIR)
                out_folder = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(out_folder, exist_ok=True)

                out_path = os.path.join(out_folder, file)

                # Load in grayscale
                img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Could not read {in_path}")
                    continue

                # Invert colors
                inverted = 255 - img

                # Save to new dataset
                cv2.imwrite(out_path, inverted)
                print(f"Inverted and saved: {out_path}")

    print(f"\nDataset inversion complete. Output saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
