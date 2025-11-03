# HNRS/src/visualize_preprocessing.py
import os
import cv2

# Mode: "folder" to invert all images, "specific" to invert selected files
MODE = "specific"  # change to "folder" when batch inverting

# Folder paths
INPUT_DIR = r"D:\Intelligent-Systems\HNRS\data\test_digits"
OUTPUT_DIR = r"D:\Intelligent-Systems\HNRS\data\test_digits"  # overwrite same folder

# Files to invert (only used if MODE == "specific")
FILES_TO_INVERT = [
    "single_number_1.jpg",
    "single_number_2.jpg",
    "single_number_3.jpg"
]

def invert_image(input_path, output_path):
    """Invert a single grayscale image and save it."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {input_path}")
        return False
    inverted = 255 - img
    cv2.imwrite(output_path, inverted)
    return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if MODE.lower() == "folder":
        print(f"Inverting all images in {INPUT_DIR}")
        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(INPUT_DIR, filename)
                output_path = os.path.join(OUTPUT_DIR, filename)
                if invert_image(input_path, output_path):
                    print(f"Inverted -> {filename}")

    elif MODE.lower() == "specific":
        print("Inverting selected files...")
        for filename in FILES_TO_INVERT:
            input_path = os.path.join(INPUT_DIR, filename)
            if not os.path.exists(input_path):
                print(f"File not found: {filename}")
                continue
            output_path = os.path.join(OUTPUT_DIR, filename)
            if invert_image(input_path, output_path):
                print(f"Inverted and saved: {filename}")

    else:
        print("Invalid MODE. Use 'folder' or 'specific'.")

    print("Inversion complete.")

if __name__ == "__main__":
    main()
