import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from preprocess import preprocess_to_mnist
from datetime import datetime

# For visuals (force non-interactive backend to avoid popup windows)
import matplotlib
matplotlib.use("Agg")  # disable GUI backends
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # double-check runtime backend
plt.ioff()  # disable interactive mode
import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

MODEL_PATH = "../models/mnist_cnn.keras"


# Function: Segment digits in multi-digit image

def segment_digits_from_image(img_path):
    """Segment multi-digit image into individual 28x28 digit crops."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Binarize and invert
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours (each contour should represent one digit)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right (so digits are in correct order)
    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

    digit_imgs = []
    for (x, y, w, h) in boxes:
        # Filter out small noise contours
        if w < 10 or h < 10:
            continue
        digit = thresh[y:y + h, x:x + w]
        digit = cv2.resize(digit, (28, 28))
        digit = digit.astype("float32") / 255.0
        digit_imgs.append(digit.reshape(1, 28, 28, 1))

    return digit_imgs


# Function: Predict single digit
 
def predict_single(model, img_path):
    x = preprocess_to_mnist(img_path, invert_if_needed=True, threshold=False)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs



# Main program

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict_image.py <image_or_folder_path>")
        sys.exit(1)

    path = sys.argv[1]

    # Ensure trained model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train it first with: python src/train_baseline.py")
        sys.exit(1)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Create results folder if needed
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"models/results_{timestamp}.txt"

    with open(result_file, "w") as f:

     
        # Folder of images
     
        if os.path.isdir(path):
            results = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(root, file)
                        try:
                            pred, probs = predict_single(model, img_path)
                            true_label = os.path.basename(root)

                            # Only accept numeric folder names as labels
                            if not true_label.isdigit():
                                continue

                            results.append((img_path, true_label, pred))
                            line = f"{img_path} | True: {true_label}, Pred: {pred}\n"
                            print(line.strip())
                            f.write(line)
                        except Exception as e:
                            err = f"Error with {img_path}: {e}\n"
                            print(err.strip())
                            f.write(err)

            # Accuracy summary
            total = len(results)
            correct = sum(1 for _, true, pred in results if str(true) == str(pred))
            summary = (
                f"\nTotal: {total}, Correct: {correct}, Accuracy: {correct/total:.4f}\n"
                if total > 0 else "No valid images found.\n"
            )
            print(summary.strip())
            f.write(summary)

            # VISUALS and REPORTS START 
            if total > 0:
                y_true = [int(true) for _, true, _ in results]
                y_pred = [int(pred) for _, _, pred in results]
                cm = confusion_matrix(y_true, y_pred)

                class_totals = Counter(y_true)
                class_correct = Counter([t for t, p in zip(y_true, y_pred) if t == p])

                f.write("\nPer-class accuracy:\n")
                print("\nPer-class accuracy:")
                for digit in sorted(class_totals.keys()):
                    acc = class_correct[digit] / class_totals[digit]
                    line = f"Digit {digit}: {class_correct[digit]}/{class_totals[digit]} correct ({acc:.4f})\n"
                    print(line.strip())
                    f.write(line)

                report = classification_report(y_true, y_pred, digits=4)
                print("\nClassification Report:\n", report)
                f.write("\nClassification Report:\n")
                f.write(report + "\n")

                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(10), yticklabels=range(10)
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")
                cm_path = f"models/confusion_matrix_{timestamp}.png"
                plt.savefig(cm_path)
                plt.close("all")
                print(f"Saved confusion matrix -> {cm_path}")
                f.write(f"Confusion matrix saved -> {cm_path}\n")

                max_show = min(25, total)
                plt.figure(figsize=(12, 12))
                for i, (img_path, true, pred) in enumerate(results[:max_show]):
                    plt.subplot(5, 5, i + 1)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    plt.imshow(img, cmap="gray")
                    color = "green" if str(true) == str(pred) else "red"
                    plt.title(f"T:{true} P:{pred}", color=color, fontsize=8)
                    plt.axis("off")

                plt.tight_layout()
                gallery_path = f"models/sample_preds_{timestamp}.png"
                plt.savefig(gallery_path)
                plt.close("all")
                print(f"Saved sample predictions gallery -> {gallery_path}")
                f.write(f"Sample predictions gallery saved -> {gallery_path}\n")
            #  VISUALS + REPORTS END 

    
        # Case 2: Single image (detect multi-digit)
        else:
            try:
                digit_imgs = segment_digits_from_image(path)
            except Exception as e:
                print(f"Segmentation failed: {e}")
                digit_imgs = []

            if len(digit_imgs) > 1:
                print(f"Detected {len(digit_imgs)} digits – performing multi-digit recognition...\n")
                result_digits = []
                for d in digit_imgs:
                    probs = model.predict(d, verbose=0)[0]
                    pred = int(np.argmax(probs))
                    result_digits.append(str(pred))
                final_number = "".join(result_digits)
                print(f"Predicted Number: {final_number}")
                f.write(f"Predicted multi-digit number: {final_number}\n")
            else:
                pred, probs = predict_single(model, path)
                probs_str = " ".join(f"{p:.3f}" for p in probs)
                line = f"Prediction: {pred}\nProbs (0–9): {probs_str}\n"
                print(line.strip())
                f.write(line)

                # VISUAL FOR SINGLE IMAGE 
                plt.bar(range(10), probs)
                plt.title(f"Prediction Probabilities for {os.path.basename(path)}")
                bar_path = f"models/probabilities_{timestamp}.png"
                plt.savefig(bar_path)
                plt.close("all")
                print(f"Saved probability bar chart -> {bar_path}")
                f.write(f"Probability bar chart saved -> {bar_path}\n")

    print(f"Results saved -> {result_file}")


if __name__ == "__main__":
    main()
