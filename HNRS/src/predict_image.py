import os
import sys
import numpy as np
import tensorflow as tf
from preprocess import preprocess_to_mnist

MODEL_PATH = "models/mnist_cnn.keras"

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python src/predict_image.py path/to/digit.jpg")
        sys.exit(1)

    img_path = sys.argv[1]

    # Ensure trained model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train it first with: python src/train_baseline.py")
        sys.exit(1)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Preprocess input image
    x = preprocess_to_mnist(img_path, invert_if_needed=True, threshold=False)

    # Predict digit
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))

    print(f"Prediction: {pred}")
    print("Probs (0–9):", " ".join(f"{p:.3f}" for p in probs))

if __name__ == "__main__":
    main()
