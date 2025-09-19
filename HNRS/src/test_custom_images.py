import os
import tensorflow as tf
from preprocess import preprocess_to_mnist

MODEL_PATH = "models/mnist_cnn.keras"
TEST_DIR = "HNRS/data/test_digits"

def main():
    # Load trained model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_baseline.py first.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    # Loop through test images
    for file in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, file)
        if not (file.endswith(".jpg") or file.endswith(".png")):
            continue

        try:
            x = preprocess_to_mnist(path, invert_if_needed=True, threshold=False)
            probs = model.predict(x, verbose=0)[0]
            pred = int(probs.argmax())
            print(f"{file} -> Predicted: {pred}, Confidence: {probs[pred]:.3f}")
        except Exception as e:
            print(f"Could not process {file}: {e}")

if __name__ == "__main__":
    main()
