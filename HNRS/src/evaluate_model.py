# HNRS/src/evaluate_model.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "models/mnist_cnn.keras"
HISTORY_PATH = "models/training_history.pkl"

def main():
    # Load MNIST test set
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = (x_test / 255.0)[..., None]  # normalize + add channel

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_baseline.py first.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # Load training history if available
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)

        # Plot training vs validation accuracy and loss
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history["accuracy"], label="Train Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("Training history not found. Re-run train_baseline.py.")

    # Confusion matrix
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (MNIST Test Set)")
    plt.show()

if __name__ == "__main__":
    main()
