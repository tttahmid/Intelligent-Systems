# HNRS/src/evaluate_model.py
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import Counter
import numpy as np
from datetime import datetime


MODEL_PATH = r"HNRS/models/single_digit_cnn.keras"
HISTORY_PATH = r"HNRS/models/training_history_single.pkl"
SAVE_DIR = r"HNRS/models"


def main():
    if not os.path.exists(MODEL_PATH):
        print(" Model not found. Train the single-digit model first.")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading TensorFlow MNIST test dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)

    print(f"MNIST test data loaded: {x_test.shape[0]} samples")


    print("\nEvaluating model on MNIST test data...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nMNIST Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}\n")


    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)

    cm = confusion_matrix(y_test, y_pred)
    class_totals = Counter(y_test)
    class_correct = Counter([t for t, p in zip(y_test, y_pred) if t == p])

    print("=" * 50)
    print("PER-CLASS ACCURACY:")
    print("=" * 50)
    for digit in range(10):
        total = class_totals.get(digit, 0)
        correct = class_correct.get(digit, 0)
        acc = (correct / total * 100) if total > 0 else 0.0
        print(f"Digit {digit}: {correct}/{total} correct ({acc:.2f}%)")
    print("=" * 50 + "\n")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, digits=4, target_names=[str(i) for i in range(10)]))


    results_file = os.path.join(SAVE_DIR, f"evaluation_results_MNIST_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write(f"Model evaluated: {MODEL_PATH}\n")
        f.write("Dataset: TensorFlow MNIST (Test Set)\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("PER-CLASS ACCURACY:\n" + "="*50 + "\n")
        for digit in range(10):
            total = class_totals.get(digit, 0)
            correct = class_correct.get(digit, 0)
            acc = (correct / total * 100) if total > 0 else 0.0
            f.write(f"Digit {digit}: {correct}/{total} correct ({acc:.2f}%)\n")
        f.write("\n" + "="*50 + "\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred, digits=4, target_names=[str(i) for i in range(10)]))
    print(f"Results saved to: {results_file}")

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)

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

        train_plot_path = os.path.join(SAVE_DIR, f"training_plots_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(train_plot_path)
        plt.show()
        print(f"Training plots saved to: {train_plot_path}")
    else:
        print("Training history not found.")


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Single-digit Model on MNIST Test Set)")
    cm_path = os.path.join(SAVE_DIR, f"confusion_matrix_MNIST_{timestamp}.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"🧾 Confusion matrix saved to: {cm_path}")



if __name__ == "__main__":
    main()
