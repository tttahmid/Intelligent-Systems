# HNRS/src/evaluate_model.py
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from datetime import datetime

MODEL_PATH = r"D:\Intelligent-Systems\HNRS\models\single_digit_cnn.keras"
HISTORY_PATH = r"D:\Intelligent-Systems\HNRS\models\training_history_single.pkl"
TEST_DIR = r"D:\Intelligent-Systems\HNRS\data\test_digits_conventional"
SAVE_DIR = r"D:\Intelligent-Systems\HNRS\models"

def main():
    if not os.path.exists(MODEL_PATH):
        print(" Model not found. Train the single-digit model first.")
        return

    # Timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = tf.keras.models.load_model(MODEL_PATH)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_data = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(28, 28),
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=128,
        shuffle=False
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f"\n Custom Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}\n")

    # Save numeric results
    results_file = os.path.join(SAVE_DIR, f"evaluation_results_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write(f"Model evaluated: {MODEL_PATH}\n")
        f.write(f"Dataset: {TEST_DIR}\n")
        f.write(f"Custom Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Custom Test Loss: {test_loss:.4f}\n")
    print(f" Results saved to: {results_file}")

    # Plot training history
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
        print(f" Training plots saved to: {train_plot_path}")
    else:
        print(" Training history not found.")

    # Confusion Matrix
    y_pred = model.predict(test_data, verbose=0).argmax(axis=1)
    y_true = test_data.classes
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Single-digit Model - Custom Data)")
    cm_path = os.path.join(SAVE_DIR, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path)
    plt.show()
    print(f" Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()
