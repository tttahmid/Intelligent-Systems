# HNRS/src/evaluate_model_multi.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

MODEL_PATH = r"D:\Intelligent-Systems\HNRS\models\multi_mnist_segmentation.keras"
HISTORY_PATH = r"D:\Intelligent-Systems\HNRS\models\training_history_multi.pkl"
DATA_DIR = r"D:\Multi_digits"
SAVE_DIR = r"D:\Intelligent-Systems\HNRS\models"

def load_data():
    X = np.load(os.path.join(DATA_DIR, "combined.npy"))
    y = np.load(os.path.join(DATA_DIR, "segmented.npy"))
    X = X.astype("float32") / 255.0
    if len(X.shape) == 3:
        X = X[..., np.newaxis]
    y = y.astype("float32")

    split = int(0.8 * len(X))
    return X[split:], y[split:]

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the multi-digit model first.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    X_test, y_test = load_data()

    print(f"Loaded test data: {X_test.shape}, labels: {y_test.shape}")

    # Evaluate pixel-wise accuracy
    preds = model.predict(X_test, verbose=1)
    y_pred = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    pixel_acc = np.mean(y_pred == y_true)
    print(f"\nPixel-wise Accuracy: {pixel_acc:.4f}\n")

    # Save results
    results_file = os.path.join(SAVE_DIR, f"evaluation_results_multi_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write("Multi-Digit Evaluation Results\n")
        f.write("==============================\n")
        f.write(f"Model evaluated: {MODEL_PATH}\n")
        f.write(f"Dataset: {DATA_DIR}\n")
        f.write(f"Pixel-wise Accuracy: {pixel_acc:.4f}\n")
    print(f"Results saved to: {results_file}")

    # Plot training history if available
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["accuracy"], label="Train Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        train_plot_path = os.path.join(SAVE_DIR, f"training_plots_multi_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(train_plot_path)
        plt.show()
        print(f"Training plots saved to: {train_plot_path}")
    else:
        print("Training history not found.")

    # Visualize predictions
    for i in range(3):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(X_test[i].squeeze(), cmap="gray")
        axes[0].set_title("Input Image")

        axes[1].imshow(y_true[i], cmap="tab20")
        axes[1].set_title("Ground Truth")

        axes[2].imshow(y_pred[i], cmap="tab20")
        axes[2].set_title("Prediction")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"multi_digit_eval_{i+1}_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization -> {save_path}")

if __name__ == "__main__":
    main()
