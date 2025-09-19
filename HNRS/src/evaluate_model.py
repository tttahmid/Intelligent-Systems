import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "models/mnist_cnn.keras"

def main():
    # Load MNIST test set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = (x_test / 255.0)[..., None]  # normalize + add channel

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_baseline.py first.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    # Confusion matrix
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (MNIST Test Set)")
    plt.show()

if __name__ == "__main__":
    main()
