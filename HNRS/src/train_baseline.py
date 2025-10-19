import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
# pyright: reportMissingImports=false

def build_model():
    model = models.Sequential([
        layers.Input((64, 84, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(11, (1, 1), activation='softmax')  # 11 classes: digits 0–9 + background
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_multi_digit_dataset():
    data_dir = r"D:\Multi_digits"
    X = np.load(os.path.join(data_dir, "combined.npy"))
    y = np.load(os.path.join(data_dir, "segmented.npy"))

    print("Loaded dataset:")
    print(f"Images shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X = X.astype("float32") / 255.0
    if len(X.shape) == 3:
        X = X[..., np.newaxis]

    y = y.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_multi_digit_dataset()
    model = build_model()

    # Save inside the main HNRS\models folder
    save_dir = r"D:\Intelligent-Systems\HNRS\models"
    os.makedirs(save_dir, exist_ok=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=8,
        callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    model_path = os.path.join(save_dir, "multi_mnist_segmentation.keras")
    history_path = os.path.join(save_dir, "training_history_multi.pkl")

    model.save(model_path)
    print(f"Saved model -> {model_path}")

    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"Saved training history -> {history_path}")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
