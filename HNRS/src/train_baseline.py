# HNRS/src/train_baseline.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
import pickle

# Build CNN model
def build_model():
    model = models.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # Paths to Kaggle dataset
    train_dir = r"D:\Intelligent System\HNRS\data\train_custom"
    test_dir  = r"D:\Intelligent System\HNRS\data\test_custom"

    # Load training + testing data
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=128,
        shuffle=True
    )

    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=128,
        shuffle=False
    )

    # Build and train model
    model = build_model()
    os.makedirs("models", exist_ok=True)

    history = model.fit(
        train_data,
        epochs=15,
        validation_data=test_data,
        callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    # Save model + training history
    model.save("models/kaggle_cnn.keras")
    print("Saved model -> models/kaggle_cnn.keras")

    with open("models/training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("Saved training history -> models/training_history.pkl")

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_data, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
