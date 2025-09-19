import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore

# Build a simple CNN model for MNIST
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
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0)[..., None]  # normalize and add channel dim
    x_test  = (x_test  / 255.0)[..., None]

    model = build_model()
    os.makedirs("models", exist_ok=True)

    # Train the model
    model.fit(
    x_train, y_train,
    epochs=15,             
    batch_size=128,
    validation_split=0.1,
    callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)


    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save trained model
    model.save("models/mnist_cnn.keras")
    print("Saved model → models/mnist_cnn.keras")

if __name__ == "__main__":
    main()
