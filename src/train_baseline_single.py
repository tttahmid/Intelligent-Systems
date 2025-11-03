import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pickle


def build_model():
    model = models.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
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

def load_dataset():
    data_dir = r"HNRS/data/merged_training_data"
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2   #add validation split
    )

    # Training subset
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(28,28),       
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=128,
        shuffle=True,
        subset="training"           
    )

    # Validation subset
    test_data = datagen.flow_from_directory(
        data_dir,
        target_size=(28,28),      
        color_mode="grayscale",
        class_mode="sparse",
        batch_size=128,
        shuffle=False,
        subset="validation"        
    )

    return train_data, test_data

def main():
    print("Training SINGLE-DIGIT classification model...")
    train_data, test_data = load_dataset()
    model = build_model()

    save_dir = r"HNRS/models"
    os.makedirs(save_dir, exist_ok=True)

    history = model.fit(
        train_data,
        epochs=50,
        validation_data=test_data,
        callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    model_path = os.path.join(save_dir, "single_digit_cnn.keras")
    model.save(model_path)
    print(f"Saved model -> {model_path}")

    with open(os.path.join(save_dir, "training_history_single.pkl"), "wb") as f:
        pickle.dump(history.history, f)
    print("Saved training history -> training_history_single.pkl")

    test_loss, test_acc = model.evaluate(test_data, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
