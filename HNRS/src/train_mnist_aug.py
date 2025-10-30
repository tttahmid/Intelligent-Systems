# train_mnist_aug.py
import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ── Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Paths
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "mnist_cnn_aug.keras")
PLOT_PATH  = os.path.join(OUT_DIR, "training_plots_mnist_aug.png")

# ── Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize to [0,1] and add channel dim
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]  # (N, 28, 28, 1)
x_test  = (x_test.astype("float32") / 255.0)[..., np.newaxis]

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test,  num_classes)

# Split a validation set
val_split = 0.1
val_size  = int(len(x_train) * val_split)
x_val, y_val = x_train[-val_size:], y_train[-val_size:]
x_train, y_train = x_train[:-val_size], y_train[:-val_size]

# ── Data Augmentation (robust to real-world)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=(0.7, 1.3),
    fill_mode="nearest",
)
datagen.fit(x_train)

# ── CNN (LeNet-ish but stronger)
def build_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model()
model.summary()

# ── Callbacks
cbs = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
]

# ── Train
batch_size = 128
epochs = 30

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
    steps_per_epoch=max(1, len(x_train) // batch_size),
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=cbs,
    verbose=1,
)

# ── Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Save final model (also saved best via checkpoint)
model.save(MODEL_PATH)

# ── Plot training curves
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"MNIST + Augmentation (test_acc={test_acc:.3f})")
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved training plot to: {PLOT_PATH}")
