import tempfile
import numpy as np
import streamlit as st # type: ignore
import tensorflow as tf
from preprocess import preprocess_to_mnist

# App title
st.title("Handwritten Number Recognition (Baseline)")
st.caption("Upload a digit image (PNG/JPG). Model is trained on MNIST.")

# Load model once and cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/mnist_cnn.keras")

# File uploader
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Run prediction if an image is uploaded
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    # Preprocess and predict
    x = preprocess_to_mnist(path, invert_if_needed=True, threshold=False)
    probs = load_model().predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))

    # Show results
    st.subheader(f"Prediction: **{pred}**")
    st.write({str(i): float(f"{p:.3f}") for i, p in enumerate(probs)})
