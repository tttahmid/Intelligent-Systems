import os
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
import imutils

# -------------------------------------------------
# Streamlit setup
# -------------------------------------------------
st.set_page_config(page_title="HNRS Recognition System", layout="wide")
st.title("Handwritten Number Recognition System")

MODEL_PATH = r"D:\HNRS\models\single_digit_cnn.keras"

@st.cache_resource
def load_single_model():
    return tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


model = load_single_model()

st.sidebar.header("⚙️ Settings")

image_type = st.sidebar.selectbox(
    "Select Image Type",
    ["Binary Image", "RGB Image"]
)

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Single Digit", "Multi-Digit"]
)

segment_method = st.sidebar.selectbox(
    "Segmentation Method",
    ["Traditional Contours", "Connected Components"]
)

if image_type == "RGB Image":
    threshold_value = st.sidebar.slider("Binary Threshold Value", 0, 255, 128, 1)

st.sidebar.subheader("Morphological Operations")
use_dilation = st.sidebar.checkbox("Apply Dilation", value=False)
dilation_size = st.sidebar.slider("Dilation Kernel Size", 1, 7, 3, 2)
dilation_iter = st.sidebar.slider("Dilation Iterations", 1, 5, 1)

use_erosion = st.sidebar.checkbox("Apply Erosion", value=False)
erosion_size = st.sidebar.slider("Erosion Kernel Size", 1, 7, 3, 2)
erosion_iter = st.sidebar.slider("Erosion Iterations", 1, 5, 1)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

with col2:
    st.subheader("Results")

# ---------------- Helpers ----------------
def preprocess_image(img):
    """Apply preprocessing depending on user settings."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if image_type == "Binary Image":
        binary = gray.copy()
    else:
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    if use_dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
        binary = cv2.dilate(binary, kernel, iterations=dilation_iter)

    if use_erosion:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        binary = cv2.erode(binary, kernel, iterations=erosion_iter)

    return binary

def segment_digits(binary_img):
    binary = binary_img.copy()

    # Ensure digits are white (255) on black background
    if np.mean(binary) > 127:
        binary = 255 - binary

    boxes = []

    if segment_method == "Traditional Contours":
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 10 and h > 10:
                boxes.append((x, y, w, h))

    else:
        # --- Connected Components ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            if w > 10 and h > 10:
                boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes


if uploaded_file:
    file_bytes = uploaded_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    img = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
    binary = preprocess_image(img)

    with col1:
        st.image(binary, caption="Preprocessed Image", use_container_width=True)

    if model is None:
        with col2:
            st.error("Model not found at HNRS/models/single_digit_cnn.keras.")
    else:
        # ---------------- Single Digit Mode ----------------
        if model_type == "Single Digit":
            roi = cv2.resize(binary, (28, 28))
            roi_norm = roi.astype("float32") / 255.0
            roi_norm = np.expand_dims(roi_norm, (0, -1))

            probs = model.predict(roi_norm, verbose=0)[0]
            pred = np.argmax(probs)
            conf = probs[pred]

            with col2:
                if conf >= confidence_threshold:
                    st.success(f"**Predicted Digit:** {pred} (Confidence: {conf:.2f})")
                else:
                    st.warning("Low confidence prediction.")
                st.image(binary, caption="Processed Single Digit", use_container_width=True)

        else:
            boxes = segment_digits(binary)
            overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            for (x, y, w, h) in boxes:
                roi = binary[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi, (28, 28))
                roi_norm = roi_resized.astype("float32") / 255.0
                roi_norm = np.expand_dims(roi_norm, (0, -1))

                probs = model.predict(roi_norm, verbose=0)[0]
                pred = np.argmax(probs)
                conf = probs[pred]

                if conf >= confidence_threshold:
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_y = y - 10 if y - 10 > 10 else y + h + 20
                    cv2.putText(
                        overlay, str(pred),
                        (x + w // 2 - 10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

            with col2:
                st.image(overlay, caption=f"Detected {len(boxes)} Digits ({segment_method})", use_container_width=True)
