import os, tempfile
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from preprocess import preprocess_to_mnist

st.set_page_config(page_title="HNRS", layout="centered")
st.title("Handwritten Number Recognition")

with st.sidebar:
    mode = st.radio("Mode", ["Single digit", "Multi digit"], index=1)
    st.divider()
    st.subheader("Preprocessing")
    invert_if_needed = st.checkbox("Auto-invert (white paper)", value=True)
    use_threshold = st.checkbox("Binarize / threshold", value=True)

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return tf.keras.models.load_model(path)

MODEL_PATH = (
    "../models/mnist_cnn_aug.keras"
    if mode == "Single digit"
    else "../models/multi_mnist_segmentation.keras"
)
model = load_model(MODEL_PATH)

def as_float(v) -> float:
    return float(v.item() if hasattr(v, "item") else float(v))

def ctc_greedy_decode(seq_probs: np.ndarray, blank_idx: int | None = 10):
    best_idx = np.argmax(seq_probs, axis=1)
    best_conf = seq_probs[np.arange(len(best_idx)), best_idx]
    out, confs, prev = [], [], None
    for idx, conf in zip(best_idx, best_conf):
        if blank_idx is not None and idx == blank_idx:
            prev = None
            continue
        if prev is not None and idx == prev:
            continue
        out.append(int(idx))
        confs.append(as_float(conf))
        prev = idx
    return "".join(map(str, out)), confs

def temporal_smooth(seq: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return seq
    pad = k // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(seq)
    for t in range(seq.shape[0]):
        out[t] = padded[t : t + k].mean(axis=0)
    return out

def preprocess_row_28xh(path: str, invert=True, threshold=True) -> np.ndarray:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold:
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    if invert and bw.mean() > 127:
        bw = 255 - bw
    bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
    h, w = bw.shape
    new_w = max(28, int(round(28 * (w / h))))
    row28 = cv2.resize(bw, (new_w, 28), interpolation=cv2.INTER_AREA)
    return row28.astype("float32") / 255.0

def predict_scan_28x28(row28: np.ndarray, model, blank_idx: int | None):
    tile, stride = 28, 8
    W = row28.shape[1]
    seq_chunks = []
    if W <= tile:
        patch = row28
        if W < tile:
            patch = np.pad(patch, ((0, 0), (0, tile - W)), constant_values=0.0)
        x = patch[np.newaxis, ..., np.newaxis]
        out = np.squeeze(np.asarray(model.predict(x, verbose=0)))
        seq = out.mean(axis=0) if out.ndim == 3 else out
        return ctc_greedy_decode(temporal_smooth(seq, k=3), blank_idx=blank_idx)
    for x0 in range(0, W - tile + 1, stride):
        patch = row28[:, x0 : x0 + tile]
        x = patch[np.newaxis, ..., np.newaxis]
        out = np.squeeze(np.asarray(model.predict(x, verbose=0)))
        seq = out.mean(axis=0) if out.ndim == 3 else out
        seq_chunks.append(seq)
    tail = W - (((W - tile) // stride) * stride + tile)
    if tail > 0 and W > tile:
        patch = row28[:, -tile:]
        x = patch[np.newaxis, ..., np.newaxis]
        out = np.squeeze(np.asarray(model.predict(x, verbose=0)))
        seq = out.mean(axis=0) if out.ndim == 3 else out
        seq_chunks.append(seq)
    if not seq_chunks:
        return "", []
    seq_full = np.concatenate(seq_chunks, axis=0)
    seq_full = temporal_smooth(seq_full, k=3)
    return ctc_greedy_decode(seq_full, blank_idx=blank_idx)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        img_path = tmp.name

    st.image(img_path, caption="Input image", width="stretch")

    x = preprocess_to_mnist(
        img_path,
        invert_if_needed=invert_if_needed,
        threshold=use_threshold,
    )
    raw = model.predict(x, verbose=0)
    probs = np.squeeze(np.asarray(raw))

    if mode == "Single digit":
        if probs.ndim != 1 or probs.shape[0] < 10:
            st.error(f"Unexpected output shape for single-digit model: {probs.shape}")
        else:
            pred = int(np.argmax(probs))
            st.subheader(f"Prediction: **{pred}**")
            st.json({str(i): round(as_float(p), 3) for i, p in enumerate(probs)})
    else:
        pred, confs = "", []
        if probs.ndim == 2 and probs.shape[1] in (10, 11):
            blank = 10 if probs.shape[1] == 11 else None
            seq = temporal_smooth(probs, k=3)
            pred, confs = ctc_greedy_decode(seq, blank_idx=blank)
        elif probs.ndim == 3 and probs.shape[2] in (10, 11):
            blank = 10 if probs.shape[2] == 11 else None
            seq = temporal_smooth(probs.mean(axis=0), k=3)
            pred, confs = ctc_greedy_decode(seq, blank_idx=blank)
        else:
            blank = 10
            row28 = preprocess_row_28xh(
                img_path, invert=invert_if_needed, threshold=use_threshold
            )
            pred, confs = predict_scan_28x28(row28, model, blank_idx=blank)

        st.subheader(f"Prediction: **{pred or '(no digits)'}**")
        if confs:
            st.write({f"pos {i}": round(c, 3) for i, c in enumerate(confs)})

    try:
        os.remove(img_path)
    except Exception:
        pass
