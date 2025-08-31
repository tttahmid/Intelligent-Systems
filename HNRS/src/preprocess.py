import cv2
import numpy as np

# Pad image to a square shape before resizing
def _center_pad_to_square(img):
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=0)

def preprocess_to_mnist(path_or_img, *, invert_if_needed=True, threshold=False):
    """
    Convert an image to MNIST-like format: (1, 28, 28, 1), float32 in [0,1]
    """
    # Load image (path or already-loaded array)
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {path_or_img}")
    else:
        img = path_or_img
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pad to square so aspect ratio is preserved
    img = _center_pad_to_square(img)

    # Optional threshold (binarization for noisy scans)
    if threshold:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if needed (MNIST digits are white on black)
    if invert_if_needed and img.mean() > 127:
        img = 255 - img

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Scale to [0,1] and add batch/channel dimensions
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img
