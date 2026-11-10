import numpy as np


def grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image of shape (H, W, 3) to an 8-bit grayscale image (H, W).

    Uses the ITU-R BT.601 luma weights. Channels must be in RGB order:
    cv2.imread returns BGR, so convert with cv2.COLOR_BGR2RGB first.

    The weighted sum is float64, so it is rounded back to uint8, which is the
    type cv2.findChessboardCorners and most other OpenCV functions expect.
    """
    gray = (
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    )
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
