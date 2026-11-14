from pathlib import Path

import cv2

from .grayscale import grayscale


def find_chessboard_corners(frames_dir, pattern_size):
    """Detect the inner chessboard corners in every PNG frame in `frames_dir`.

    `pattern_size` is the number of inner corners as (per row, per column).

    Returns a dict mapping each frame's path to its corners, a float32 array of
    shape (N, 1, 2) in pixel coordinates, or to None if the full pattern was
    not found in that frame.
    """
    detections = {}
    for path in sorted(Path(frames_dir).glob('*.png')):
        bgr = cv2.imread(str(path))
        if bgr is None:
            detections[path] = None
            continue

        gray = grayscale(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
        detections[path] = corners if found else None
    return detections
