from pathlib import Path

import cv2
import numpy as np

from .calibration import find_chessboard_corners

ROOT_IN = Path('in')
ROOT_OUT = Path('out')

PATTERN_SIZE = (8, 5)
SQUARE_SIZE_CM = 2.5


def chessboard_points(pattern_size, square_size):
    corners_per_row, rows = pattern_size
    points = []
    for y in range(rows):
        for x in range(corners_per_row):
            points.append((x * square_size, y * square_size, 0))
    return np.array(points, dtype=np.float32)


def mean_reprojection_error(board_points, image_points, rotations, translations, K, D):
    distances = []
    for corners, rotation, translation in zip(image_points, rotations, translations):
        predicted, _ = cv2.projectPoints(board_points, rotation, translation, K, D)
        difference = predicted.reshape(-1, 2) - corners.reshape(-1, 2)
        distances.extend(np.linalg.norm(difference, axis=1))
    return np.mean(distances)


def calibrate(frames_dir):
    detections = find_chessboard_corners(frames_dir, PATTERN_SIZE)
    image_points = [corners for corners in detections.values() if corners is not None]
    if len(image_points) < 3:
        raise ValueError(
            f'Need at least 3 usable calibration frames in {frames_dir}; '
            f'found {len(image_points)}'
        )

    board_points = chessboard_points(PATTERN_SIZE, SQUARE_SIZE_CM)
    object_points = [board_points] * len(image_points)

    first_image = next(
        (cv2.imread(str(path)) for path in detections if detections[path] is not None),
        None,
    )
    if first_image is None:
        raise ValueError(f'No readable calibration frames in {frames_dir}')
    height, width = first_image.shape[:2]
    for path in detections:
        image = cv2.imread(str(path))
        if image is not None and image.shape[:2] != (height, width):
            raise ValueError(f'Calibration frame dimensions differ in {frames_dir}: {path}')

    _, K, D, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, (width, height), None, None)

    error = mean_reprojection_error(board_points, image_points, rotations, translations, K, D)
    return K, D, error, (width, height)


def save_calibration(camera, K, D):
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(ROOT_OUT / f'{camera}_K.csv', K, delimiter=',', fmt='%.6f')
    np.savetxt(ROOT_OUT / f'{camera}_D.csv', D, delimiter=',', fmt='%.6f')
