from pathlib import Path

from mypylibs import find_chessboard_corners

ROOT_IN = Path('in')
ROOT_OUT = Path('out')

# Inner corners (per row, per column) of the printed board: 9x6 squares -> 8x5 inner corners.
PATTERN_SIZE = (8, 5)


def detect_corners(camera: str):
    """Step 2: detect the chessboard's inner corners in each calibration frame of `camera`."""
    detections = find_chessboard_corners(ROOT_IN / f'{camera}_calibration_frames', PATTERN_SIZE)

    missed = [path.name for path, corners in detections.items() if corners is None]
    print(f'Camera {camera}: corners found in {len(detections) - len(missed)}/{len(detections)} frames')
    for name in missed:
        print(f'  no corners: {name}')

    return detections


def main():
    for frames_dir in sorted(ROOT_IN.glob('*_calibration_frames')):
        camera = frames_dir.name.split('_')[0]
        detect_corners(camera)


if __name__ == '__main__':
    main()
