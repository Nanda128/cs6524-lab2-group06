import re

from mypylibs.depth import process_scene
from mypylibs.extract_frames import extract_frames
from mypylibs.intrinsics import ROOT_IN, ROOT_OUT, calibrate, save_calibration

SCENE_NAME = re.compile(r'^(\d{2})_(\d{2})\.mp4$')


def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    for video_path in sorted(ROOT_IN.glob('*_calibration.mp4')):
        camera = video_path.name.split('_')[0]
        frames_dir = ROOT_IN / f'{camera}_calibration_frames'
        if not any(frames_dir.glob('*.png')):
            extract_frames(video_path, frames_dir)

    calibrations = {}
    for frames_dir in sorted(ROOT_IN.glob('*_calibration_frames')):
        camera = frames_dir.name.split('_')[0]
        K, D, error, image_size = calibrate(frames_dir)
        save_calibration(camera, K, D)
        calibrations[camera] = (K, D, image_size)
        print(f'Camera {camera}: calibrated from {frames_dir}')
        print(f'  K =\n{K}')
        print(f'  D = {D.ravel()}')
        print(f'  mean reprojection error = {error:.4f} px')
        if error > 0.5:
            print('  WARNING: reprojection error is above the 0.5 px target')

    if not calibrations:
        raise FileNotFoundError(f'No *_calibration_frames directories found in {ROOT_IN}')

    scenes_found = set()
    for video_path in sorted(ROOT_IN.glob('*.mp4')):
        match = SCENE_NAME.match(video_path.name)
        if match is None:
            continue
        camera, scene = match.groups()
        scenes_found.add(camera)
        if camera not in calibrations:
            print(f'Scene {video_path.name}: no matching calibration frames; skipped')
            continue
        K, D, image_size = calibrations[camera]
        process_scene(camera, scene, K, D, image_size)

    for camera in sorted(calibrations):
        if camera not in scenes_found:
            print(f'Camera {camera}: no scene videos found in {ROOT_IN}')


if __name__ == '__main__':
    main()
