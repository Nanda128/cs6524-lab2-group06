import json

from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .pose import ROOT_IN, estimate_pose, read_frames
from .pointcloud import save_point_cloud
from .video import video_writer

ROOT_OUT = Path('out')


def rectify(frame1, frame2, K, D, R, t):
    height, width = frame1.shape[:2]
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K, D, K, D, (width, height), R, t)

    map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, (width, height), cv2.CV_32FC1)
    rectified1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)
    rectified2 = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR)

    if P2[0, 3] > 0:
        rectified1, rectified2 = rectified2, rectified1

    return rectified1, rectified2, P1[:, :3]


def depth_map(frame1, frame2, K, D, R, t, baseline_cm=None, return_geometry=False):
    left, right, rectified_K = rectify(frame1, frame2, K, D, R, t)
    focal_length = rectified_K[0, 0]

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    width = gray_left.shape[1]
    num_disparities = min(128, (width // 16) * 16)
    if num_disparities < 16:
        raise ValueError(f'Frame width is too small for stereo matching: {width}')
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 5**2,
        P2=32 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
    )
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16

    baseline = float(baseline_cm) if baseline_cm is not None else float(np.linalg.norm(t))
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(f'Baseline must be positive and finite; got {baseline}')
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    found = disparity > 0
    depth[found] = focal_length * baseline / disparity[found]
    if return_geometry:
        return depth, left, rectified_K
    return depth


def save_heatmap_video(depths, out_path, depth_unit):
    finite_values = np.concatenate([depth[np.isfinite(depth)] for depth in depths])
    if finite_values.size == 0:
        raise ValueError('No valid disparity pixels were found; cannot write a depth video')
    near, far = np.percentile(finite_values, [5, 95])
    if near == far:
        far = near + 1

    writer = None
    for depth in depths:
        figure, plot = plt.subplots(figsize=(8, 6))
        heatmap = plot.imshow(depth, cmap='turbo_r', vmin=near, vmax=far)
        plot.axis('off')
        figure.colorbar(heatmap, ax=plot, label=f'Depth ({depth_unit})')

        figure.canvas.draw()
        picture = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        plt.close(figure)

        if writer is None:
            height, width = picture.shape[:2]
            writer = video_writer(out_path, 30, (width, height))
            if not writer.isOpened():
                raise OSError(f'Could not create depth video: {out_path}')
        writer.write(cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()


def load_scene_baselines(config_path, pair_count):
    if not config_path.exists():
        return [None] * pair_count
    with config_path.open(encoding='utf-8') as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f'{config_path} must contain a JSON object')

    if 'baseline_cm_per_pair' in config:
        baselines = config['baseline_cm_per_pair']
        if not isinstance(baselines, list) or len(baselines) != pair_count:
            raise ValueError(
                f'{config_path} must contain exactly {pair_count} values in '
                '"baseline_cm_per_pair"'
            )
    elif 'baseline_cm' in config:
        baselines = [config['baseline_cm']] * pair_count
    else:
        return [None] * pair_count

    try:
        baselines = [float(value) for value in baselines]
    except (TypeError, ValueError) as error:
        raise ValueError(f'Baselines in {config_path} must be numeric') from error
    if any(not np.isfinite(value) or value <= 0 for value in baselines):
        raise ValueError(f'All baselines in {config_path} must be positive and finite')
    return baselines


def process_scene(camera, scene, K, D, expected_size):
    video_path = ROOT_IN / f'{camera}_{scene}.mp4'
    frames = read_frames(video_path)
    actual_size = (frames[0].shape[1], frames[0].shape[0])
    if actual_size != expected_size:
        raise ValueError(
            f'{video_path} is {actual_size[0]}x{actual_size[1]}, but calibration is '
            f'{expected_size[0]}x{expected_size[1]}; use the same video preprocessing'
        )

    baselines = load_scene_baselines(
        ROOT_IN / f'{camera}_{scene}.json',
        len(frames) - 1,
    )
    depths = []
    metric_baselines = []
    skipped_pairs = []
    point_cloud = None
    for i, baseline_cm in enumerate(baselines):
        try:
            R, t = estimate_pose(frames[i], frames[i + 1], K, D)
            if point_cloud is None:
                depth, rectified_frame, rectified_K = depth_map(
                    frames[i],
                    frames[i + 1],
                    K,
                    D,
                    R,
                    t,
                    baseline_cm,
                    return_geometry=True,
                )
            else:
                depth = depth_map(frames[i], frames[i + 1], K, D, R, t, baseline_cm)
            if not np.isfinite(depth).any():
                raise ValueError('Stereo matching found no valid disparity pixels')
            if point_cloud is None:
                point_cloud = (depth, rectified_frame, rectified_K)
            depths.append(depth)
            metric_baselines.append(baseline_cm is not None)
        except (ValueError, cv2.error) as error:
            skipped_pairs.append((i, str(error)))

    if not depths:
        print(f'  Scene {camera}_{scene}: no usable frame pairs; no video written')
        return

    for pair_index, reason in skipped_pairs[:5]:
        print(f'  Scene {camera}_{scene}, pair {pair_index}: skipped: {reason}')
    if len(skipped_pairs) > 5:
        print(f'  Scene {camera}_{scene}: {len(skipped_pairs) - 5} additional pairs skipped')

    depth_unit = 'cm' if all(metric_baselines) else 'relative units (scale unknown)'
    out_path = ROOT_OUT / f'{camera}-{scene}.mp4'
    save_heatmap_video(depths, out_path, depth_unit)
    print(f'Saved {out_path} ({len(depths)} frame pairs; {depth_unit})')
    cloud_path = ROOT_OUT / f'{camera}-{scene}.ply'
    save_point_cloud(cloud_path, *point_cloud, depth_unit=depth_unit)
    print(f'Saved {cloud_path} (first valid depth frame; {depth_unit})')
