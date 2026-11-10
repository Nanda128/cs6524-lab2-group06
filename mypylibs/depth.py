import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .pose import ROOT_IN, ROOT_OUT, estimate_pose, read_frames
from .video import video_writer

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5)


def rectify(frame1, frame2, K, D, R, t):
    height, width = frame1.shape[:2]
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K, D, K, D, (width, height), R, t)

    map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, (width, height), cv2.CV_32FC1)
    rectified1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)
    rectified2 = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR)

    if P2[0, 3] > 0:
        rectified1, rectified2 = rectified2, rectified1

    focal_length = P1[0, 0]
    return rectified1, rectified2, focal_length


def depth_map(frame1, frame2, K, D, R, t):
    left, right, focal_length = rectify(frame1, frame2, K, D, R, t)

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16

    baseline = np.linalg.norm(t)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    found = disparity > 0
    depth[found] = focal_length * baseline / disparity[found]
    return depth


def save_heatmap_video(depths, out_path):
    near, far = np.nanpercentile(np.stack(depths), [5, 95])

    writer = None
    for depth in depths:
        figure, plot = plt.subplots(figsize=(4, 6))
        heatmap = plot.imshow(depth, cmap='turbo_r', vmin=near, vmax=far)
        plot.axis('off')
        figure.colorbar(heatmap, ax=plot, label='Depth (cm)')

        figure.canvas.draw()
        picture = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        plt.close(figure)

        if writer is None:
            height, width = picture.shape[:2]
            writer = video_writer(out_path, 30, (width, height))
        writer.write(cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))

    writer.release()


if __name__ == '__main__':
    camera, scene = sys.argv[1], sys.argv[2]
    K = np.loadtxt(ROOT_OUT / f'{camera}_K.csv', delimiter=',')
    D = np.loadtxt(ROOT_OUT / f'{camera}_D.csv', delimiter=',')
    frames = read_frames(ROOT_IN / f'{camera}_{scene}.mp4')

    depths = []
    for i in range(len(frames) - 1):
        R, t = estimate_pose(frames[i], frames[i + 1], K)
        depths.append(depth_map(frames[i], frames[i + 1], K, D, R, t))

    save_heatmap_video(depths, ROOT_OUT / f'{camera}-{scene}.mp4')
    print('Saved', ROOT_OUT / f'{camera}-{scene}.mp4')
