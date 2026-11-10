import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_IN = Path('in')
ROOT_OUT = Path('out')

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()


def read_frames(video_path):
    video = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        found, frame = video.read()
        if not found:
            break
        frames.append(frame)
    video.release()
    return frames


def match_keypoints(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    keypoints1, features1 = sift.detectAndCompute(gray1, None)
    keypoints2, features2 = sift.detectAndCompute(gray2, None)

    matches = matcher.match(features1, features2)
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])
    return points1, points2


def estimate_pose(frame1, frame2, K):
    points1, points2 = match_keypoints(frame1, frame2)

    F, _ = cv2.findFundamentalMat(points1, points2)
    E = K.T @ F @ K

    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    return R, t


if __name__ == '__main__':
    camera, scene = sys.argv[1], sys.argv[2]
    K = np.loadtxt(ROOT_OUT / f'{camera}_K.csv', delimiter=',')
    frames = read_frames(ROOT_IN / f'{camera}_{scene}.mp4')

    for i in range(len(frames) - 1):
        R, t = estimate_pose(frames[i], frames[i + 1], K)
        print(i, 'R =', R.ravel().round(3), 't =', t.ravel().round(3))
