from pathlib import Path

import cv2
import numpy as np

ROOT_IN = Path('in')


def read_frames(video_path):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise OSError(f'Could not open video: {video_path}')

    frames = []
    while True:
        found, frame = video.read()
        if not found:
            break
        frames.append(frame)
    video.release()
    if len(frames) < 2:
        raise ValueError(f'Video must contain at least two readable frames: {video_path}')
    return frames


def match_keypoints(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, features1 = sift.detectAndCompute(gray1, None)
    keypoints2, features2 = sift.detectAndCompute(gray2, None)
    if features1 is None or features2 is None:
        raise ValueError('Could not find SIFT features in both frames')

    matches = cv2.BFMatcher().knnMatch(features1, features2, k=2)
    good_matches = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good_matches) < 8:
        raise ValueError(f'Only {len(good_matches)} reliable feature matches; need at least 8')

    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches])
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches])
    return points1, points2


def estimate_pose(frame1, frame2, K, D=None):
    points1, points2 = match_keypoints(frame1, frame2)

    if D is not None:
        points1 = cv2.undistortPoints(points1.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        points2 = cv2.undistortPoints(points2.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    E, mask = cv2.findEssentialMat(
        points1,
        points2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if E is None or mask is None:
        raise ValueError('Could not estimate an essential matrix for this frame pair')

    inliers, R, t, _ = cv2.recoverPose(E, points1, points2, K, mask=mask)
    if inliers < 8:
        raise ValueError(f'Pose recovery retained only {inliers} inlier matches')
    return R, t
