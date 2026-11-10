import sys

import cv2

from .video import video_writer

SIZE = (360, 640)
FPS = 30


def shrink_video(video_path, out_path):
    video = cv2.VideoCapture(str(video_path))
    original_fps = video.get(cv2.CAP_PROP_FPS)
    keep_every = round(original_fps / FPS)

    writer = video_writer(out_path, FPS, SIZE)

    frame_number = 0
    while True:
        found, frame = video.read()
        if not found:
            break
        if frame_number % keep_every == 0:
            writer.write(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
        frame_number += 1

    video.release()
    writer.release()
    print(f'Saved {out_path}')


if __name__ == '__main__':
    shrink_video(sys.argv[1], sys.argv[2])
