import sys
from pathlib import Path

import cv2


def extract_frames(video_path, out_dir, count=24):
    video = cv2.VideoCapture(str(video_path))
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_numbers = [round(i * (total - 1) / (count - 1)) for i in range(count)]

    for i, frame_number in enumerate(frame_numbers, start=1):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        found, frame = video.read()
        if found:
            cv2.imwrite(str(out_dir / f'{i:02}_frame{frame_number:04}.png'), frame)

    video.release()
    print(f'Saved {count} frames to {out_dir}')


if __name__ == '__main__':
    video_path = sys.argv[1]
    out_dir = sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    extract_frames(video_path, out_dir, count)
