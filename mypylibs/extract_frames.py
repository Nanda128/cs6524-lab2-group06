import sys
from pathlib import Path

import cv2


def extract_frames(video_path, out_dir, count=24):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise OSError(f'Could not open calibration video: {video_path}')
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 1:
        video.release()
        raise ValueError('Frame count must be at least one')
    if total < count:
        video.release()
        raise ValueError(
            f'{video_path} has {total} frames; cannot extract {count} distinct frames'
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_numbers = (
        [total // 2]
        if count == 1
        else [round(i * (total - 1) / (count - 1)) for i in range(count)]
    )

    for i, frame_number in enumerate(frame_numbers, start=1):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        found, frame = video.read()
        if found:
            output_path = out_dir / f'{i:02}_frame{frame_number:04}.png'
            if not cv2.imwrite(str(output_path), frame):
                video.release()
                raise OSError(f'Could not write calibration frame: {output_path}')
        else:
            video.release()
            raise OSError(f'Could not read frame {frame_number} from {video_path}')

    video.release()
    print(f'Saved {len(frame_numbers)} frames to {out_dir}')


if __name__ == '__main__':
    video_path = sys.argv[1]
    out_dir = sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    extract_frames(video_path, out_dir, count)
