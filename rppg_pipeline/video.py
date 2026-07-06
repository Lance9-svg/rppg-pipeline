# Video loading helpers for the rPPG MVP pipeline.

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    # Basic metadata reported by OpenCV for one video file.

    # Absolute input video path.
    video_path: str
    # Frames per second reported by the container.
    fps: float
    # Total frame count reported by the container.
    frame_count: int
    # Duration computed as frame_count / fps.
    duration_sec: float
    # Frame width in pixels.
    width: int
    # Frame height in pixels.
    height: int

    def to_dict(self) -> dict[str, object]:
        # Convert dataclass fields to a JSON/CSV friendly dict.
        return asdict(self)


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    # Read video metadata without processing every frame.

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {path}")

    # OpenCV VideoCapture opens the video container for metadata queries.
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")

        # CAP_PROP_* reads metadata without decoding all frames.
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            video_path=str(path.resolve()),
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )
    finally:
        capture.release()
