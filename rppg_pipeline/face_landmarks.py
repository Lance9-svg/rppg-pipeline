# MediaPipe Tasks Face Landmarker wrapper.

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Single-subject mode keeps MediaPipe landmark smoothing enabled.
FACE_LANDMARKER_NUM_FACES = 1
# Minimum confidence for initial face detection.
MIN_FACE_DETECTION_CONFIDENCE = 0.50
# Minimum confidence that landmarks belong to a face.
MIN_FACE_PRESENCE_CONFIDENCE = 0.50
# Minimum confidence for frame-to-frame tracking.
MIN_TRACKING_CONFIDENCE = 0.50


class FaceLandmarkDetector:
    def __init__(self, model_path: str | Path) -> None:
        # Load the local MediaPipe .task model.
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Face landmarker model not found: {path}")

        # BaseOptions points MediaPipe to the model asset.
        base_options = python.BaseOptions(model_asset_path=str(path))
        # FaceLandmarkerOptions configures video-mode landmark tracking.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=FACE_LANDMARKER_NUM_FACES,
            min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        # create_from_options builds the MediaPipe task instance.
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int):
        # Wrap the RGB array in a MediaPipe image container.
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        # detect_for_video requires monotonically increasing timestamps.
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        # Release MediaPipe native resources.
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkDetector:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    # Convert normalized MediaPipe landmarks to pixel coordinates.
    points = np.full((len(landmarks), 2), np.nan, dtype=np.float32)
    for idx, landmark in enumerate(landmarks):
        x = float(landmark.x) * (width - 1)
        y = float(landmark.y) * (height - 1)
        points[idx] = [x, y]
    return points
