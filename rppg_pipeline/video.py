# Video and facial ROI processing

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from rppg_pipeline.degradation import ROIS, roi_shift_offsets

FACE_OVAL_IDS = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
RIGHT_EYE_IDS = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
LEFT_EYE_IDS = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
]
MOUTH_IDS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
]
FOREHEAD_IDS = [103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104]
LEFT_CHEEK_IDS = [
    234,
    93,
    132,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
    123,
    116,
    111,
    117,
    118,
    119,
    100,
    47,
]
RIGHT_CHEEK_IDS = [
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    352,
    345,
    340,
    346,
    347,
    348,
    329,
    277,
]


@dataclass(frozen=True)
class ROIMeasurement:
    # ROI color measurement
    rgb: tuple[float, float, float]
    n_pixels: int
    brightness: float
    overexposure_ratio: float
    valid: bool


class LandmarkDetector(Protocol):
    # Landmark detector interface
    def detect(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> np.ndarray | None: ...


class FaceLandmarker:
    # MediaPipe face landmarker
    def __init__(self, model_path: str | Path) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> np.ndarray | None:
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return np.array(
            [[landmark.x, landmark.y] for landmark in result.face_landmarks[0]],
            dtype=float,
        )

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> FaceLandmarker:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# Extract multiple ROI shifts
def extract_rgb_traces(
    video_path: str | Path,
    detector: LandmarkDetector,
    roi_shift_fractions: tuple[float, ...],
    max_frames: int | None = None,
) -> dict[float, pd.DataFrame]:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    rows: dict[float, list[dict[str, object]]] = {
        fraction: [] for fraction in roi_shift_fractions
    }
    previous_center: np.ndarray | None = None
    previous_area: float | None = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        while max_frames is None or frame_index < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * 1000.0 / fps)
            time_sec = frame_index / fps
            normalized = detector.detect(frame_rgb, timestamp_ms)
            base: dict[str, object] = {
                "frame_idx": frame_index,
                "time_sec": time_sec,
                "landmark_valid": normalized is not None,
            }
            if normalized is None:
                base.update(_empty_face_values())
                masks = None
                face_width = np.nan
            else:
                height, width = frame_rgb.shape[:2]
                landmarks = landmarks_to_pixels(normalized, width, height)
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                face_width = float(x_max - x_min)
                face_area = float((x_max - x_min) * (y_max - y_min))
                center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                center_shift = (
                    float(np.linalg.norm(center - previous_center) / face_width)
                    if previous_center is not None and face_width > 0
                    else 0.0
                )
                area_change = (
                    abs(face_area - previous_area) / previous_area
                    if previous_area is not None and previous_area > 0
                    else 0.0
                )
                previous_center = center
                previous_area = face_area
                base.update(
                    {
                        "face_width_px": face_width,
                        "face_area_ratio": face_area / float(width * height),
                        "face_center_shift": center_shift,
                        "face_area_change": area_change,
                    }
                )
                masks = build_roi_masks(landmarks, (height, width))

            for fraction in roi_shift_fractions:
                row = {**base, "roi_shift_fraction": fraction}
                if masks is None:
                    offset_x, offset_y = np.nan, np.nan
                    measurements = _empty_measurements()
                    retention = {roi: np.nan for roi in ROIS}
                elif fraction:
                    offset_x, offset_y = roi_shift_offsets(
                        np.array([time_sec]),
                        np.array([face_width]),
                        fraction,
                    )[0]
                    shifted_masks = shift_roi_masks(masks, offset_x, offset_y)
                    retention = {
                        roi: _roi_retention_ratio(masks[roi], shifted_masks[roi])
                        for roi in ROIS
                    }
                    measurements = measure_rois(frame_rgb, shifted_masks)
                else:
                    offset_x, offset_y = 0.0, 0.0
                    retention = {roi: 1.0 for roi in ROIS}
                    measurements = measure_rois(frame_rgb, masks)
                row.update(
                    {
                        "roi_shift_x_px": float(offset_x),
                        "roi_shift_y_px": float(offset_y),
                    }
                )
                for roi, measurement in measurements.items():
                    row.update(_measurement_columns(roi, measurement))
                    row[f"{roi}_retention_ratio"] = retention[roi]
                rows[fraction].append(row)
            frame_index += 1
    finally:
        capture.release()
    return {fraction: pd.DataFrame(values) for fraction, values in rows.items()}


# Convert landmarks to pixels
def landmarks_to_pixels(
    normalized_landmarks: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    landmarks = np.asarray(normalized_landmarks, dtype=float)
    pixels = landmarks.copy()
    pixels[:, 0] *= width - 1
    pixels[:, 1] *= height - 1
    return pixels


# Build three ROI masks
def build_roi_masks(
    landmarks_px: np.ndarray,
    frame_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    face_points = _points(landmarks_px, FACE_OVAL_IDS)
    center = np.mean(face_points, axis=0)
    inner_face = _polygon_mask(
        center + 0.82 * (face_points - center),
        frame_shape,
        convex=False,
    )

    excluded = np.zeros(frame_shape, dtype=np.uint8)
    for landmark_ids in (RIGHT_EYE_IDS, LEFT_EYE_IDS, MOUTH_IDS):
        excluded = cv2.bitwise_or(
            excluded,
            _polygon_mask(_points(landmarks_px, landmark_ids), frame_shape),
        )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    full_face = cv2.bitwise_and(
        inner_face,
        cv2.bitwise_not(cv2.dilate(excluded, kernel)),
    )
    forehead = cv2.bitwise_and(
        _polygon_mask(_points(landmarks_px, FOREHEAD_IDS), frame_shape),
        full_face,
    )
    left_cheek = cv2.bitwise_and(
        _polygon_mask(_points(landmarks_px, LEFT_CHEEK_IDS), frame_shape),
        full_face,
    )
    right_cheek = cv2.bitwise_and(
        _polygon_mask(_points(landmarks_px, RIGHT_CHEEK_IDS), frame_shape),
        full_face,
    )
    cheeks = cv2.bitwise_or(left_cheek, right_cheek)
    return {
        "full_face_inner": full_face,
        "forehead": forehead,
        "cheeks_mean": cheeks,
    }


# Shift all ROI masks
def shift_roi_masks(
    masks: dict[str, np.ndarray],
    offset_x_px: float,
    offset_y_px: float,
) -> dict[str, np.ndarray]:
    height, width = next(iter(masks.values())).shape
    transform = np.array(
        [[1.0, 0.0, offset_x_px], [0.0, 1.0, offset_y_px]],
        dtype=np.float32,
    )
    return {
        roi: cv2.warpAffine(
            mask,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for roi, mask in masks.items()
    }


# Measure ROI colors
def measure_rois(
    frame_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, ROIMeasurement]:
    measurements = {}
    for roi in ROIS:
        mask = masks[roi]
        pixels = frame_rgb[mask > 0].astype(float)
        if len(pixels) < 50:
            measurements[roi] = ROIMeasurement(
                rgb=(np.nan, np.nan, np.nan),
                n_pixels=len(pixels),
                brightness=np.nan,
                overexposure_ratio=np.nan,
                valid=False,
            )
            continue
        rgb = tuple(float(value) for value in np.mean(pixels, axis=0))
        brightness = float(
            np.mean(0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2])
        )
        measurements[roi] = ROIMeasurement(
            rgb=rgb,
            n_pixels=len(pixels),
            brightness=brightness,
            overexposure_ratio=float(np.mean(np.any(pixels >= 250.0, axis=1))),
            valid=True,
        )
    return measurements


# Build missing face values
def _empty_face_values() -> dict[str, float]:
    return {
        "face_width_px": np.nan,
        "face_area_ratio": np.nan,
        "face_center_shift": np.nan,
        "face_area_change": np.nan,
    }


# Build missing ROI values
def _empty_measurements() -> dict[str, ROIMeasurement]:
    return {
        roi: ROIMeasurement(
            rgb=(np.nan, np.nan, np.nan),
            n_pixels=0,
            brightness=np.nan,
            overexposure_ratio=np.nan,
            valid=False,
        )
        for roi in ROIS
    }


# Build ROI trace columns
def _measurement_columns(
    roi: str,
    measurement: ROIMeasurement,
) -> dict[str, object]:
    return {
        f"{roi}_r_mean": measurement.rgb[0],
        f"{roi}_g_mean": measurement.rgb[1],
        f"{roi}_b_mean": measurement.rgb[2],
        f"{roi}_n_pixels": measurement.n_pixels,
        f"{roi}_brightness": measurement.brightness,
        f"{roi}_overexposure_ratio": measurement.overexposure_ratio,
        f"{roi}_valid": measurement.valid,
    }


# Calculate retained ROI area
def _roi_retention_ratio(
    original_mask: np.ndarray,
    shifted_mask: np.ndarray,
) -> float:
    original = original_mask > 0
    retained = np.count_nonzero(original & (shifted_mask > 0))
    return float(retained / np.count_nonzero(original))


# Select landmark points
def _points(landmarks_px: np.ndarray, landmark_ids: list[int]) -> np.ndarray:
    return np.asarray(landmarks_px, dtype=float)[landmark_ids]


# Rasterize landmark polygon
def _polygon_mask(
    points: np.ndarray,
    frame_shape: tuple[int, int],
    convex: bool = True,
) -> np.ndarray:
    height, width = frame_shape
    integer_points = np.rint(points).astype(np.int32)
    integer_points[:, 0] = np.clip(integer_points[:, 0], 0, width - 1)
    integer_points[:, 1] = np.clip(integer_points[:, 1], 0, height - 1)
    mask = np.zeros(frame_shape, dtype=np.uint8)
    if convex:
        cv2.fillConvexPoly(mask, cv2.convexHull(integer_points), 255)
    else:
        cv2.fillPoly(mask, [integer_points], 255)
    return mask
