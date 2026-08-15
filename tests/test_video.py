from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def test_roi_masks_return_three_formal_regions() -> None:
    from rppg_pipeline.video import build_roi_masks

    masks = build_roi_masks(_synthetic_landmarks(), (100, 100))

    assert tuple(masks) == ("full_face_inner", "forehead", "cheeks_mean")
    assert all(mask.shape == (100, 100) for mask in masks.values())
    assert all(np.count_nonzero(mask) > 50 for mask in masks.values())


def test_measure_rois_returns_rgb_means() -> None:
    from rppg_pipeline.video import build_roi_masks, measure_rois

    frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_rgb[:, :, 0] = 40
    frame_rgb[:, :, 1] = 80
    frame_rgb[:, :, 2] = 120
    masks = build_roi_masks(_synthetic_landmarks(), frame_rgb.shape[:2])

    measurements = measure_rois(frame_rgb, masks)

    assert tuple(measurements) == (
        "full_face_inner",
        "forehead",
        "cheeks_mean",
    )
    for measurement in measurements.values():
        assert measurement.rgb == (40.0, 80.0, 120.0)
        assert measurement.n_pixels > 50
        assert measurement.valid


def test_landmarks_to_pixels_uses_frame_bounds() -> None:
    from rppg_pipeline.video import landmarks_to_pixels

    normalized = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

    pixels = landmarks_to_pixels(normalized, width=101, height=81)

    np.testing.assert_allclose(pixels, [[0.0, 0.0], [50.0, 40.0], [100.0, 80.0]])


def test_extract_rgb_trace_reads_frames_and_marks_missing_face(
    tmp_path: Path,
) -> None:
    from rppg_pipeline.video import extract_rgb_traces

    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (100, 100),
    )
    for _ in range(3):
        writer.write(np.full((100, 100, 3), (120, 80, 40), dtype=np.uint8))
    writer.release()
    normalized = _synthetic_landmarks() / 99.0
    detector = _SequenceDetector([normalized, None, normalized])

    trace = extract_rgb_traces(video_path, detector, (0.0,))[0.0]

    assert len(trace) == 3
    np.testing.assert_allclose(trace["time_sec"], [0.0, 0.1, 0.2])
    assert trace["landmark_valid"].tolist() == [True, False, True]
    assert trace.loc[[0, 2], "forehead_valid"].all()
    assert not bool(trace.loc[1, "forehead_valid"])
    assert np.isnan(trace.loc[1, "forehead_r_mean"])
    assert trace.loc[0, "forehead_r_mean"] == pytest.approx(40.0, abs=3.0)


def test_extract_rgb_traces_builds_two_shifts_in_one_pass(tmp_path: Path) -> None:
    from rppg_pipeline.video import extract_rgb_traces

    video_path = tmp_path / "gradient.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (100, 100),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 2] = np.arange(100, dtype=np.uint8)
    for _ in range(6):
        writer.write(frame)
    writer.release()
    normalized = _synthetic_landmarks() / 99.0

    traces = extract_rgb_traces(
        video_path,
        _SequenceDetector([normalized] * 6),
        (0.0, 0.03, 0.05),
    )
    original = traces[0.0]
    shifted = traces[0.05]

    assert tuple(traces) == (0.0, 0.03, 0.05)
    assert original["roi_shift_x_px"].eq(0.0).all()
    assert shifted["roi_shift_fraction"].eq(0.05).all()
    assert traces[0.05].loc[5, "roi_shift_x_px"] > traces[0.03].loc[5, "roi_shift_x_px"]
    assert shifted.loc[5, "roi_shift_x_px"] == pytest.approx(
        0.05 * shifted.loc[5, "face_width_px"]
    )
    assert shifted.loc[5, "roi_shift_y_px"] == pytest.approx(0.0, abs=1e-12)
    for roi in ("full_face_inner", "forehead", "cheeks_mean"):
        retention = shifted.loc[5, f"{roi}_retention_ratio"]
        assert 0.0 < retention < 1.0
    assert (
        shifted.loc[5, "full_face_inner_r_mean"]
        > original.loc[5, "full_face_inner_r_mean"]
    )


def test_extract_rgb_traces_opens_video_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rppg_pipeline import video

    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (100, 100),
    )
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    real_capture = cv2.VideoCapture
    opened_paths: list[str] = []

    def open_capture(path: str) -> cv2.VideoCapture:
        opened_paths.append(path)
        return real_capture(path)

    monkeypatch.setattr(video.cv2, "VideoCapture", open_capture)
    normalized = _synthetic_landmarks() / 99.0

    video.extract_rgb_traces(video_path, _SequenceDetector([normalized]), (0.0,))

    assert opened_paths == [str(video_path)]


def test_face_landmarker_uses_cpu_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    from mediapipe.tasks import python

    from rppg_pipeline import video

    captured = {}
    monkeypatch.setattr(
        video.vision.FaceLandmarker,
        "create_from_options",
        lambda options: captured.setdefault("options", options),
    )

    video.FaceLandmarker("face_landmarker.task")

    assert captured["options"].base_options.delegate is python.BaseOptions.Delegate.CPU


def _synthetic_landmarks() -> np.ndarray:
    from rppg_pipeline.video import (
        FACE_OVAL_IDS,
        FOREHEAD_IDS,
        LEFT_CHEEK_IDS,
        LEFT_EYE_IDS,
        MOUTH_IDS,
        RIGHT_CHEEK_IDS,
        RIGHT_EYE_IDS,
    )

    landmarks = np.full((478, 2), 50.0)
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(FACE_OVAL_IDS), endpoint=False)
    landmarks[FACE_OVAL_IDS, 0] = 50.0 + 36.0 * np.cos(angles)
    landmarks[FACE_OVAL_IDS, 1] = 50.0 + 44.0 * np.sin(angles)
    _place_polygon(landmarks, FOREHEAD_IDS, 30.0, 25.0, 70.0, 43.0)
    _place_polygon(landmarks, LEFT_CHEEK_IDS, 18.0, 48.0, 44.0, 76.0)
    _place_polygon(landmarks, RIGHT_CHEEK_IDS, 56.0, 48.0, 82.0, 76.0)
    _place_polygon(landmarks, LEFT_EYE_IDS, 30.0, 40.0, 43.0, 48.0)
    _place_polygon(landmarks, RIGHT_EYE_IDS, 57.0, 40.0, 70.0, 48.0)
    _place_polygon(landmarks, MOUTH_IDS, 40.0, 65.0, 60.0, 75.0)
    return landmarks


def _place_polygon(
    landmarks: np.ndarray,
    indices: list[int],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, len(indices), endpoint=False)
    landmarks[indices, 0] = (x_min + x_max) / 2 + (x_max - x_min) / 2 * np.cos(angles)
    landmarks[indices, 1] = (y_min + y_max) / 2 + (y_max - y_min) / 2 * np.sin(angles)


class _SequenceDetector:
    def __init__(self, results: list[np.ndarray | None]) -> None:
        self.results = iter(results)

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        return next(self.results)
