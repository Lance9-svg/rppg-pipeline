from __future__ import annotations

import math

import numpy as np

from rppg_pipeline.rgb_trace import build_valid_segments, build_wide_rgb_trace
from rppg_pipeline.roi import (
    FACE_OVAL_IDS,
    FOREHEAD_IDS,
    LEFT_CHEEK_IDS,
    MOUTH_EXCLUDE_IDS,
    RIGHT_CHEEK_IDS,
    ROIConfig,
    bgr_to_rgb,
    create_roi_masks,
    frame_timestamp,
    make_cheeks_mean,
    make_invalid_measurement,
    measure_roi,
)


def test_bgr_to_rgb_channel_order() -> None:
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)
    assert rgb.tolist() == [[[30, 20, 10]]]


def test_frame_timestamp_uses_frame_index_and_fps() -> None:
    time_sec, timestamp_ms = frame_timestamp(10, 29.264106)
    assert math.isclose(time_sec, 10 / 29.264106)
    assert timestamp_ms == int(10 * 1000 / 29.264106)


def test_roi_masks_are_intersected_with_full_face_inner() -> None:
    landmarks = _synthetic_landmarks()
    config = ROIConfig()
    masks, base_masks = create_roi_masks(landmarks, (120, 120), config)

    for roi in ("forehead", "left_cheek", "right_cheek"):
        assert np.count_nonzero(masks[roi]) > 0
        outside_inner = np.count_nonzero(
            (masks[roi] > 0) & (masks["full_face_inner"] == 0)
        )
        assert outside_inner == 0
        assert np.count_nonzero(masks[roi]) <= np.count_nonzero(base_masks[roi])


def test_cheeks_mean_requires_both_cheeks_valid() -> None:
    landmarks = _synthetic_landmarks()
    frame = np.full((120, 120, 3), 100, dtype=np.uint8)
    config = ROIConfig()
    masks, base_masks = create_roi_masks(landmarks, (120, 120), config)
    face = _face_stub()

    left = measure_roi(
        frame,
        "left_cheek",
        masks["left_cheek"],
        base_masks["left_cheek"],
        True,
        face,
        config,
    )
    right = make_invalid_measurement("right_cheek", "test_invalid")
    cheeks = make_cheeks_mean(left, right)

    assert not cheeks.roi_valid
    assert np.isnan(cheeks.r_mean)


def test_cheeks_mean_averages_only_when_both_cheeks_valid() -> None:
    landmarks = _synthetic_landmarks()
    frame = np.full((120, 120, 3), [100, 110, 120], dtype=np.uint8)
    config = ROIConfig()
    masks, base_masks = create_roi_masks(landmarks, (120, 120), config)
    face = _face_stub()

    left = measure_roi(
        frame,
        "left_cheek",
        masks["left_cheek"],
        base_masks["left_cheek"],
        True,
        face,
        config,
    )
    right = measure_roi(
        frame,
        "right_cheek",
        masks["right_cheek"],
        base_masks["right_cheek"],
        True,
        face,
        config,
    )
    cheeks = make_cheeks_mean(left, right)

    assert cheeks.roi_valid
    assert cheeks.r_mean == 100
    assert cheeks.g_mean == 110
    assert cheeks.b_mean == 120


def test_valid_segments_require_minimum_duration() -> None:
    import pandas as pd

    rows = [
        {
            "frame_idx": idx,
            "time_sec": idx / 10,
            "roi": "cheeks_mean",
            "roi_valid": True,
        }
        for idx in range(100)
    ]
    rows.append(
        {
            "frame_idx": 100,
            "time_sec": 10.0,
            "roi": "cheeks_mean",
            "roi_valid": False,
            "invalid_reason": "gap",
        }
    )
    trace = pd.DataFrame(rows)
    config = ROIConfig(min_valid_segment_duration_sec=10.0)
    segments = build_valid_segments(trace, fps=10.0, config=config)

    assert len(segments) == 1
    assert bool(segments.iloc[0]["is_long_enough"])
    assert segments.iloc[0]["n_frames"] == 100


def test_wide_rgb_trace_has_one_strict_timestamp_per_frame() -> None:
    import pandas as pd

    rows = []
    for frame_idx in range(3):
        for roi in ("left_cheek", "right_cheek"):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "time_sec": frame_idx / 10,
                    "timestamp_ms": int(frame_idx * 100),
                    "roi": roi,
                    "r_mean": 1.0,
                    "g_mean": 2.0,
                    "b_mean": 3.0,
                    "n_pixels": 10,
                    "fill_ratio": 1.0,
                    "brightness": 2.0,
                    "overexposure_ratio": 0.0,
                    "landmark_available": True,
                    "roi_valid": True,
                    "invalid_reason": "",
                    "face_area_ratio": 0.1,
                    "touches_frame_border": False,
                    "bbox_center_x": 50.0,
                    "bbox_center_y": 60.0,
                    "bbox_center_jump": np.nan,
                    "bbox_area_change": np.nan,
                }
            )
    wide = build_wide_rgb_trace(pd.DataFrame(rows))

    assert len(wide) == 3
    assert wide["timestamp_ms"].is_monotonic_increasing
    assert wide["timestamp_ms"].is_unique
    assert "left_cheek_r_mean" in wide.columns
    assert "right_cheek_r_mean" in wide.columns


def _synthetic_landmarks() -> np.ndarray:
    points = np.full((478, 2), [60.0, 60.0], dtype=np.float32)
    for idx, landmark_id in enumerate(FACE_OVAL_IDS):
        angle = 2 * np.pi * idx / len(FACE_OVAL_IDS)
        points[landmark_id] = [60 + 36 * np.cos(angle), 60 + 48 * np.sin(angle)]
    _fill_region(points, FOREHEAD_IDS, 44, 28, 76, 44)
    _fill_region(points, LEFT_CHEEK_IDS, 28, 54, 50, 80)
    _fill_region(points, RIGHT_CHEEK_IDS, 70, 54, 92, 80)
    _fill_region(points, MOUTH_EXCLUDE_IDS, 48, 78, 72, 92)
    return points


def _fill_region(
    points: np.ndarray,
    ids: list[int],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> None:
    corners = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )
    for offset, landmark_id in enumerate(ids):
        points[landmark_id] = corners[offset % len(corners)]


def _face_stub():
    from rppg_pipeline.roi import FaceGeometry

    return FaceGeometry(
        bbox=(24, 12, 96, 108),
        bbox_center=(60.0, 60.0),
        bbox_area=7081.0,
        face_area_ratio=7081.0 / (120 * 120),
        touches_frame_border=False,
    )
