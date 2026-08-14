from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_formal_conditions_are_fixed() -> None:
    from rppg_pipeline.degradation import FORMAL_CONDITIONS

    assert FORMAL_CONDITIONS == (
        "original",
        "fps15",
        "fps10",
        "roi_shift_3",
        "roi_shift_5",
    )


@pytest.mark.parametrize(
    ("target_fps", "expected_indices"),
    [
        (15.0, np.arange(0, 30, 2)),
        (10.0, np.arange(0, 30, 3)),
    ],
)
def test_select_frame_indices_uses_nearest_unique_samples(
    target_fps: float,
    expected_indices: np.ndarray,
) -> None:
    from rppg_pipeline.degradation import select_frame_indices

    time_sec = np.arange(0.0, 1.0, 1.0 / 30.0)

    selected = select_frame_indices(time_sec, 0.0, 1.0, target_fps)

    np.testing.assert_array_equal(selected, expected_indices)
    assert len(np.unique(selected)) == len(selected)


@pytest.mark.parametrize(("target_fps", "expected_count"), [(15.0, 150), (10.0, 100)])
def test_downsample_window_preserves_source_rows(
    target_fps: float,
    expected_count: int,
) -> None:
    from rppg_pipeline.degradation import downsample_window, effective_sample_rate

    source_fps = 23.976
    time_sec = np.arange(0.0, 10.0, 1.0 / source_fps)
    trace = pd.DataFrame(
        {
            "frame_idx": np.arange(len(time_sec)),
            "time_sec": time_sec,
            "value": np.arange(len(time_sec), dtype=float),
        }
    )

    sampled = downsample_window(trace, 0.0, 10.0, target_fps)

    assert len(sampled) == expected_count
    assert sampled["frame_idx"].is_unique
    assert sampled["time_sec"].is_monotonic_increasing
    np.testing.assert_array_equal(sampled["value"], sampled["frame_idx"])
    effective_fps = effective_sample_rate(sampled["time_sec"].to_numpy())
    assert effective_fps == pytest.approx(target_fps, abs=0.4)


def test_roi_shift_offsets_are_deterministic_smooth_and_bounded() -> None:
    from rppg_pipeline.degradation import roi_shift_offsets

    time_sec = np.linspace(0.0, 10.0, 301)
    face_width_px = np.full(len(time_sec), 200.0)

    first = roi_shift_offsets(time_sec, face_width_px, 0.05)
    second = roi_shift_offsets(time_sec, face_width_px, 0.05)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[0], first[-1], atol=1e-12)
    radial_fraction = np.linalg.norm(first, axis=1) / face_width_px
    assert np.max(radial_fraction) == pytest.approx(0.05, abs=1e-12)
    assert np.max(np.linalg.norm(np.diff(first, axis=0), axis=1)) < 1.0


def test_shift_roi_masks_applies_one_clipped_translation() -> None:
    from rppg_pipeline.video import shift_roi_masks

    masks = {
        "full_face_inner": _point_mask(2, 2),
        "forehead": _point_mask(4, 5),
        "cheeks_mean": _point_mask(0, 7),
    }

    shifted = shift_roi_masks(masks, offset_x_px=2.0, offset_y_px=-1.0)

    np.testing.assert_array_equal(np.argwhere(shifted["full_face_inner"]), [[1, 4]])
    np.testing.assert_array_equal(np.argwhere(shifted["forehead"]), [[3, 7]])
    assert not np.any(shifted["cheeks_mean"])


@pytest.mark.parametrize("sample_rate_hz", [15.0, 10.0])
@pytest.mark.parametrize("method", ["POS", "CHROM"])
def test_rppg_methods_run_at_degraded_rates(
    sample_rate_hz: float,
    method: str,
) -> None:
    from rppg_pipeline.rppg import estimate_hr

    expected_bpm = 72.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)

    estimate = estimate_hr(
        _synthetic_rgb(time_sec, expected_bpm / 60.0),
        sample_rate_hz,
        method,
    )

    assert estimate == pytest.approx(expected_bpm, abs=0.5)


def test_subject_candidates_keep_all_conditions_and_failures() -> None:
    from rppg_pipeline.experiment import build_subject_candidates

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    trace = _trace_frame(time_sec, _synthetic_rgb(time_sec, 1.2))
    failed_trace = trace.copy()
    failed_trace["forehead_valid"] = False
    condition_traces = {
        "original": trace,
        "fps15": trace,
        "fps10": trace,
        "roi_shift_3": trace,
        "roi_shift_5": failed_trace,
    }

    candidates = build_subject_candidates(
        "subject1",
        _reference_frame(),
        condition_traces,
    )

    assert len(candidates) == 30
    assert candidates["condition"].drop_duplicates().tolist() == list(condition_traces)
    assert not candidates.duplicated(
        ["subject", "condition", "window_id", "roi", "method"]
    ).any()
    failed = candidates[
        candidates["condition"].eq("roi_shift_5")
        & candidates["roi"].eq("forehead")
    ]
    assert len(failed) == 2
    assert failed["window_status"].eq("low_valid_fraction").all()
    assert failed["rppg_hr_bpm"].isna().all()


def _point_mask(row: int, column: int) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[row, column] = 255
    return mask


def _reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "subject1",
                "window_id": 1,
                "start_time_sec": 0.0,
                "end_time_sec": 10.0,
                "duration_sec": 10.0,
                "reference_hr_bpm": 72.0,
                "reference_valid": True,
            }
        ]
    )


def _synthetic_rgb(time_sec: np.ndarray, pulse_hz: float) -> np.ndarray:
    pulse = np.sin(2.0 * np.pi * pulse_hz * time_sec)
    motion = 0.004 * np.sin(2.0 * np.pi * 0.15 * time_sec)
    return np.column_stack(
        [
            180.0 * (1.0 + 0.003 * pulse + motion),
            140.0 * (1.0 + 0.010 * pulse + motion),
            120.0 * (1.0 + 0.005 * pulse + motion),
        ]
    )


def _trace_frame(time_sec: np.ndarray, rgb: np.ndarray) -> pd.DataFrame:
    frame: dict[str, object] = {
        "frame_idx": np.arange(len(time_sec)),
        "time_sec": time_sec,
        "landmark_valid": True,
        "face_area_ratio": 0.1,
        "face_center_shift": 0.001,
        "face_area_change": 0.002,
    }
    for roi in ("full_face_inner", "forehead", "cheeks_mean"):
        frame[f"{roi}_r_mean"] = rgb[:, 0]
        frame[f"{roi}_g_mean"] = rgb[:, 1]
        frame[f"{roi}_b_mean"] = rgb[:, 2]
        frame[f"{roi}_n_pixels"] = 1000
        frame[f"{roi}_brightness"] = np.mean(rgb, axis=1)
        frame[f"{roi}_overexposure_ratio"] = 0.0
        frame[f"{roi}_valid"] = True
    return pd.DataFrame(frame)
