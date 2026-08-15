from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("method", ["POS", "CHROM"])
def test_estimate_hr_recovers_synthetic_rate(method: str) -> None:
    from rppg_pipeline.rppg import estimate_hr

    sample_rate_hz = 30.0
    expected_bpm = 73.4
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    rgb = _synthetic_rgb(time_sec, expected_bpm / 60.0)

    estimate = estimate_hr(rgb, sample_rate_hz, method)

    assert estimate == pytest.approx(expected_bpm, abs=0.5)


def test_build_original_candidates_returns_six_rows_per_window() -> None:
    from rppg_pipeline.rppg import build_original_candidates

    sample_rate_hz = 30.0
    expected_bpm = 72.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    rgb = _synthetic_rgb(time_sec, expected_bpm / 60.0)
    trace = _trace_frame(time_sec, rgb)
    references = pd.DataFrame(
        [
            {
                "subject": "subject1",
                "window_id": 1,
                "start_time_sec": 0.0,
                "end_time_sec": 10.0,
                "duration_sec": 10.0,
                "reference_hr_bpm": expected_bpm,
                "reference_valid": True,
            }
        ]
    )

    candidates = build_original_candidates("subject1", references, trace)

    assert len(candidates) == 6
    assert candidates["condition"].eq("original").all()
    assert set(candidates["roi"]) == {
        "full_face_inner",
        "forehead",
        "cheeks_mean",
    }
    assert set(candidates["method"]) == {"POS", "CHROM"}
    assert candidates["window_status"].eq("ok").all()
    assert candidates["absolute_error_bpm"].lt(0.5).all()
    assert not candidates.duplicated(
        ["subject", "condition", "window_id", "roi", "method"]
    ).any()


def test_candidate_rejects_long_missing_rgb_gap() -> None:
    from rppg_pipeline.rppg import build_original_candidates

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    rgb = _synthetic_rgb(time_sec, 1.2)
    trace = _trace_frame(time_sec, rgb)
    missing = (trace["time_sec"] >= 3.0) & (trace["time_sec"] < 4.0)
    trace.loc[missing, "forehead_valid"] = False
    references = pd.DataFrame(
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

    candidates = build_original_candidates("subject1", references, trace)
    forehead = candidates[candidates["roi"].eq("forehead")]

    assert forehead["window_status"].eq("long_missing_gap").all()
    assert forehead["rppg_hr_bpm"].isna().all()


def test_candidates_use_each_windows_effective_sample_rate() -> None:
    from rppg_pipeline.rppg import build_candidates

    first_time = np.arange(0.0, 10.0, 1.0 / 30.0)
    second_time = np.arange(10.0, 20.0, 1.0 / 15.0)
    trace = pd.concat(
        [
            _trace_frame(first_time, _synthetic_rgb(first_time, 1.2)),
            _trace_frame(second_time, _synthetic_rgb(second_time, 1.2)),
        ],
        ignore_index=True,
    )
    references = pd.DataFrame(
        [
            _reference_row(1, 0.0, 10.0),
            _reference_row(2, 10.0, 20.0),
        ]
    )

    candidates = build_candidates("subject1", references, trace, "fps15")

    rates = candidates.groupby("window_id")["sample_rate_hz"].first()
    assert rates.loc[1] == pytest.approx(30.0)
    assert rates.loc[2] == pytest.approx(15.0)


def test_empty_window_keeps_six_failed_candidates() -> None:
    from rppg_pipeline.rppg import build_candidates

    time_sec = np.arange(0.0, 1.0, 1.0 / 30.0)
    trace = _trace_frame(time_sec, _synthetic_rgb(time_sec, 1.2))
    references = pd.DataFrame([_reference_row(1, 10.0, 20.0)])

    candidates = build_candidates("subject1", references, trace, "fps10")

    assert len(candidates) == 6
    assert candidates["window_status"].eq("low_video_coverage").all()
    assert candidates["n_frames"].eq(0).all()
    assert candidates["sample_rate_hz"].isna().all()
    assert candidates["rppg_hr_bpm"].isna().all()


def _synthetic_rgb(time_sec: np.ndarray, pulse_hz: float) -> np.ndarray:
    pulse = np.sin(2.0 * np.pi * pulse_hz * time_sec)
    motion = 0.004 * np.sin(2.0 * np.pi * 0.15 * time_sec)
    red = 180.0 * (1.0 + 0.003 * pulse + motion)
    green = 140.0 * (1.0 + 0.010 * pulse + motion)
    blue = 120.0 * (1.0 + 0.005 * pulse + motion)
    return np.column_stack([red, green, blue])


def _reference_row(window_id: int, start: float, end: float) -> dict[str, object]:
    return {
        "subject": "subject1",
        "window_id": window_id,
        "start_time_sec": start,
        "end_time_sec": end,
        "duration_sec": end - start,
        "reference_hr_bpm": 72.0,
        "reference_valid": True,
    }


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
