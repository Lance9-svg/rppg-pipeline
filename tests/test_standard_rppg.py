from __future__ import annotations

import numpy as np
import pandas as pd

from rppg_pipeline.standard_rppg import (
    StandardRPPGConfig,
    build_method_roi_metrics,
    build_window_quality,
    estimate_spectral_hr,
    extract_standard_bvp,
    process_subject_windows,
)


def test_standard_pos_and_chrom_recover_synthetic_rate() -> None:
    sample_rate_hz = 30.0
    expected_bpm = 73.4
    time_sec = np.arange(0.0, 30.0, 1.0 / sample_rate_hz)
    rgb = _synthetic_rgb(time_sec, expected_bpm / 60.0)
    config = StandardRPPGConfig()

    for method in ("POS", "CHROM"):
        bvp = extract_standard_bvp(rgb, sample_rate_hz, method, config)
        estimate = estimate_spectral_hr(
            bvp[-int(10 * sample_rate_hz) :],
            sample_rate_hz,
            config,
        )

        assert np.isfinite(bvp).all()
        assert abs(estimate["rppg_hr_bpm"] - expected_bpm) < 0.5
        assert 0.0 <= estimate["spectral_entropy"] <= 1.0


def test_process_subject_windows_preserves_phase1_window_keys() -> None:
    sample_rate_hz = 30.0
    expected_bpm = 72.0
    time_sec = np.arange(0.0, 20.0, 1.0 / sample_rate_hz)
    trace = _trace_frame(time_sec, expected_bpm / 60.0)
    reference = pd.DataFrame(
        [
            _reference_row(1, 0.0, expected_bpm),
            _reference_row(2, 5.0, expected_bpm),
        ]
    )
    config = StandardRPPGConfig(
        rois=("cheeks_mean",),
        methods=("CHROM", "POS"),
    )

    results = process_subject_windows(
        "subject1",
        reference,
        trace,
        config,
    )

    assert len(results) == 4
    assert set(results["window_id"]) == {1, 2}
    assert set(results["method"]) == {"CHROM", "POS"}
    assert results["window_status"].eq("ok").all()
    assert results["rppg_hr_bpm"].between(71.0, 73.0).all()
    assert results["pos_chrom_abs_diff_bpm"].notna().all()
    assert results["absolute_error_bpm"].lt(1.0).all()

    metrics = build_method_roi_metrics(results)
    assert len(metrics) == 2
    assert metrics["n_evaluated_windows"].eq(2).all()


def test_window_quality_rejects_a_long_interpolation_gap() -> None:
    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    trace = _trace_frame(time_sec, 1.2)
    invalid = (trace["time_sec"] >= 3.0) & (trace["time_sec"] < 4.0)
    trace.loc[invalid, "cheeks_mean_roi_valid"] = False

    quality = build_window_quality(
        trace,
        "cheeks_mean",
        start_time_sec=0.0,
        duration_sec=10.0,
        sample_rate_hz=sample_rate_hz,
    )

    assert quality["valid_frame_fraction"] > 0.8
    assert quality["max_interpolation_gap_sec"] > 0.5
    assert quality["window_status"] == "interpolation_gap_too_large"


def _synthetic_rgb(time_sec: np.ndarray, pulse_hz: float) -> np.ndarray:
    pulse = np.sin(2.0 * np.pi * pulse_hz * time_sec)
    slow_motion = 0.004 * np.sin(2.0 * np.pi * 0.15 * time_sec)
    red = 180.0 * (1.0 + 0.003 * pulse + slow_motion)
    green = 140.0 * (1.0 + 0.010 * pulse + slow_motion)
    blue = 120.0 * (1.0 + 0.005 * pulse + slow_motion)
    return np.column_stack([red, green, blue])


def _trace_frame(time_sec: np.ndarray, pulse_hz: float) -> pd.DataFrame:
    rgb = _synthetic_rgb(time_sec, pulse_hz)
    return pd.DataFrame(
        {
            "frame_idx": np.arange(len(time_sec)),
            "time_sec": time_sec,
            "face_area_ratio": 0.08,
            "bbox_center_jump": 0.001,
            "bbox_area_change": 0.002,
            "cheeks_mean_r_mean": rgb[:, 0],
            "cheeks_mean_g_mean": rgb[:, 1],
            "cheeks_mean_b_mean": rgb[:, 2],
            "cheeks_mean_n_pixels": 3000,
            "cheeks_mean_fill_ratio": 0.9,
            "cheeks_mean_brightness": np.mean(rgb, axis=1),
            "cheeks_mean_overexposure_ratio": 0.0,
            "cheeks_mean_roi_valid": True,
        }
    )


def _reference_row(
    window_id: int,
    start_time_sec: float,
    expected_bpm: float,
) -> dict[str, object]:
    end_time_sec = start_time_sec + 10.0 - (1.0 / 30.0)
    return {
        "window_id": window_id,
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "window_center_time_sec": (start_time_sec + end_time_sec) / 2.0,
        "duration_sec": 10.0,
        "time_hr_bpm": expected_bpm,
        "frequency_hr_bpm": expected_bpm,
        "reference_category": "concordant",
        "eligible_primary": True,
    }
