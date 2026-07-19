from __future__ import annotations

import numpy as np
import pandas as pd

from rppg_pipeline.rppg import (
    RPPGConfig,
    bandpass_filter,
    chrom_signal,
    estimate_heart_rate,
    mean_normalize_rgb,
    pos_signal,
    process_rppg_from_outputs,
    select_rgb_segments,
)


def test_mean_normalize_rgb_centers_each_channel() -> None:
    rgb = np.array([[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]])
    normalized = mean_normalize_rgb(rgb)

    assert np.allclose(np.mean(normalized, axis=0), 0.0)


def test_chrom_and_pos_return_finite_nonconstant_signals() -> None:
    fps = 30.0
    time_sec = np.arange(0, 20, 1 / fps)
    rgb = _synthetic_rgb(time_sec, pulse_hz=1.2)

    chrom = chrom_signal(rgb)
    pos = pos_signal(rgb)

    assert np.isfinite(chrom).all()
    assert np.isfinite(pos).all()
    assert np.std(chrom) > 0
    assert np.std(pos) > 0


def test_bandpass_and_welch_hr_recover_synthetic_frequency() -> None:
    fps = 30.0
    time_sec = np.arange(0, 20, 1 / fps)
    pulse = np.sin(2 * np.pi * 1.2 * time_sec)
    filtered = bandpass_filter(pulse, fps=fps, low_hz=0.7, high_hz=4.0)
    estimate = estimate_heart_rate(filtered, fps=fps, estimator="welch")

    assert 70.0 <= estimate["hr_bpm"] <= 74.0


def test_select_rgb_segments_uses_only_long_cheeks_segments() -> None:
    trace = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3],
            "time_sec": [0.0, 0.1, 0.2, 0.3],
            "cheeks_mean_r_mean": [100.0, 101.0, 102.0, 103.0],
            "cheeks_mean_g_mean": [90.0, 91.0, 92.0, 93.0],
            "cheeks_mean_b_mean": [80.0, 81.0, 82.0, 83.0],
            "cheeks_mean_roi_valid": [True, True, False, True],
        }
    )
    segments = pd.DataFrame(
        [
            {
                "segment_id": 1,
                "roi": "cheeks_mean",
                "start_frame": 0,
                "end_frame": 3,
                "duration_sec": 0.4,
                "is_long_enough": True,
            },
            {
                "segment_id": 2,
                "roi": "forehead",
                "start_frame": 0,
                "end_frame": 3,
                "duration_sec": 0.4,
                "is_long_enough": True,
            },
        ]
    )

    selected = select_rgb_segments(trace, segments, roi="cheeks_mean")

    assert len(selected) == 1
    assert selected[0].frame_idx.tolist() == [0, 1, 3]
    assert selected[0].rgb.shape == (3, 3)


def test_process_rppg_from_outputs_writes_expected_files(tmp_path) -> None:
    fps = 30.0
    time_sec = np.arange(0, 20, 1 / fps)
    rgb = _synthetic_rgb(time_sec, pulse_hz=1.2)
    trace = pd.DataFrame(
        {
            "frame_idx": np.arange(len(time_sec)),
            "time_sec": time_sec,
            "timestamp_ms": (time_sec * 1000).astype(int),
            "cheeks_mean_r_mean": rgb[:, 0],
            "cheeks_mean_g_mean": rgb[:, 1],
            "cheeks_mean_b_mean": rgb[:, 2],
            "cheeks_mean_roi_valid": True,
        }
    )
    segments = pd.DataFrame(
        [
            {
                "segment_id": 1,
                "roi": "cheeks_mean",
                "start_frame": 0,
                "end_frame": len(time_sec) - 1,
                "duration_sec": len(time_sec) / fps,
                "is_long_enough": True,
            }
        ]
    )
    trace.to_csv(tmp_path / "rgb_trace.csv", index=False)
    segments.to_csv(tmp_path / "valid_segments.csv", index=False)

    process_rppg_from_outputs(
        tmp_path,
        fps=fps,
        config=RPPGConfig(hr_window_sec=10.0, hr_step_sec=5.0),
    )

    assert (tmp_path / "rppg_signal.csv").exists()
    assert (tmp_path / "hr_results.csv").exists()
    assert (tmp_path / "hr_curve.png").exists()
    assert (tmp_path / "runtime_results.csv").exists()
    hr = pd.read_csv(tmp_path / "hr_results.csv")
    segment_hr = hr[hr["estimate_type"] == "segment"]["hr_bpm"]
    assert segment_hr.between(60.0, 84.0).all()


def _synthetic_rgb(time_sec: np.ndarray, pulse_hz: float) -> np.ndarray:
    # Create RGB traces with a known pulse component.
    pulse = np.sin(2 * np.pi * pulse_hz * time_sec)
    red = 180.0 * (1.0 + 0.01 * pulse)
    green = 140.0 * (1.0 + 0.03 * pulse)
    blue = 130.0 * (1.0 + 0.00 * pulse)
    return np.column_stack([red, green, blue])
