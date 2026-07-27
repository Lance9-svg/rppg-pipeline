from __future__ import annotations

import numpy as np
import pandas as pd

from rppg_pipeline.evaluation import (
    EvaluationConfig,
    build_evaluation_metrics,
    build_evaluation_results,
    build_ppg_hr_results,
    process_ubfc_evaluation,
)
from rppg_pipeline.rppg import bandpass_filter, estimate_heart_rate
from rppg_pipeline.ubfc import UBFCGroundTruth, read_ubfc_ground_truth


def test_read_ubfc_ground_truth_parses_three_rows(tmp_path) -> None:
    data = np.array(
        [
            [0.1, 0.2, 0.3],
            [70.0, 71.0, 72.0],
            [0.0, 0.1, 0.2],
        ]
    )
    path = tmp_path / "ground_truth.txt"
    np.savetxt(path, data)

    ground_truth = read_ubfc_ground_truth(path)

    assert ground_truth.ppg_trace.tolist() == [0.1, 0.2, 0.3]
    assert ground_truth.sensor_hr.tolist() == [70.0, 71.0, 72.0]
    assert ground_truth.timestamp_sec.tolist() == [0.0, 0.1, 0.2]


def test_build_ppg_hr_results_matches_rppg_windows() -> None:
    fps = 30.0
    time_sec = np.arange(0, 20, 1 / fps)
    ground_truth = _ground_truth(time_sec, pulse_hz=1.2)
    hr_df = _hr_results_for_windows(fps, ground_truth.ppg_trace, offset_bpm=2.0)

    ppg_hr = build_ppg_hr_results(
        hr_df,
        ground_truth,
        fps=fps,
        config=EvaluationConfig(),
    )

    assert len(ppg_hr) == 2
    assert ppg_hr["estimate_type"].tolist() == ["window", "window"]
    assert ppg_hr["start_frame"].tolist() == [0, 300]
    assert ppg_hr["ppg_hr_bpm"].between(70.0, 74.0).all()


def test_evaluation_metrics_report_expected_error() -> None:
    fps = 30.0
    time_sec = np.arange(0, 20, 1 / fps)
    ground_truth = _ground_truth(time_sec, pulse_hz=1.2)
    hr_df = _hr_results_for_windows(fps, ground_truth.ppg_trace, offset_bpm=2.0)
    ppg_hr = build_ppg_hr_results(
        hr_df,
        ground_truth,
        fps=fps,
        config=EvaluationConfig(),
    )
    evaluation = build_evaluation_results(hr_df, ppg_hr)
    metrics = build_evaluation_metrics(evaluation)

    assert len(evaluation) == 2
    assert np.allclose(evaluation["error_bpm"], 2.0)
    assert len(metrics) == 1
    assert metrics.iloc[0]["mae_bpm"] == 2.0
    assert metrics.iloc[0]["rmse_bpm"] == 2.0
    assert metrics.iloc[0]["bias_bpm"] == 2.0


def test_process_ubfc_evaluation_writes_expected_files(tmp_path) -> None:
    fps = 30.0
    time_sec = np.arange(0, 20, 1 / fps)
    ground_truth = _ground_truth(time_sec, pulse_hz=1.2)
    gt_path = tmp_path / "ground_truth.txt"
    np.savetxt(
        gt_path,
        np.vstack(
            [
                ground_truth.ppg_trace,
                ground_truth.sensor_hr,
                ground_truth.timestamp_sec,
            ]
        ),
    )
    hr_df = _hr_results_for_windows(fps, ground_truth.ppg_trace, offset_bpm=1.5)
    hr_df.to_csv(tmp_path / "hr_results.csv", index=False)

    process_ubfc_evaluation(tmp_path, gt_path, fps=fps)

    assert (tmp_path / "ppg_hr_results.csv").exists()
    assert (tmp_path / "evaluation_results.csv").exists()
    assert (tmp_path / "evaluation_metrics.csv").exists()
    assert (tmp_path / "evaluation_plot.png").exists()
    metrics = pd.read_csv(tmp_path / "evaluation_metrics.csv")
    assert metrics.iloc[0]["mae_bpm"] == 1.5


def _ground_truth(time_sec: np.ndarray, pulse_hz: float):
    ppg = np.sin(2 * np.pi * pulse_hz * time_sec)
    sensor_hr = np.full_like(time_sec, pulse_hz * 60.0)
    return UBFCGroundTruth(
        ppg_trace=ppg,
        sensor_hr=sensor_hr,
        timestamp_sec=time_sec,
    )


def _hr_results_for_windows(
    fps: float,
    ppg: np.ndarray,
    offset_bpm: float,
) -> pd.DataFrame:
    rows = []
    for start_frame, end_frame in [(0, 299), (300, 599)]:
        ppg_filtered = bandpass_filter(
            ppg[start_frame : end_frame + 1],
            fps=fps,
        )
        estimate = estimate_heart_rate(ppg_filtered, fps=fps)
        rows.append(
            {
                "segment_id": 1,
                "roi": "cheeks_mean",
                "method": "CHROM",
                "estimate_type": "window",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": start_frame / fps,
                "end_time_sec": end_frame / fps,
                "window_center_time_sec": (start_frame + end_frame) / (2 * fps),
                "duration_sec": (end_frame - start_frame + 1) / fps,
                "n_samples": end_frame - start_frame + 1,
                "hr_bpm": estimate["hr_bpm"] + offset_bpm,
                "peak_frequency_hz": estimate["peak_frequency_hz"],
                "peak_power": estimate["peak_power"],
            }
        )
    return pd.DataFrame(rows)
