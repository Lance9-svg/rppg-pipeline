from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_run_original_baseline_writes_complete_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from rppg_pipeline import experiment

    dataset_root = tmp_path / "dataset"
    subject_dir = dataset_root / "subject1"
    subject_dir.mkdir(parents=True)
    (subject_dir / "vid.avi").touch()
    model_path = tmp_path / "face_landmarker.task"
    model_path.touch()
    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    expected_bpm = 72.0
    ppg = np.sin(2.0 * np.pi * (expected_bpm / 60.0) * time_sec)
    _write_ground_truth(
        subject_dir / "ground_truth.txt",
        ppg,
        np.full(len(time_sec), expected_bpm),
        time_sec,
    )
    trace = _trace_frame(time_sec, _synthetic_rgb(time_sec, expected_bpm / 60.0))
    monkeypatch.setattr(experiment, "FaceLandmarker", _FakeLandmarker)
    monkeypatch.setattr(
        experiment,
        "extract_rgb_trace",
        lambda video_path, detector: trace,
    )
    output = tmp_path / "baseline"

    result = experiment.run_original_baseline(dataset_root, model_path, output)

    assert result["subject_count"] == 1
    assert result["reference_window_count"] == 1
    assert result["candidate_count"] == 6
    references = pd.read_csv(output / "reference_windows.csv")
    candidates = pd.read_csv(output / "original_candidates.csv")
    metrics = pd.read_csv(output / "original_metrics.csv")
    assert len(references) == 1
    assert len(candidates) == 6
    assert len(metrics) == 6
    assert candidates["condition"].eq("original").all()
    assert (output / "traces" / "subject1.csv").is_file()


class _FakeLandmarker:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = model_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


def _write_ground_truth(
    path: Path,
    ppg: np.ndarray,
    sensor_hr: np.ndarray,
    time_sec: np.ndarray,
) -> None:
    path.write_text(
        "\n".join(
            " ".join(str(value) for value in values)
            for values in (ppg, sensor_hr, time_sec)
        ),
        encoding="utf-8",
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
