from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
        "extract_rgb_traces",
        lambda video_path, detector, fractions: {0.0: trace},
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


def test_run_degradation_experiment_builds_five_condition_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from rppg_pipeline import experiment
    from rppg_pipeline.dataset import SubjectFiles

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
    original_trace = _trace_frame(
        time_sec,
        _synthetic_rgb(time_sec, expected_bpm / 60.0),
    )
    baseline_output = _write_baseline_output(
        tmp_path,
        subject_dir / "ground_truth.txt",
        original_trace,
    )

    references = pd.read_csv(baseline_output / "reference_windows.csv")
    calls = []

    def fake_extract(video_path, detector, roi_shift_fractions):
        calls.append(tuple(roi_shift_fractions))
        return {
            fraction: _trace_frame(
                time_sec,
                _synthetic_rgb(time_sec, expected_bpm / 60.0),
                fraction,
            )
            for fraction in roi_shift_fractions
        }

    monkeypatch.setattr(experiment, "FaceLandmarker", _FakeLandmarker)
    monkeypatch.setattr(experiment, "extract_rgb_traces", fake_extract)

    tables = experiment.run_degradation_experiment(
        [
            SubjectFiles(
                "subject1", subject_dir / "vid.avi", subject_dir / "ground_truth.txt"
            )
        ],
        model_path,
        baseline_output,
        references,
    )

    assert calls == [(0.03, 0.05)]
    assert set(tables) == {"candidates", "audit"}
    candidates = tables["candidates"]
    audit = tables["audit"]
    assert len(candidates) == 30
    assert len(audit) == 15
    assert "effective_fps" not in candidates
    assert candidates["condition"].drop_duplicates().tolist() == [
        "original",
        "fps15",
        "fps10",
        "roi_shift_3",
        "roi_shift_5",
    ]
    assert not candidates.duplicated(
        ["subject", "condition", "window_id", "roi", "method"]
    ).any()
    fps15 = audit[audit["condition"].eq("fps15")]
    assert fps15["source_frame_count"].eq(300).all()
    assert fps15["retained_frame_count"].eq(150).all()
    assert np.allclose(fps15["effective_fps"], 15.0, atol=0.4)
    shifted = audit[audit["condition"].eq("roi_shift_5")]
    assert np.allclose(shifted["roi_shift_fraction"], 0.05)
    assert np.allclose(shifted["max_roi_shift_fraction"], 0.05, atol=1e-12)
    assert np.allclose(shifted["mean_roi_retention_ratio"], 0.95)


def test_condition_audit_ignores_missing_landmark_measurements() -> None:
    from rppg_pipeline.experiment import build_condition_audit

    time_sec = np.arange(0.0, 10.0, 1.0 / 30.0)
    trace = _trace_frame(time_sec, _synthetic_rgb(time_sec, 1.2), 0.05)
    trace.loc[0, "face_width_px"] = np.nan
    trace.loc[0, "forehead_retention_ratio"] = np.nan
    references = pd.DataFrame(
        [
            {
                "subject": "subject1",
                "window_id": 1,
                "start_time_sec": 0.0,
                "end_time_sec": 10.0,
            }
        ]
    )
    condition_traces = {
        condition: trace.copy()
        for condition in ("original", "fps15", "fps10", "roi_shift_3", "roi_shift_5")
    }

    audit = build_condition_audit(references, condition_traces)

    row = audit[
        audit["condition"].eq("roi_shift_5") & audit["roi"].eq("forehead")
    ].iloc[0]
    assert row["auditable_frame_count"] == 299
    assert row["auditable_frame_fraction"] == pytest.approx(299 / 300)
    assert row["max_roi_shift_fraction"] == pytest.approx(0.05)
    assert row["mean_roi_retention_ratio"] == pytest.approx(0.95)


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


def _write_baseline_output(
    tmp_path: Path,
    ground_truth_path: Path,
    trace: pd.DataFrame,
) -> Path:
    from rppg_pipeline.dataset import build_reference_windows, load_ground_truth

    output = tmp_path / "baseline"
    trace_path = output / "traces"
    trace_path.mkdir(parents=True)
    trace.to_csv(trace_path / "subject1.csv", index=False)
    references = build_reference_windows(
        "subject1",
        load_ground_truth(ground_truth_path),
    )
    references.to_csv(output / "reference_windows.csv", index=False)
    return output


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


def _trace_frame(
    time_sec: np.ndarray,
    rgb: np.ndarray,
    roi_shift_fraction: float = 0.0,
) -> pd.DataFrame:
    phase = 2.0 * np.pi * time_sec / 10.0
    face_width_px = 200.0
    frame: dict[str, object] = {
        "frame_idx": np.arange(len(time_sec)),
        "time_sec": time_sec,
        "landmark_valid": True,
        "face_width_px": face_width_px,
        "face_area_ratio": 0.1,
        "face_center_shift": 0.001,
        "face_area_change": 0.002,
        "roi_shift_fraction": roi_shift_fraction,
        "roi_shift_x_px": face_width_px * roi_shift_fraction * np.sin(phase),
        "roi_shift_y_px": (
            0.5 * face_width_px * roi_shift_fraction * np.sin(2.0 * phase)
        ),
    }
    for roi in ("full_face_inner", "forehead", "cheeks_mean"):
        frame[f"{roi}_r_mean"] = rgb[:, 0]
        frame[f"{roi}_g_mean"] = rgb[:, 1]
        frame[f"{roi}_b_mean"] = rgb[:, 2]
        frame[f"{roi}_n_pixels"] = 1000
        frame[f"{roi}_brightness"] = np.mean(rgb, axis=1)
        frame[f"{roi}_overexposure_ratio"] = 0.0
        frame[f"{roi}_valid"] = True
        frame[f"{roi}_retention_ratio"] = 1.0 - roi_shift_fraction
    return pd.DataFrame(frame)
