from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_discover_subjects_uses_numeric_subject_order(tmp_path: Path) -> None:
    from rppg_pipeline.dataset import discover_subjects

    for name in ("subject10", "subject2", "subject1"):
        subject_dir = tmp_path / name
        subject_dir.mkdir()
        (subject_dir / "vid.avi").touch()
        (subject_dir / "ground_truth.txt").touch()
    (tmp_path / "notes").mkdir()

    subjects = discover_subjects(tmp_path)

    assert [item.name for item in subjects] == [
        "subject1",
        "subject2",
        "subject10",
    ]
    assert subjects[0].video_path == tmp_path / "subject1" / "vid.avi"
    assert subjects[0].ground_truth_path == (
        tmp_path / "subject1" / "ground_truth.txt"
    )


def test_load_ground_truth_reads_the_three_ubfc_rows(tmp_path: Path) -> None:
    from rppg_pipeline.dataset import load_ground_truth

    path = tmp_path / "ground_truth.txt"
    path.write_text(
        "0.1 0.2 0.3\n72 73 74\n0.0 0.5 1.0\n",
        encoding="utf-8",
    )

    ground_truth = load_ground_truth(path)

    np.testing.assert_allclose(ground_truth.ppg, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(ground_truth.sensor_hr, [72.0, 73.0, 74.0])
    np.testing.assert_allclose(ground_truth.time_sec, [0.0, 0.5, 1.0])


def test_load_ground_truth_rejects_mismatched_row_lengths(tmp_path: Path) -> None:
    from rppg_pipeline.dataset import load_ground_truth

    path = tmp_path / "ground_truth.txt"
    path.write_text("0.1 0.2\n72\n0.0 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="equal lengths"):
        load_ground_truth(path)


def test_reference_uses_ten_second_non_overlapping_windows() -> None:
    from rppg_pipeline.dataset import GroundTruth, build_reference_windows

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 30.0, 1.0 / sample_rate_hz)
    expected_bpm = 72.0
    ground_truth = GroundTruth(
        ppg=np.sin(2.0 * np.pi * (expected_bpm / 60.0) * time_sec),
        sensor_hr=np.full(len(time_sec), expected_bpm),
        time_sec=time_sec,
    )

    windows = build_reference_windows("subject1", ground_truth)

    np.testing.assert_allclose(windows["start_time_sec"], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(windows["end_time_sec"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(windows["duration_sec"], 10.0)
    np.testing.assert_allclose(windows["reference_hr_bpm"], 72.0, atol=0.5)
    assert windows["reference_valid"].all()


def test_sensor_hr_wrap_is_corrected_for_reference_check() -> None:
    from rppg_pipeline.dataset import GroundTruth, build_reference_windows

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    expected_bpm = 130.0
    ground_truth = GroundTruth(
        ppg=np.sin(2.0 * np.pi * (expected_bpm / 60.0) * time_sec),
        sensor_hr=np.full(len(time_sec), 2.0),
        time_sec=time_sec,
    )

    windows = build_reference_windows("subject1", ground_truth)

    assert windows.loc[0, "sensor_hr_bpm"] == pytest.approx(expected_bpm)
    assert bool(windows.loc[0, "reference_valid"])


def test_reference_validity_uses_ppg_agreement_not_sensor_hr() -> None:
    from rppg_pipeline.dataset import GroundTruth, build_reference_windows

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    expected_bpm = 72.0
    ground_truth = GroundTruth(
        ppg=np.sin(2.0 * np.pi * (expected_bpm / 60.0) * time_sec),
        sensor_hr=np.full(len(time_sec), 120.0),
        time_sec=time_sec,
    )

    windows = build_reference_windows("subject1", ground_truth)

    assert windows.loc[0, "time_hr_bpm"] == pytest.approx(expected_bpm, abs=0.5)
    assert windows.loc[0, "frequency_hr_bpm"] == pytest.approx(
        expected_bpm,
        abs=0.5,
    )
    assert windows.loc[0, "sensor_hr_bpm"] == pytest.approx(120.0)
    assert bool(windows.loc[0, "reference_valid"])


def test_time_domain_hr_refines_off_grid_peaks() -> None:
    from rppg_pipeline.dataset import GroundTruth, build_reference_windows

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    expected_bpm = 73.7
    ground_truth = GroundTruth(
        ppg=np.sin(2.0 * np.pi * (expected_bpm / 60.0) * time_sec),
        sensor_hr=np.full(len(time_sec), expected_bpm),
        time_sec=time_sec,
    )

    windows = build_reference_windows("subject1", ground_truth)

    assert windows.loc[0, "time_hr_bpm"] == pytest.approx(expected_bpm, abs=0.5)


def test_time_domain_hr_requires_five_peaks() -> None:
    from rppg_pipeline.dataset import GroundTruth, build_reference_windows

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    ground_truth = GroundTruth(
        ppg=np.sin(2.0 * np.pi * 0.2 * time_sec),
        sensor_hr=np.full(len(time_sec), 72.0),
        time_sec=time_sec,
    )

    windows = build_reference_windows("subject1", ground_truth)

    assert np.isnan(windows.loc[0, "time_hr_bpm"])
    assert not bool(windows.loc[0, "reference_valid"])


def test_frequency_domain_hr_limits_spectral_leakage() -> None:
    from rppg_pipeline.dataset import GroundTruth, build_reference_windows

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    expected_bpm = 73.4
    ground_truth = GroundTruth(
        ppg=np.sin(2.0 * np.pi * (expected_bpm / 60.0) * time_sec),
        sensor_hr=np.full(len(time_sec), expected_bpm),
        time_sec=time_sec,
    )

    windows = build_reference_windows("subject1", ground_truth)

    assert windows.loc[0, "frequency_hr_bpm"] == pytest.approx(
        expected_bpm,
        abs=0.05,
    )
