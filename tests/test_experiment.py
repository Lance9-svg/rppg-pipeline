import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rppg_pipeline.experiment as experiment
from rppg_pipeline.candidates import build_candidate_table
from rppg_pipeline.experiment import (
    ExperimentConfig,
    build_inventory,
    parse_args,
    run_experiment,
)
from rppg_pipeline.ppg_reference import build_subject_reference
from rppg_pipeline.standard_rppg import process_subject_windows
from rppg_pipeline.ubfc import discover_subjects


def test_parse_args_builds_fixed_experiment_config(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ubfc"
    face_model = tmp_path / "face_landmarker.task"
    output = tmp_path / "experiment"

    config = parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--face-model",
            str(face_model),
            "--output",
            str(output),
            "--subjects",
            "1",
            "3",
        ]
    )

    assert config == ExperimentConfig(
        dataset_root=dataset_root,
        face_model=face_model,
        output=output,
        subjects=(1, 3),
    )


def test_package_module_exposes_the_single_experiment_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rppg_pipeline", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: python -m rppg_pipeline" in result.stdout
    assert "--dataset-root" in result.stdout
    assert "--face-model" in result.stdout
    assert "--output" in result.stdout


def test_run_experiment_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    config = ExperimentConfig(
        dataset_root=tmp_path / "ubfc",
        face_model=tmp_path / "face_landmarker.task",
        output=output,
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        run_experiment(config)


def test_build_inventory_uses_stable_relative_names(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ubfc"
    for subject_id in (3, 1):
        subject_dir = dataset_root / f"subject{subject_id}"
        subject_dir.mkdir(parents=True)
        (subject_dir / "vid.avi").write_bytes(f"video-{subject_id}".encode())
        (subject_dir / "ground_truth.txt").write_text(
            f"ground-truth-{subject_id}", encoding="utf-8"
        )

    inventory = build_inventory(discover_subjects(dataset_root))

    assert inventory["subject"].tolist() == ["subject1", "subject3"]
    assert inventory["video"].tolist() == [
        "subject1/vid.avi",
        "subject3/vid.avi",
    ]
    assert inventory["ground_truth"].tolist() == [
        "subject1/ground_truth.txt",
        "subject3/ground_truth.txt",
    ]
    assert str(dataset_root) not in inventory.to_json()


def test_build_inventory_rejects_missing_subject_input(tmp_path: Path) -> None:
    subject_dir = tmp_path / "ubfc" / "subject1"
    subject_dir.mkdir(parents=True)
    (subject_dir / "vid.avi").write_bytes(b"video")

    with pytest.raises(FileNotFoundError, match="ground truth"):
        build_inventory(discover_subjects(subject_dir.parent))


def test_run_experiment_executes_the_fixed_stages_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, face_model = _single_subject_inputs(tmp_path)
    output = tmp_path / "experiment"
    stages: list[str] = []

    def fake_read_video_metadata(_video_path: Path) -> object:
        stages.append("video_metadata")
        return object()

    def fake_process_video_rgb_trace(
        _video_path: Path,
        out_dir: Path,
        _model_path: Path,
        _metadata: object,
        **_kwargs: object,
    ) -> None:
        stages.append("rgb_trace")
        trace_dir = Path(out_dir)
        trace_dir.mkdir(parents=True)
        _synthetic_trace().to_csv(trace_dir / "rgb_trace.csv", index=False)

    def traced_build_reference(*args: object, **kwargs: object) -> object:
        stages.append("reference")
        return build_subject_reference(*args, **kwargs)

    def traced_process_windows(*args: object, **kwargs: object) -> pd.DataFrame:
        stages.append("rppg")
        return process_subject_windows(*args, **kwargs)

    def traced_build_candidates(*args: object, **kwargs: object) -> pd.DataFrame:
        stages.append("candidates")
        return build_candidate_table(*args, **kwargs)

    monkeypatch.setattr(experiment, "read_video_metadata", fake_read_video_metadata)
    monkeypatch.setattr(
        experiment,
        "process_video_rgb_trace",
        fake_process_video_rgb_trace,
    )
    monkeypatch.setattr(experiment, "build_subject_reference", traced_build_reference)
    monkeypatch.setattr(experiment, "process_subject_windows", traced_process_windows)
    monkeypatch.setattr(experiment, "build_candidate_table", traced_build_candidates)

    status = run_experiment(
        ExperimentConfig(
            dataset_root=dataset_root,
            face_model=face_model,
            output=output,
            subjects=(1,),
        )
    )

    candidates = pd.read_csv(output / "candidate_windows.csv")
    features = pd.read_csv(output / "reliability_features.csv")
    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert status == 0
    assert stages == [
        "video_metadata",
        "rgb_trace",
        "reference",
        "rppg",
        "candidates",
    ]
    assert len(candidates) == 10
    assert not candidates.duplicated(
        ["subject", "window_id", "roi", "method"]
    ).any()
    assert run_record["status"] == "success"
    assert {
        "created_at_utc",
        "started_at_utc",
        "finished_at_utc",
    }.isdisjoint(run_record)
    assert run_record["selected_subjects"] == ["subject1"]
    assert run_record["summary"] == {
        "selected_subject_count": 1,
        "completed_subject_count": 1,
        "reference_window_count": 1,
        "candidate_row_count": 10,
        "feature_row_count": 10,
    }
    assert {item["name"] for item in run_record["inputs"]} == {
        "face_landmarker.task",
        "subject1/ground_truth.txt",
        "subject1/vid.avi",
    }
    assert (output / "inventory.csv").is_file()
    assert (output / "reference_windows.csv").is_file()
    assert len(features) == 10
    assert (output / "feature_dictionary.csv").is_file()
    assert (output / "feature_audit.csv").is_file()


def test_run_experiment_records_failure_without_retrying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, face_model = _single_subject_inputs(tmp_path)
    output = tmp_path / "failed-experiment"
    process_calls = 0
    later_stages: list[str] = []

    monkeypatch.setattr(experiment, "read_video_metadata", lambda _path: object())

    def fail_once(*_args: object, **_kwargs: object) -> None:
        nonlocal process_calls
        process_calls += 1
        raise RuntimeError("synthetic face-processing failure")

    def unexpected_reference(*_args: object, **_kwargs: object) -> object:
        later_stages.append("reference")
        raise AssertionError("reference stage must not run")

    monkeypatch.setattr(experiment, "process_video_rgb_trace", fail_once)
    monkeypatch.setattr(experiment, "build_subject_reference", unexpected_reference)

    status = run_experiment(
        ExperimentConfig(
            dataset_root=dataset_root,
            face_model=face_model,
            output=output,
            subjects=(1,),
        )
    )

    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert status == 1
    assert process_calls == 1
    assert later_stages == []
    assert run_record["status"] == "failed"
    assert run_record["failed_stage"] == "rgb_trace:subject1"
    assert run_record["error"] == {
        "type": "RuntimeError",
        "message": "synthetic face-processing failure",
    }
    assert run_record["command"][:3] == ["python", "-m", "rppg_pipeline"]
    assert not (output / "candidate_windows.csv").exists()


def test_run_experiment_records_provenance_failure_before_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, face_model = _single_subject_inputs(tmp_path)
    output = tmp_path / "provenance-failure"
    processing_started = False

    def fail_provenance(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic provenance failure")

    def unexpected_processing(*_args: object, **_kwargs: object) -> None:
        nonlocal processing_started
        processing_started = True

    monkeypatch.setattr(experiment, "build_run_manifest", fail_provenance)
    monkeypatch.setattr(
        experiment,
        "process_video_rgb_trace",
        unexpected_processing,
    )

    status = run_experiment(
        ExperimentConfig(
            dataset_root=dataset_root,
            face_model=face_model,
            output=output,
            subjects=(1,),
        )
    )

    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert status == 1
    assert processing_started is False
    assert run_record["status"] == "failed"
    assert run_record["failed_stage"] == "provenance"
    assert run_record["error"] == {
        "type": "RuntimeError",
        "message": "synthetic provenance failure",
    }


def test_run_experiment_captures_clean_git_before_writing_inside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    dataset_root, face_model = _single_subject_inputs(repository)
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "add inputs")
    output = repository / "unignored-output"

    monkeypatch.setattr(experiment, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(experiment, "read_video_metadata", lambda _path: object())

    def fail_processing(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("stop after inventory")

    monkeypatch.setattr(experiment, "process_video_rgb_trace", fail_processing)

    status = run_experiment(
        ExperimentConfig(
            dataset_root=dataset_root,
            face_model=face_model,
            output=output,
            subjects=(1,),
        )
    )

    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert status == 1
    assert run_record["git_dirty"] is False
    assert (output / "inventory.csv").is_file()


def test_run_experiment_records_inventory_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, face_model = _single_subject_inputs(tmp_path)
    output = tmp_path / "inventory-failure"

    def fail_csv_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic inventory write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv_write)

    status = run_experiment(
        ExperimentConfig(
            dataset_root=dataset_root,
            face_model=face_model,
            output=output,
            subjects=(1,),
        )
    )

    run_record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert status == 1
    assert run_record["failed_stage"] == "inventory"
    assert run_record["error"] == {
        "type": "OSError",
        "message": "synthetic inventory write failure",
    }


def _single_subject_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "ubfc"
    subject_dir = dataset_root / "subject1"
    subject_dir.mkdir(parents=True)
    (subject_dir / "vid.avi").write_bytes(b"synthetic-video-boundary")

    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    pulse = np.sin(2.0 * np.pi * 1.2 * time_sec)
    np.savetxt(
        subject_dir / "ground_truth.txt",
        np.vstack([pulse, np.full(len(time_sec), 72.0), time_sec]),
    )
    face_model = tmp_path / "face_landmarker.task"
    face_model.write_bytes(b"synthetic-model-boundary")
    return dataset_root, face_model


def _synthetic_trace() -> pd.DataFrame:
    sample_rate_hz = 30.0
    time_sec = np.arange(0.0, 10.0, 1.0 / sample_rate_hz)
    pulse = np.sin(2.0 * np.pi * 1.2 * time_sec)
    slow_motion = 0.004 * np.sin(2.0 * np.pi * 0.15 * time_sec)
    rgb = np.column_stack(
        [
            180.0 * (1.0 + 0.003 * pulse + slow_motion),
            140.0 * (1.0 + 0.010 * pulse + slow_motion),
            120.0 * (1.0 + 0.005 * pulse + slow_motion),
        ]
    )
    columns: dict[str, object] = {
        "frame_idx": np.arange(len(time_sec)),
        "time_sec": time_sec,
        "face_area_ratio": 0.08,
        "bbox_center_jump": 0.001,
        "bbox_area_change": 0.002,
    }
    for roi in (
        "full_face_inner",
        "forehead",
        "left_cheek",
        "right_cheek",
        "cheeks_mean",
    ):
        columns.update(
            {
                f"{roi}_r_mean": rgb[:, 0],
                f"{roi}_g_mean": rgb[:, 1],
                f"{roi}_b_mean": rgb[:, 2],
                f"{roi}_n_pixels": 3000,
                f"{roi}_fill_ratio": 0.9,
                f"{roi}_brightness": np.mean(rgb, axis=1),
                f"{roi}_overexposure_ratio": 0.0,
                f"{roi}_roi_valid": True,
            }
        )
    return pd.DataFrame(columns)


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
