from __future__ import annotations

import json
from pathlib import Path

from rppg_pipeline.run_batch import (
    build_subject_command,
    is_output_complete,
    parse_args,
    required_output_files,
    run_batch,
)
from rppg_pipeline.ubfc import discover_subjects


def test_discover_subjects_uses_original_ids_and_ignores_metadata(tmp_path) -> None:
    for name in [
        "subject10",
        "subject1_metadata",
        "subject3",
        "subject1",
        "subject10_roi",
        "subject01",
        "notes",
    ]:
        (tmp_path / name).mkdir()

    subjects = discover_subjects(tmp_path)

    assert [subject.name for subject in subjects] == [
        "subject1",
        "subject3",
        "subject10",
    ]
    assert [subject.subject_id for subject in subjects] == [1, 3, 10]


def test_subject1_metadata_does_not_count_as_roi_output(tmp_path) -> None:
    args = parse_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--out-root",
            str(tmp_path / "results"),
        ]
    )
    required = required_output_files(args)
    metadata_dir = tmp_path / "results" / "subject1_metadata"
    metadata_dir.mkdir(parents=True)
    for name in required:
        (metadata_dir / name).touch()

    assert not is_output_complete(tmp_path / "results" / "subject1_roi", required)

    roi_dir = tmp_path / "results" / "subject1_roi"
    roi_dir.mkdir()
    for name in required:
        (roi_dir / name).touch()
    assert is_output_complete(roi_dir, required)


def test_build_subject_command_uses_adjacent_ground_truth(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    subject_dir = dataset_root / "subject8"
    subject_dir.mkdir(parents=True)
    (subject_dir / "vid.avi").touch()
    (subject_dir / "ground_truth.txt").touch()
    subject = discover_subjects(dataset_root)[0]
    args = parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--out-root",
            str(tmp_path / "results"),
            "--run-rppg",
            "--evaluate",
        ]
    )

    command = build_subject_command(
        subject,
        Path(args.out_root) / "subject8_roi",
        args,
    )

    assert str(subject.video_path) in command
    assert str(subject.ground_truth_path) in command
    assert "--run-rppg" in command
    assert "--evaluate" in command


def test_run_batch_does_not_skip_for_subject1_metadata(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    subject_dir = dataset_root / "subject1"
    subject_dir.mkdir(parents=True)
    (subject_dir / "vid.avi").touch()
    out_root = tmp_path / "results"
    metadata_dir = out_root / "subject1_metadata"
    metadata_dir.mkdir(parents=True)
    for name in ["video_metadata.json", "video_metadata.csv"]:
        (metadata_dir / name).touch()

    dry_run_args = parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--out-root",
            str(out_root),
            "--dry-run",
        ]
    )
    assert run_batch(dry_run_args) == 0
    summary = json.loads((out_root / "batch_summary.json").read_text())
    assert summary["subjects"][0]["status"] == "dry_run"

    roi_dir = out_root / "subject1_roi"
    roi_dir.mkdir()
    for name in ["video_metadata.json", "video_metadata.csv"]:
        (roi_dir / name).touch()
    skip_args = parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--out-root",
            str(out_root),
        ]
    )
    assert run_batch(skip_args) == 0
    summary = json.loads((out_root / "batch_summary.json").read_text())
    assert summary["subjects"][0]["status"] == "skipped_complete"
