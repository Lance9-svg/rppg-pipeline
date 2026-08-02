"""Run the fixed UBFC reliability experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from rppg_pipeline.candidates import build_candidate_table
from rppg_pipeline.ppg_reference import PPGReferenceConfig, build_subject_reference
from rppg_pipeline.provenance import build_run_manifest
from rppg_pipeline.rgb_trace import process_video_rgb_trace
from rppg_pipeline.standard_rppg import (
    StandardRPPGConfig,
    process_subject_windows,
)
from rppg_pipeline.ubfc import (
    SubjectDataset,
    discover_subjects,
    read_ubfc_ground_truth,
    select_subjects,
)
from rppg_pipeline.video import read_video_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    """Paths and optional subject subset for one experiment run."""

    dataset_root: Path
    face_model: Path
    output: Path
    subjects: tuple[int, ...] | None = None


def build_inventory(subjects: Sequence[SubjectDataset]) -> pd.DataFrame:
    """Build a stable, portable inventory for canonical UBFC subject inputs."""
    rows: list[dict[str, int | str]] = []
    for subject in sorted(subjects, key=lambda item: item.subject_id):
        if not subject.video_path.is_file():
            raise FileNotFoundError(f"video not found: {subject.video_path}")
        if not subject.ground_truth_path.is_file():
            raise FileNotFoundError(
                f"ground truth not found: {subject.ground_truth_path}"
            )
        rows.append(
            {
                "subject": subject.name,
                "subject_id": subject.subject_id,
                "video": f"{subject.name}/vid.avi",
                "ground_truth": f"{subject.name}/ground_truth.txt",
            }
        )

    if not rows:
        raise ValueError("no canonical UBFC subjects found")
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    """Parse the single public experiment command."""
    parser = argparse.ArgumentParser(
        prog="python -m rppg_pipeline",
        description="Run the standalone UBFC rPPG reliability experiment.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+")
    args = parser.parse_args(argv)
    subjects = tuple(args.subjects) if args.subjects is not None else None
    return ExperimentConfig(
        dataset_root=args.dataset_root,
        face_model=args.face_model,
        output=args.output,
        subjects=subjects,
    )


def run_experiment(config: ExperimentConfig) -> int:
    """Run one experiment without overwriting an existing output."""
    if config.output.exists():
        raise FileExistsError(f"output already exists: {config.output}")

    subjects = select_subjects(
        discover_subjects(config.dataset_root),
        config.subjects,
    )
    inventory = build_inventory(subjects)
    if not config.face_model.is_file():
        raise FileNotFoundError(f"face model not found: {config.face_model}")

    config.output.mkdir(parents=True)
    started_at = datetime.now(UTC)
    completed_stages: list[str] = []
    current_stage = "provenance"
    base_record: dict[str, object] = {
        "schema_version": 1,
        "stage": "dataset-to-candidate",
    }

    try:
        base_record = _build_base_record(config, subjects)
        completed_stages.append(current_stage)
        current_stage = "inventory"
        inventory.to_csv(config.output / "inventory.csv", index=False)
        completed_stages.append(current_stage)
        reference_frames: list[pd.DataFrame] = []
        result_frames: list[pd.DataFrame] = []
        reference_config = PPGReferenceConfig()
        rppg_config = StandardRPPGConfig()

        for subject in subjects:
            current_stage = f"rgb_trace:{subject.name}"
            metadata = read_video_metadata(subject.video_path)
            trace_dir = config.output / "traces" / subject.name
            process_video_rgb_trace(
                subject.video_path,
                trace_dir,
                config.face_model,
                metadata,
                debug_every=0,
            )
            completed_stages.append(current_stage)

            current_stage = f"reference:{subject.name}"
            ground_truth = read_ubfc_ground_truth(subject.ground_truth_path)
            reference = build_subject_reference(
                subject.name,
                ground_truth,
                reference_config,
            )
            reference_frames.append(reference.windows)
            completed_stages.append(current_stage)

            current_stage = f"rppg:{subject.name}"
            trace = pd.read_csv(trace_dir / "rgb_trace.csv")
            result_frames.append(
                process_subject_windows(
                    subject.name,
                    reference.windows,
                    trace,
                    rppg_config,
                )
            )
            completed_stages.append(current_stage)

        current_stage = "candidates"
        reference_windows = pd.concat(reference_frames, ignore_index=True)
        rppg_windows = pd.concat(result_frames, ignore_index=True)
        candidates = build_candidate_table(reference_windows, rppg_windows)
        reference_windows.to_csv(
            config.output / "reference_windows.csv",
            index=False,
        )
        candidates.to_csv(config.output / "candidate_windows.csv", index=False)
        completed_stages.append(current_stage)
    except Exception as error:
        _write_run_record(
            base_record,
            config,
            subjects,
            status="failed",
            started_at=started_at,
            completed_stages=completed_stages,
            failed_stage=current_stage,
            error=error,
        )
        return 1

    _write_run_record(
        base_record,
        config,
        subjects,
        status="success",
        started_at=started_at,
        completed_stages=completed_stages,
        result_summary={
            "reference_window_count": len(reference_windows),
            "candidate_row_count": len(candidates),
        },
    )
    return 0


def _build_base_record(
    config: ExperimentConfig,
    subjects: Sequence[SubjectDataset],
) -> dict[str, object]:
    inputs = [config.face_model]
    for subject in subjects:
        inputs.extend([subject.video_path, subject.ground_truth_path])
    return build_run_manifest(
        "dataset-to-candidate",
        inputs,
        {
            "reference": PPGReferenceConfig().to_dict(),
            "rppg": StandardRPPGConfig().to_dict(),
        },
        repository_root=REPOSITORY_ROOT,
        input_root=config.dataset_root,
    )


def _write_run_record(
    base_record: dict[str, object],
    config: ExperimentConfig,
    subjects: Sequence[SubjectDataset],
    *,
    status: str,
    started_at: datetime,
    completed_stages: Sequence[str],
    failed_stage: str | None = None,
    error: Exception | None = None,
    result_summary: dict[str, int] | None = None,
) -> None:
    record = dict(base_record)
    record.update(
        {
            "status": status,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "selected_subjects": [subject.name for subject in subjects],
            "completed_stages": list(completed_stages),
            "command": _reproduction_command(config),
            "summary": {
                "selected_subject_count": len(subjects),
                "completed_subject_count": sum(
                    stage.startswith("rppg:") for stage in completed_stages
                ),
                **(result_summary or {}),
            },
        }
    )
    if failed_stage is not None:
        record["failed_stage"] = failed_stage
    if error is not None:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    (config.output / "run.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _reproduction_command(config: ExperimentConfig) -> list[str]:
    command = [
        "python",
        "-m",
        "rppg_pipeline",
        "--dataset-root",
        str(config.dataset_root),
        "--face-model",
        str(config.face_model),
        "--output",
        str(config.output),
    ]
    if config.subjects:
        command.append("--subjects")
        command.extend(str(subject_id) for subject_id in config.subjects)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public experiment command."""
    return run_experiment(parse_args(argv))
