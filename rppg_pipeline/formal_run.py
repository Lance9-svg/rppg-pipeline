# Formal degradation run

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import mediapipe
import numpy as np
import pandas as pd
import scipy

from rppg_pipeline.dataset import discover_subjects
from rppg_pipeline.degradation import FORMAL_CONDITIONS, ROIS
from rppg_pipeline.experiment import run_degradation_experiment
from rppg_pipeline.rppg import METHODS

FORMAL_SUBJECT_COUNT = 42
FORMAL_REFERENCE_WINDOW_COUNT = 248
OUTPUT_TABLES = {
    "degradation_candidates.csv": "candidates",
    "condition_audit.csv": "audit",
}
CANDIDATE_KEY = ["subject", "condition", "window_id", "roi", "method"]
AUDIT_KEY = ["subject", "condition", "window_id", "roi"]
REFERENCE_VALUES = ("start_time_sec", "end_time_sec", "reference_hr_bpm")
ALLOWED_STATUSES = {
    "ok",
    "low_video_coverage",
    "low_valid_fraction",
    "long_missing_gap",
}
SOURCE_FILES = (
    "rppg_pipeline/dataset.py",
    "rppg_pipeline/degradation.py",
    "rppg_pipeline/experiment.py",
    "rppg_pipeline/formal_run.py",
    "rppg_pipeline/rppg.py",
    "rppg_pipeline/video.py",
)


# Hash one file
def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Read package versions
def collect_environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "mediapipe": mediapipe.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


# Run one Git query
def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# Read Git provenance
def collect_git_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return {
        "branch": _git_output(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_output(root, "rev-parse", "HEAD"),
    }


# Hash experiment sources
def _source_hashes(root: Path) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


# Read row keys
def _row_keys(frame: pd.DataFrame, columns) -> set[tuple]:
    return set(frame[list(columns)].itertuples(index=False, name=None))


# Check result grids
def _validate_grids(
    references: pd.DataFrame,
    candidates: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    reference_keys = _row_keys(references, ["subject", "window_id"])
    expected_candidates = {
        (subject, condition, window_id, roi, method)
        for subject, window_id in reference_keys
        for condition in FORMAL_CONDITIONS
        for roi in ROIS
        for method in METHODS
    }
    if (
        len(candidates) != len(expected_candidates)
        or _row_keys(candidates, CANDIDATE_KEY) != expected_candidates
    ):
        raise ValueError("candidate grid does not match references")

    expected_audit = {
        (subject, condition, window_id, roi)
        for subject, window_id in reference_keys
        for condition in FORMAL_CONDITIONS
        for roi in ROIS
    }
    if (
        len(audit) != len(expected_audit)
        or _row_keys(audit, AUDIT_KEY) != expected_audit
    ):
        raise ValueError("condition audit grid does not match references")


# Check candidate values
def _validate_candidates(
    references: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    canonical = references[
        ["subject", "window_id", *REFERENCE_VALUES, "reference_valid"]
    ].rename(
        columns={
            **{name: f"stored_{name}" for name in REFERENCE_VALUES},
            "reference_valid": "stored_reference_valid",
        }
    )
    joined = candidates.merge(canonical, on=["subject", "window_id"])
    if len(joined) != len(candidates) or not np.allclose(
        joined[list(REFERENCE_VALUES)],
        joined[[f"stored_{name}" for name in REFERENCE_VALUES]],
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    ):
        raise ValueError("candidate reference values do not match")
    if not joined["reference_valid"].eq(joined["stored_reference_valid"]).all():
        raise ValueError("candidate reference validity does not match")

    statuses = candidates["window_status"]
    if statuses.isna().any() or not set(statuses).issubset(ALLOWED_STATUSES):
        raise ValueError("candidate status is not recognized")
    ok = statuses.eq("ok")
    if not np.isfinite(candidates.loc[ok, "rppg_hr_bpm"]).all():
        raise ValueError("candidate status does not match heart-rate estimate")
    unavailable = candidates.loc[
        ~ok,
        ["rppg_hr_bpm", "signed_error_bpm", "absolute_error_bpm"],
    ]
    if unavailable.notna().any().any():
        raise ValueError("candidate status does not match unavailable estimate")

    evaluated = np.isfinite(candidates["rppg_hr_bpm"]) & np.isfinite(
        candidates["reference_hr_bpm"]
    )
    expected = (
        candidates.loc[evaluated, "rppg_hr_bpm"]
        - candidates.loc[evaluated, "reference_hr_bpm"]
    )
    if (
        not np.allclose(candidates.loc[evaluated, "signed_error_bpm"], expected)
        or not np.allclose(
            candidates.loc[evaluated, "absolute_error_bpm"], expected.abs()
        )
        or candidates.loc[
            ~evaluated,
            ["signed_error_bpm", "absolute_error_bpm"],
        ]
        .notna()
        .any()
        .any()
    ):
        raise ValueError("candidate errors do not match heart-rate estimates")


# Check degradation strength
def _validate_degradation(audit: pd.DataFrame) -> None:
    for condition, target in (("fps15", 15.0), ("fps10", 10.0)):
        rows = audit[audit["condition"].eq(condition)]
        available = rows[rows["retained_frame_count"].ge(2)]
        if not np.allclose(rows["target_fps"], target) or (
            available.empty
            or not np.isfinite(available["effective_fps"]).all()
            or (available["effective_fps"] - target).abs().gt(0.4).any()
        ):
            raise ValueError("effective frame rate outside tolerance")

    for condition, shift in (("roi_shift_3", 0.03), ("roi_shift_5", 0.05)):
        rows = audit[audit["condition"].eq(condition)]
        available = rows[rows["auditable_frame_count"].gt(0)]
        observed = available["max_roi_shift_fraction"]
        if not np.allclose(rows["roi_shift_fraction"], shift) or (
            available.empty
            or not np.isfinite(observed).all()
            or observed.gt(shift + 1e-9).any()
            or observed.lt(shift - 0.001).any()
        ):
            raise ValueError("observed ROI shift outside tolerance")


# Validate in-memory tables
def validate_tables(
    references: pd.DataFrame,
    candidates: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, object]:
    _validate_grids(references, candidates, audit)
    _validate_candidates(references, candidates)
    _validate_degradation(audit)
    subjects = references["subject"].unique()
    return {
        "subject_count": len(subjects),
        "completed_subject_count": len(subjects),
        "reference_window_count": len(references),
        "valid_reference_window_count": int(references["reference_valid"].sum()),
        "candidate_count": len(candidates),
        "condition_audit_count": len(audit),
        "candidate_status_counts": {
            str(key): int(value)
            for key, value in candidates["window_status"].value_counts().items()
        },
    }


# Run and record experiment
def run_formal_experiment(
    dataset_root: str | Path,
    face_model: str | Path,
    baseline_output: str | Path,
    output: str | Path,
) -> dict[str, object]:
    dataset_path = Path(dataset_root).resolve()
    model_path = Path(face_model).resolve()
    baseline_path = Path(baseline_output).resolve()
    output_path = Path(output).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("formal output directory must be empty")
    output_path.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC)
    start_clock = time.monotonic()
    source_root = Path(__file__).resolve().parents[1]
    reference_path = baseline_path / "reference_windows.csv"
    dataset_info = {"root": str(dataset_path), "readable": False, "subject_count": 0}
    baseline_info = {
        "path": str(baseline_path),
        "reference_window_count": 0,
        "reference_sha256": None,
    }
    model_info = {"path": str(model_path), "size_bytes": None, "sha256": None}
    failure: Exception | None = None

    try:
        if not dataset_path.is_dir():
            raise ValueError("dataset root is not readable")
        dataset_info["readable"] = True
        subjects = discover_subjects(dataset_path)
        dataset_info["subject_count"] = len(subjects)
        if len(subjects) != FORMAL_SUBJECT_COUNT:
            raise ValueError("dataset subject count does not match formal protocol")

        if not model_path.is_file():
            raise ValueError("face model does not exist")
        model_info["size_bytes"] = model_path.stat().st_size
        model_info["sha256"] = file_sha256(model_path)

        if not reference_path.is_file():
            raise ValueError("baseline reference table does not exist")
        references = pd.read_csv(reference_path)
        baseline_info["reference_window_count"] = len(references)
        baseline_info["reference_sha256"] = file_sha256(reference_path)
        if len(references) != FORMAL_REFERENCE_WINDOW_COUNT:
            raise ValueError("baseline window count does not match formal protocol")
        if references.duplicated(["subject", "window_id"]).any():
            raise ValueError("duplicate baseline reference keys")
        if {subject.name for subject in subjects} != set(references["subject"]):
            raise ValueError("baseline subjects do not match dataset")
        if not all(
            (baseline_path / "traces" / f"{subject.name}.csv").is_file()
            for subject in subjects
        ):
            raise ValueError("baseline trace is missing")

        tables = run_degradation_experiment(
            subjects,
            model_path,
            baseline_path,
            references,
        )
        summary = validate_tables(references, tables["candidates"], tables["audit"])
        for name, key in OUTPUT_TABLES.items():
            tables[key].to_csv(output_path / name, index=False)
        status = "completed"
        anomalies: list[dict[str, str]] = []
    except Exception as error:
        failure = error
        status = "failed"
        summary = {"completed_subject_count": 0}
        anomalies = [{"type": type(error).__name__, "message": str(error)}]

    code = collect_git_provenance()
    code["source_sha256"] = _source_hashes(source_root)
    output_files = {
        name: {"sha256": file_sha256(output_path / name)}
        for name in OUTPUT_TABLES
        if (output_path / name).is_file()
    }
    manifest: dict[str, object] = {
        "schema_version": "degradation_run_v1",
        "status": status,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "duration_sec": round(time.monotonic() - start_clock, 3),
        "environment": collect_environment_versions(),
        "dataset": dataset_info,
        "baseline": baseline_info,
        "model": model_info,
        "code": code,
        "output": str(output_path),
        "output_files": output_files,
        "result": summary,
        "anomalies": anomalies,
    }
    (output_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if failure is not None:
        raise failure
    return manifest


# Parse run arguments
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--face-model", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = run_formal_experiment(
        args.dataset_root,
        args.face_model,
        args.baseline_output,
        args.output,
    )
    print(json.dumps(manifest["result"], indent=2))


if __name__ == "__main__":
    main()
