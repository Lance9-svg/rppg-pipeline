from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rppg_pipeline import formal_run
from rppg_pipeline.degradation import CONDITION_SETTINGS, FORMAL_CONDITIONS, ROIS
from rppg_pipeline.formal_run import validate_tables
from rppg_pipeline.rppg import METHODS


def test_validate_tables_accepts_complete_grid() -> None:
    references = _references()
    tables = _tables(references)

    summary = validate_tables(references, tables["candidates"], tables["audit"])

    assert summary == {
        "subject_count": 1,
        "completed_subject_count": 1,
        "reference_window_count": 1,
        "valid_reference_window_count": 1,
        "candidate_count": 30,
        "condition_audit_count": 15,
        "candidate_status_counts": {"ok": 30},
    }


@pytest.mark.parametrize("table", ["candidates", "audit"])
def test_validate_tables_rejects_incomplete_grid(table: str) -> None:
    references = _references()
    tables = _tables(references)
    tables[table].loc[0, "roi"] = "not_a_formal_roi"

    with pytest.raises(ValueError, match="grid"):
        validate_tables(references, tables["candidates"], tables["audit"])


@pytest.mark.parametrize(
    ("condition", "column", "value", "message"),
    [
        ("fps10", "effective_fps", 12.0, "frame rate"),
        ("roi_shift_5", "max_roi_shift_fraction", 0.02, "ROI shift"),
    ],
)
def test_validate_tables_rejects_wrong_degradation_strength(
    condition: str,
    column: str,
    value: float,
    message: str,
) -> None:
    references = _references()
    tables = _tables(references)
    rows = tables["audit"]["condition"].eq(condition)
    tables["audit"].loc[rows, column] = value
    with pytest.raises(ValueError, match=message):
        validate_tables(references, tables["candidates"], tables["audit"])


def test_validate_tables_rejects_unavailable_finite_estimate() -> None:
    references = _references()
    tables = _tables(references)
    tables["candidates"].loc[0, "window_status"] = "low_video_coverage"

    with pytest.raises(ValueError, match="status"):
        validate_tables(references, tables["candidates"], tables["audit"])


def test_run_formal_experiment_checks_inputs_once_and_writes_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset, model, baseline, output = _inputs(tmp_path)
    references = pd.read_csv(baseline / "reference_windows.csv")
    tables = _tables(references)
    _patch_protocol(monkeypatch)
    read_paths = []

    def read_csv(path, *args, **kwargs):
        read_paths.append(Path(path))
        return references

    def run(subjects, face_model, baseline_output, baseline_references):
        assert [subject.name for subject in subjects] == ["subject1"]
        assert face_model == model.resolve()
        assert baseline_output == baseline.resolve()
        assert baseline_references is references
        return tables

    monkeypatch.setattr(formal_run.pd, "read_csv", read_csv)
    monkeypatch.setattr(formal_run, "run_degradation_experiment", run)

    manifest = formal_run.run_formal_experiment(dataset, model, baseline, output)

    assert read_paths == [baseline / "reference_windows.csv"]
    assert manifest["status"] == "completed"
    assert manifest["result"]["candidate_count"] == 30
    assert set(manifest["output_files"]) == {
        "degradation_candidates.csv",
        "condition_audit.csv",
    }
    assert not (output / "reference_windows.csv").exists()
    assert not (output / "traces").exists()
    assert json.loads((output / "run_manifest.json").read_text()) == manifest


def test_run_formal_experiment_rejects_missing_trace_before_processing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset, model, baseline, output = _inputs(tmp_path)
    (baseline / "traces" / "subject1.csv").unlink()
    _patch_protocol(monkeypatch)
    monkeypatch.setattr(
        formal_run,
        "run_degradation_experiment",
        lambda *args: pytest.fail("processing started before input check"),
    )

    with pytest.raises(ValueError, match="baseline trace"):
        formal_run.run_formal_experiment(dataset, model, baseline, output)

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["result"] == {"completed_subject_count": 0}


def test_run_formal_experiment_records_processing_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset, model, baseline, output = _inputs(tmp_path)
    _patch_protocol(monkeypatch)

    def fail(*args):
        raise RuntimeError("video processing failed")

    monkeypatch.setattr(formal_run, "run_degradation_experiment", fail)

    with pytest.raises(RuntimeError, match="video processing failed"):
        formal_run.run_formal_experiment(dataset, model, baseline, output)

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["anomalies"] == [
        {"type": "RuntimeError", "message": "video processing failed"}
    ]


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "dataset"
    subject = dataset / "subject1"
    subject.mkdir(parents=True)
    (subject / "vid.avi").touch()
    (subject / "ground_truth.txt").touch()
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"model")
    baseline = tmp_path / "baseline"
    (baseline / "traces").mkdir(parents=True)
    (baseline / "traces" / "subject1.csv").write_text("time_sec\n0.0\n")
    _references().to_csv(baseline / "reference_windows.csv", index=False)
    return dataset, model, baseline, tmp_path / "output"


def _patch_protocol(monkeypatch) -> None:
    monkeypatch.setattr(formal_run, "FORMAL_SUBJECT_COUNT", 1)
    monkeypatch.setattr(formal_run, "FORMAL_REFERENCE_WINDOW_COUNT", 1)
    monkeypatch.setattr(
        formal_run,
        "collect_environment_versions",
        lambda: {"python": "test"},
    )
    monkeypatch.setattr(
        formal_run,
        "collect_git_provenance",
        lambda: {"branch": "rppg-degradation", "commit": "test-commit"},
    )
    monkeypatch.setattr(
        formal_run,
        "_source_hashes",
        lambda root: {"rppg_pipeline/formal_run.py": "source-hash"},
    )


def _references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "subject1",
                "window_id": 1,
                "start_time_sec": 0.0,
                "end_time_sec": 10.0,
                "duration_sec": 10.0,
                "reference_hr_bpm": 72.0,
                "reference_valid": True,
            }
        ]
    )


def _tables(references: pd.DataFrame) -> dict[str, pd.DataFrame]:
    audits = []
    candidates = []
    for reference in references.itertuples(index=False):
        for condition in FORMAL_CONDITIONS:
            factor, severity, target_fps = CONDITION_SETTINGS[condition]
            effective_fps = target_fps if np.isfinite(target_fps) else 30.0
            shift = severity if condition.startswith("roi_shift") else 0.0
            for roi in ROIS:
                audit = {
                    "subject": reference.subject,
                    "condition": condition,
                    "window_id": reference.window_id,
                    "roi": roi,
                    "degradation_factor": factor,
                    "degradation_severity": severity,
                    "target_fps": target_fps,
                    "effective_fps": effective_fps,
                    "source_frame_count": 300,
                    "retained_frame_count": int(round(effective_fps * 10.0)),
                    "roi_shift_fraction": shift,
                    "auditable_frame_count": 300,
                    "auditable_frame_fraction": 1.0,
                    "max_roi_shift_fraction": shift,
                    "mean_roi_retention_ratio": 1.0 - shift,
                    "min_roi_retention_ratio": 1.0 - shift,
                }
                audits.append(audit)
                for method in METHODS:
                    candidates.append(
                        {
                            "subject": reference.subject,
                            "condition": condition,
                            "window_id": reference.window_id,
                            "roi": roi,
                            "start_time_sec": reference.start_time_sec,
                            "end_time_sec": reference.end_time_sec,
                            "duration_sec": reference.duration_sec,
                            "method": method,
                            "reference_hr_bpm": reference.reference_hr_bpm,
                            "reference_valid": reference.reference_valid,
                            "window_status": "ok",
                            "rppg_hr_bpm": 73.0,
                            "signed_error_bpm": 1.0,
                            "absolute_error_bpm": 1.0,
                        }
                    )
    return {
        "candidates": pd.DataFrame(candidates),
        "audit": pd.DataFrame(audits),
    }
