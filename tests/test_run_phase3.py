from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from rppg_pipeline.run_phase3 import parse_args, run_phase3

ROIS = (
    "full_face_inner",
    "forehead",
    "left_cheek",
    "right_cheek",
    "cheeks_mean",
)
METHODS = ("CHROM", "POS")


def test_parse_args_accepts_phase_inputs_and_output() -> None:
    args = parse_args(
        [
            "--phase1-dir",
            "phase1",
            "--phase2-dir",
            "phase2",
            "--out",
            "phase3",
        ]
    )

    assert args.phase1_dir == "phase1"
    assert args.phase2_dir == "phase2"
    assert args.out == "phase3"


def test_run_phase3_writes_validated_outputs(tmp_path: Path) -> None:
    phase1_dir, phase2_dir = _write_phase_inputs(tmp_path)
    out_dir = tmp_path / "phase3"

    exit_code = run_phase3(_args(phase1_dir, phase2_dir, out_dir))

    assert exit_code == 0
    assert {path.name for path in out_dir.iterdir()} == {
        "candidate_windows.csv",
        "candidate_summary.csv",
        "candidate_status_summary.csv",
        "phase3_config.json",
        "phase3_run_manifest.json",
    }
    candidates = pd.read_csv(out_dir / "candidate_windows.csv")
    summary = pd.read_csv(out_dir / "candidate_summary.csv").iloc[0]
    status = pd.read_csv(out_dir / "candidate_status_summary.csv").iloc[0]
    config = json.loads((out_dir / "phase3_config.json").read_text("utf-8"))
    manifest = json.loads((out_dir / "phase3_run_manifest.json").read_text("utf-8"))

    assert len(candidates) == 10
    assert int(summary["subjects"]) == 1
    assert int(summary["reference_windows"]) == 1
    assert int(summary["candidate_rows"]) == 10
    assert int(summary["expected_candidate_rows"]) == 10
    assert int(summary["primary_analysis_eligible_rows"]) == 10
    assert status["window_status"] == "ok"
    assert int(status["candidate_rows"]) == 10
    assert config["methods"] == ["CHROM", "POS"]
    assert config["error_thresholds_bpm"] == [3.0, 5.0, 10.0]
    assert [item["name"] for item in manifest["inputs"]] == [
        "ppg_reference_windows.csv",
        "rppg_window_results.csv",
    ]
    assert str(tmp_path.resolve()) not in str(config)
    assert str(tmp_path.resolve()) not in str(manifest)


def test_run_phase3_refuses_existing_output_directory(tmp_path: Path) -> None:
    phase1_dir, phase2_dir = _write_phase_inputs(tmp_path)
    out_dir = tmp_path / "phase3"
    out_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_phase3(_args(phase1_dir, phase2_dir, out_dir))


def _write_phase_inputs(tmp_path: Path) -> tuple[Path, Path]:
    phase1_dir = tmp_path / "phase1"
    phase2_dir = tmp_path / "phase2"
    phase1_dir.mkdir()
    phase2_dir.mkdir()
    reference_hr = 72.0
    phase1 = pd.DataFrame(
        [
            {
                "subject": "subject1",
                "window_id": 1,
                "start_time_sec": 0.0,
                "end_time_sec": 9.9667,
                "window_center_time_sec": 4.98335,
                "duration_sec": 10.0,
                "time_hr_bpm": reference_hr,
                "frequency_hr_bpm": 72.2,
                "reference_category": "concordant",
                "eligible_primary": True,
            }
        ]
    )
    rows = []
    for index, (roi, method) in enumerate(
        (roi, method) for roi in ROIS for method in METHODS
    ):
        signed_error = float(index) / 2.0
        rows.append(
            {
                "subject": "subject1",
                "window_id": 1,
                "roi": roi,
                "method": method,
                "start_time_sec": 0.0,
                "end_time_sec": 9.9667,
                "window_center_time_sec": 4.98335,
                "duration_sec": 10.0,
                "reference_hr_bpm": reference_hr,
                "reference_frequency_hr_bpm": 72.2,
                "reference_category": "concordant",
                "eligible_primary": True,
                "window_status": "ok",
                "rppg_hr_bpm": reference_hr + signed_error,
                "signed_error_bpm": signed_error,
                "absolute_error_bpm": signed_error,
                "spectral_entropy": 0.3,
            }
        )
    phase1.to_csv(phase1_dir / "ppg_reference_windows.csv", index=False)
    pd.DataFrame(rows).to_csv(
        phase2_dir / "rppg_window_results.csv",
        index=False,
    )
    return phase1_dir, phase2_dir


def _args(phase1_dir: Path, phase2_dir: Path, out_dir: Path) -> Namespace:
    return Namespace(
        phase1_dir=str(phase1_dir),
        phase2_dir=str(phase2_dir),
        out=str(out_dir),
    )
