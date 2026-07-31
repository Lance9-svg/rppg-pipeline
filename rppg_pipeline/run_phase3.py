from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .candidates import (
    CANDIDATE_KEY,
    ERROR_THRESHOLDS_BPM,
    FORMAL_METHODS,
    FORMAL_ROIS,
    build_candidate_table,
)
from .provenance import build_run_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the Phase 3 formal candidate table."
    )
    parser.add_argument(
        "--phase1-dir",
        required=True,
        help="Directory containing ppg_reference_windows.csv.",
    )
    parser.add_argument(
        "--phase2-dir",
        required=True,
        help="Directory containing rppg_window_results.csv.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="New directory for Phase 3 candidate outputs.",
    )
    return parser.parse_args(argv)


def run_phase3(args: Namespace) -> int:
    phase1_path = Path(args.phase1_dir) / "ppg_reference_windows.csv"
    phase2_path = Path(args.phase2_dir) / "rppg_window_results.csv"
    output_dir = Path(args.out)

    # Refusing an existing directory prevents a new run from silently mixing with
    # or overwriting evidence from an earlier run.
    if output_dir.exists():
        raise FileExistsError(
            f"Phase 3 output directory already exists: {output_dir}"
        )

    phase1_windows = pd.read_csv(phase1_path)
    phase2_rows = pd.read_csv(phase2_path)
    candidates = build_candidate_table(phase1_windows, phase2_rows)

    expected_rows = (
        len(phase1_windows) * len(FORMAL_ROIS) * len(FORMAL_METHODS)
    )
    summary = pd.DataFrame(
        [
            {
                "subjects": int(phase1_windows["subject"].nunique()),
                "reference_windows": len(phase1_windows),
                "candidate_rows": len(candidates),
                "expected_candidate_rows": expected_rows,
                "primary_analysis_eligible_rows": int(
                    candidates["primary_analysis_eligible"].sum()
                ),
            }
        ]
    )
    status_summary = (
        candidates.groupby("window_status", dropna=False, sort=True)
        .agg(
            candidate_rows=("window_status", "size"),
            primary_analysis_eligible_rows=(
                "primary_analysis_eligible",
                "sum",
            ),
        )
        .reset_index()
    )
    status_summary["primary_analysis_eligible_rows"] = status_summary[
        "primary_analysis_eligible_rows"
    ].astype(int)

    config = {
        "stage": "phase3_candidate_table",
        "candidate_key": list(CANDIDATE_KEY),
        "methods": list(FORMAL_METHODS),
        "rois": list(FORMAL_ROIS),
        "error_thresholds_bpm": list(ERROR_THRESHOLDS_BPM),
        "primary_analysis_rule": (
            "eligible_primary is true, window_status is ok, and reference HR, "
            "rPPG HR, and absolute error are finite"
        ),
        "phase1_input": phase1_path.name,
        "phase2_input": phase2_path.name,
    }
    manifest = build_run_manifest(
        stage="phase3_candidate_table",
        inputs=[phase1_path, phase2_path],
        config=config,
    )

    output_dir.mkdir(parents=True)
    candidates.to_csv(output_dir / "candidate_windows.csv", index=False)
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    status_summary.to_csv(
        output_dir / "candidate_status_summary.csv",
        index=False,
    )
    _write_json(output_dir / "phase3_config.json", config)
    _write_json(output_dir / "phase3_run_manifest.json", manifest)

    print(
        "Phase 3 candidate table complete: "
        f"{len(candidates)} rows, "
        f"{int(summary.iloc[0]['primary_analysis_eligible_rows'])} "
        f"primary-analysis eligible"
    )
    return 0


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    raise SystemExit(run_phase3(parse_args()))


if __name__ == "__main__":
    main()
