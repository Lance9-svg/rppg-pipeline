"""Run formal Phase 2 POS/CHROM processing from existing RGB traces."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from rppg_pipeline.standard_rppg import (
    StandardRPPGConfig,
    build_method_roi_metrics,
    build_subject_qc,
    process_subject_windows,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run formal window-aligned POS/CHROM estimates from existing RGB traces."
        )
    )
    parser.add_argument(
        "--phase1-dir",
        required=True,
        help="Directory containing ppg_reference_windows.csv.",
    )
    parser.add_argument(
        "--trace-root",
        required=True,
        help="Directory containing subject<number>_roi/rgb_trace.csv outputs.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Phase 2 output directory, normally results_v2/phase2.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Optional original subject IDs for a fixed pilot subset.",
    )
    return parser.parse_args(argv)


def run_phase2(args: argparse.Namespace) -> int:
    """Run Phase 2 without reading videos or changing the Phase 1 reference."""
    phase1_dir = Path(args.phase1_dir)
    trace_root = Path(args.trace_root)
    out_dir = Path(args.out)
    reference_path = phase1_dir / "ppg_reference_windows.csv"
    reference_df = pd.read_csv(reference_path)

    subject_names = sorted(
        reference_df["subject"].unique(),
        key=_subject_id,
    )
    if args.subjects:
        selected_ids = set(args.subjects)
        available_ids = {_subject_id(subject) for subject in subject_names}
        missing_ids = sorted(selected_ids - available_ids)
        if missing_ids:
            raise ValueError(f"Subject IDs not found in Phase 1: {missing_ids}")
        subject_names = [
            subject
            for subject in subject_names
            if _subject_id(subject) in selected_ids
        ]

    config = StandardRPPGConfig()
    result_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for index, subject in enumerate(subject_names, start=1):
        subject_start = time.perf_counter()
        subject_reference = reference_df[reference_df["subject"] == subject].copy()
        trace_path = trace_root / f"{subject}_roi" / "rgb_trace.csv"
        trace_df = pd.read_csv(trace_path)
        subject_results = process_subject_windows(
            subject,
            subject_reference,
            trace_df,
            config,
        )
        result_frames.append(subject_results)
        elapsed = time.perf_counter() - subject_start
        runtime_rows.append(
            {
                "subject": subject,
                "subject_id": _subject_id(subject),
                "reference_windows": len(subject_reference),
                "result_rows": len(subject_results),
                "runtime_sec": elapsed,
            }
        )
        print(
            f"[{index}/{len(subject_names)}] {subject}: "
            f"{len(subject_results)} rows in {elapsed:.2f}s"
        )

    results = pd.concat(result_frames, ignore_index=True)
    subject_qc = build_subject_qc(results)
    metrics = build_method_roi_metrics(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "rppg_window_results.csv", index=False)
    subject_qc.to_csv(out_dir / "phase2_subject_qc.csv", index=False)
    metrics.to_csv(out_dir / "method_roi_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(
        out_dir / "phase2_runtime.csv",
        index=False,
    )

    status_counts = (
        results["window_status"].value_counts(dropna=False).sort_index().to_dict()
    )
    total_runtime = time.perf_counter() - total_start
    run_summary = {
        "subjects": len(subject_names),
        "reference_windows": int(
            reference_df[reference_df["subject"].isin(subject_names)].shape[0]
        ),
        "result_rows": len(results),
        "expected_rows": int(
            reference_df[reference_df["subject"].isin(subject_names)].shape[0]
            * len(config.rois)
            * len(config.methods)
        ),
        "status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "finite_hr_rows": int(np.isfinite(results["rppg_hr_bpm"]).sum()),
        "primary_evaluated_rows": int(
            (
                results["eligible_primary"].astype(bool)
                & results["window_status"].eq("ok")
                & np.isfinite(results["rppg_hr_bpm"])
            ).sum()
        ),
        "total_runtime_sec": total_runtime,
    }
    (out_dir / "phase2_run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    protocol = {
        **config.to_dict(),
        "phase1_dir": str(phase1_dir.resolve()),
        "trace_root": str(trace_root.resolve()),
        "selected_subject_ids": [_subject_id(name) for name in subject_names],
        "primary_reference": "Phase 1 time_hr_bpm (median inter-beat interval)",
        "reference_qc": "Phase 1 frequency_hr_bpm and reference_category",
        "algorithm_sources": {
            "CHROM": "de Haan and Jeanne (2013), DOI 10.1109/TBME.2013.2266196",
            "POS": "Wang et al. (2017), DOI 10.1109/TBME.2016.2609282",
        },
    }
    (out_dir / "phase2_config.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    print(f"Phase 2 completed subjects: {len(subject_names)}/{len(subject_names)}")
    print(f"Rows: {len(results)}/{run_summary['expected_rows']}")
    print(results["window_status"].value_counts().to_string())
    print(f"Outputs: {out_dir}")
    return 0


def _subject_id(subject: str) -> int:
    prefix = "subject"
    if not subject.startswith(prefix):
        raise ValueError(f"Invalid subject name in Phase 1: {subject}")
    return int(subject[len(prefix) :])


def main() -> None:
    raise SystemExit(run_phase2(parse_args()))


if __name__ == "__main__":
    main()
