"""Batch entry point for processing UBFC subject directories."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from rppg_pipeline.ubfc import SubjectDataset, discover_subjects, select_subjects


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse batch options while keeping parity with ``run_single``."""
    parser = argparse.ArgumentParser(
        description=(
            "Process canonical subject<number> directories with the rPPG pipeline."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Directory containing subject<number> folders.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Root directory for subject<number>_roi outputs.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Optional original subject IDs to process.",
    )
    parser.add_argument(
        "--model",
        help="Path to the MediaPipe Face Landmarker .task model.",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=150,
        help="Save one ROI debug image every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for smoke tests.",
    )
    parser.add_argument(
        "--run-rppg",
        action="store_true",
        help="Run legacy full-segment CHROM/POS and coarse HR estimation.",
    )
    parser.add_argument(
        "--rppg-roi",
        default="cheeks_mean",
        help="ROI prefix used as the RGB source for rPPG.",
    )
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=0.7,
        help="Lower bandpass cutoff in Hz.",
    )
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=4.0,
        help="Upper bandpass cutoff in Hz.",
    )
    parser.add_argument(
        "--hr-estimator",
        choices=["welch", "fft"],
        default="welch",
        help="Spectral method used for heart-rate estimation.",
    )
    parser.add_argument(
        "--hr-window-sec",
        type=float,
        default=10.0,
        help="Sliding window length for HR curve estimates.",
    )
    parser.add_argument(
        "--hr-step-sec",
        type=float,
        default=1.0,
        help="Sliding window step for HR curve estimates.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the legacy coarse-grid evaluation for the bias audit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess subjects even when all requested outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover subjects and print commands without running them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed subject instead of continuing.",
    )
    return parser.parse_args(argv)


def required_output_files(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the files that make one requested subject run complete."""
    required = ["video_metadata.json", "video_metadata.csv"]
    if args.model:
        required.extend(
            [
                "rgb_trace.csv",
                "rgb_trace_long.csv",
                "valid_segments.csv",
                "quality_summary.json",
                "run_metadata.json",
            ]
        )
    if args.run_rppg:
        required.extend(
            [
                "rppg_signal.csv",
                "hr_results.csv",
                "runtime_results.csv",
                "hr_curve.png",
            ]
        )
    elif args.evaluate:
        required.append("hr_results.csv")
    if args.evaluate:
        required.extend(
            [
                "ppg_hr_results.csv",
                "evaluation_results.csv",
                "evaluation_metrics.csv",
                "evaluation_plot.png",
            ]
        )
    return tuple(required)


def is_output_complete(out_dir: str | Path, required: Sequence[str]) -> bool:
    """Check one exact subject ROI output directory for requested files."""
    output = Path(out_dir)
    return output.is_dir() and all((output / name).is_file() for name in required)


def build_subject_command(
    subject: SubjectDataset,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build an isolated single-subject command for the current interpreter."""
    command = [
        sys.executable,
        "-m",
        "rppg_pipeline.run_single",
        "--video",
        str(subject.video_path),
        "--out",
        str(out_dir),
    ]
    if args.model:
        command.extend(["--model", str(args.model)])
        command.extend(["--debug-every", str(args.debug_every)])
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.run_rppg:
        command.extend(
            [
                "--run-rppg",
                "--rppg-roi",
                args.rppg_roi,
                "--bandpass-low",
                str(args.bandpass_low),
                "--bandpass-high",
                str(args.bandpass_high),
                "--hr-estimator",
                args.hr_estimator,
                "--hr-window-sec",
                str(args.hr_window_sec),
                "--hr-step-sec",
                str(args.hr_step_sec),
            ]
        )
    if args.evaluate:
        command.extend(
            [
                "--evaluate",
                "--ground-truth",
                str(subject.ground_truth_path),
            ]
        )
    return command


def run_batch(args: argparse.Namespace) -> int:
    """Run selected subjects and write a machine-readable batch summary."""
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    subjects = select_subjects(
        discover_subjects(dataset_root),
        args.subjects,
    )
    if not subjects:
        raise ValueError(f"No canonical subject<number> directories in {dataset_root}")

    if args.model and not Path(args.model).is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {args.model}")

    out_root.mkdir(parents=True, exist_ok=True)
    required = required_output_files(args)
    rows: list[dict[str, object]] = []

    for index, subject in enumerate(subjects, start=1):
        out_dir = out_root / f"{subject.name}_roi"
        command = build_subject_command(subject, out_dir, args)
        row: dict[str, object] = {
            "subject": subject.name,
            "subject_id": subject.subject_id,
            "video_path": str(subject.video_path),
            "ground_truth_path": str(subject.ground_truth_path),
            "out_dir": str(out_dir),
            "command": subprocess.list2cmdline(command),
            "status": "pending",
            "return_code": "",
            "duration_sec": 0.0,
            "error": "",
        }

        print(f"[{index}/{len(subjects)}] {subject.name}", flush=True)
        missing_input = _missing_input(subject, evaluate=args.evaluate)
        if missing_input:
            row["status"] = "failed"
            row["error"] = missing_input
            rows.append(row)
            print(f"  failed: {missing_input}", flush=True)
            if args.fail_fast:
                break
            continue

        if not args.force and is_output_complete(out_dir, required):
            row["status"] = "skipped_complete"
            rows.append(row)
            print("  skipped: requested outputs already exist", flush=True)
            continue

        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            print(f"  {row['command']}", flush=True)
            continue

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            row["return_code"] = completed.returncode
            row["duration_sec"] = round(time.perf_counter() - started, 3)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "batch_stdout.log").write_text(
                completed.stdout or "",
                encoding="utf-8",
            )
            (out_dir / "batch_stderr.log").write_text(
                completed.stderr or "",
                encoding="utf-8",
            )
            if completed.returncode == 0 and is_output_complete(out_dir, required):
                row["status"] = "completed"
                print(f"  completed in {row['duration_sec']} s", flush=True)
            else:
                row["status"] = "failed"
                row["error"] = _failure_message(completed, out_dir, required)
                print(f"  failed: {row['error']}", flush=True)
        except OSError as exc:
            row["status"] = "failed"
            row["duration_sec"] = round(time.perf_counter() - started, 3)
            row["error"] = str(exc)
            print(f"  failed: {exc}", flush=True)

        rows.append(row)
        if row["status"] == "failed" and args.fail_fast:
            break

    _write_batch_summary(out_root, dataset_root, required, rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    print(
        f"Batch summary: {out_root / 'batch_summary.csv'} "
        f"({failed_count} failed)",
        flush=True,
    )
    return 1 if failed_count else 0


def _missing_input(subject: SubjectDataset, evaluate: bool) -> str:
    if not subject.video_path.is_file():
        return f"Video not found: {subject.video_path}"
    if evaluate and not subject.ground_truth_path.is_file():
        return f"Ground truth not found: {subject.ground_truth_path}"
    return ""


def _failure_message(
    completed: subprocess.CompletedProcess[str],
    out_dir: Path,
    required: Sequence[str],
) -> str:
    if completed.returncode != 0:
        stderr_lines = (completed.stderr or "").strip().splitlines()
        detail = stderr_lines[-1] if stderr_lines else "no stderr output"
        return f"Command exited with {completed.returncode}: {detail}"
    missing = [name for name in required if not (out_dir / name).is_file()]
    return f"Command succeeded but outputs are missing: {', '.join(missing)}"


def _write_batch_summary(
    out_root: Path,
    dataset_root: Path,
    required: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    fieldnames = [
        "subject",
        "subject_id",
        "video_path",
        "ground_truth_path",
        "out_dir",
        "status",
        "return_code",
        "duration_sec",
        "error",
        "command",
    ]
    with (out_root / "batch_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "dataset_root": str(dataset_root),
        "subject_count": len(rows),
        "required_outputs": list(required),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({str(row["status"]) for row in rows})
        },
        "subjects": list(rows),
    }
    (out_root / "batch_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    try:
        return_code = run_batch(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
