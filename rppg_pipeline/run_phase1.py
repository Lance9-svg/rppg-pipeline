"""Run the Phase 1 PPG-only reference pilot across UBFC subjects."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import pandas as pd

from rppg_pipeline.ppg_reference import (
    PPGReferenceConfig,
    WindowExample,
    build_subject_qc,
    build_subject_reference,
)
from rppg_pipeline.ubfc import (
    discover_subjects,
    read_ubfc_ground_truth,
    select_subjects,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build timestamp-aware contact-PPG references without rPPG data."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Directory containing canonical subject<number> folders.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Phase 1 output directory, normally results_v2/phase1.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Optional original subject IDs for a fixed pilot subset.",
    )
    parser.add_argument(
        "--prominence-mad-multiplier",
        type=float,
        default=0.5,
        help="Adaptive prominence threshold multiplier.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help="Primary PPG reference window length.",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=1.0,
        help="PPG reference window step.",
    )
    return parser.parse_args(argv)


def run_phase1(args: argparse.Namespace) -> int:
    """Run Phase 1 without loading videos, RGB traces, or rPPG results."""
    subjects = select_subjects(
        discover_subjects(args.dataset_root),
        args.subjects,
    )
    if not subjects:
        raise ValueError("No canonical subject<number> directories were found")

    config = PPGReferenceConfig(
        prominence_mad_multiplier=args.prominence_mad_multiplier,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
    )
    out_dir = Path(args.out)
    plot_dir = out_dir / "ppg_reference_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []
    examples: dict[str, WindowExample] = {}

    for subject in subjects:
        ground_truth = read_ubfc_ground_truth(subject.ground_truth_path)
        result = build_subject_reference(subject.name, ground_truth, config)
        uniform = result.uniform_ground_truth
        manifest_rows.append(
            {
                "subject": subject.name,
                "subject_id": subject.subject_id,
                "ground_truth_path": str(subject.ground_truth_path),
                "video_path": str(subject.video_path),
                "original_sample_count": uniform.original_sample_count,
                "unique_timestamp_count": uniform.unique_timestamp_count,
                "duplicate_timestamp_count": uniform.duplicate_timestamp_count,
                "sensor_wrap_corrected_sample_count": (
                    uniform.sensor_wrap_corrected_sample_count
                ),
                "uniform_sample_count": len(uniform.time_sec),
                "median_dt_sec": uniform.median_dt_sec,
                "sample_rate_hz": uniform.sample_rate_hz,
                "uniform_duration_sec": uniform.time_sec[-1] - uniform.time_sec[0],
            }
        )
        window_frames.append(result.windows)
        qc_rows.append(build_subject_qc(subject.name, result))
        for category, example in result.examples.items():
            examples.setdefault(category, example)

    manifest_df = pd.DataFrame(manifest_rows)
    qc_df = pd.DataFrame(qc_rows)
    windows_df = pd.concat(window_frames, ignore_index=True)
    manifest_df.to_csv(out_dir / "subject_manifest.csv", index=False)
    qc_df.to_csv(out_dir / "ppg_reference_qc.csv", index=False)
    windows_df.to_csv(out_dir / "ppg_reference_windows.csv", index=False)
    (out_dir / "phase1_config.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
    for category in ("concordant", "uncertain", "discordant", "insufficient"):
        example = examples.get(category)
        if example is not None:
            _plot_example(
                example,
                plot_dir / f"{category}_{example.subject}_w{example.window_id}.png",
            )

    print(f"Phase 1 completed subjects: {len(subjects)}/{len(subjects)}")
    print(windows_df["reference_category"].value_counts().to_string())
    print(f"Outputs: {out_dir}")
    return 0


def _plot_example(example: WindowExample, out_path: Path) -> None:
    relative_time = example.time_sec - example.time_sec[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(relative_time, example.filtered_ppg, linewidth=1.2, color="#205493")
    if len(example.peak_indices):
        ax.scatter(
            relative_time[example.peak_indices],
            example.filtered_ppg[example.peak_indices],
            s=20,
            color="#d62728",
            label="Detected peaks",
            zorder=3,
        )
    ax.set(
        xlabel="Window time (sec)",
        ylabel="Filtered contact PPG",
        title=(
            f"{example.subject}, window {example.window_id}, {example.category}\n"
            f"time HR={example.time_hr_bpm:.2f}, "
            f"frequency HR={example.frequency_hr_bpm:.2f} bpm"
        ),
    )
    ax.grid(True, alpha=0.25)
    if len(example.peak_indices):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    raise SystemExit(run_phase1(parse_args()))


if __name__ == "__main__":
    main()
