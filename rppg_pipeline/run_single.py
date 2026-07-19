# Command-line entry point for processing one video.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rppg_pipeline.rgb_trace import process_video_rgb_trace
from rppg_pipeline.rppg import RPPGConfig, process_rppg_from_outputs
from rppg_pipeline.video import read_video_metadata


def parse_args() -> argparse.Namespace:
    # Define CLI parameters for one-video processing.
    parser = argparse.ArgumentParser(
        description="Run the rPPG MVP pipeline on a single video."
    )
    # Input video path.
    parser.add_argument("--video", required=True, help="Path to the input video.")
    # Output directory for CSV, JSON, and debug files.
    parser.add_argument(
        "--out",
        required=True,
        help="Directory for generated result files.",
    )
    # Optional MediaPipe model path for ROI extraction.
    parser.add_argument(
        "--model",
        help=(
            "Path to MediaPipe Face Landmarker .task model. "
            "If omitted, only metadata is written."
        ),
    )
    # Debug image interval in frames.
    parser.add_argument(
        "--debug-every",
        type=int,
        default=150,
        help="Save one ROI debug image every N frames when --model is provided.",
    )
    # Optional frame cap for fast local checks.
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for smoke tests.",
    )
    # Enable CHROM/POS processing from rgb_trace.csv.
    parser.add_argument(
        "--run-rppg",
        action="store_true",
        help="Run CHROM/POS and HR estimation from existing or newly written traces.",
    )
    # ROI prefix used as the RGB source for rPPG.
    parser.add_argument(
        "--rppg-roi",
        default="cheeks_mean",
        help="ROI prefix to use for rPPG processing.",
    )
    # Lower frequency cutoff for pulse band filtering.
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=0.7,
        help="Lower bandpass cutoff in Hz.",
    )
    # Upper frequency cutoff for pulse band filtering.
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=4.0,
        help="Upper bandpass cutoff in Hz.",
    )
    # HR spectral estimator.
    parser.add_argument(
        "--hr-estimator",
        choices=["welch", "fft"],
        default="welch",
        help="Spectral method used for heart-rate estimation.",
    )
    # HR curve window length.
    parser.add_argument(
        "--hr-window-sec",
        type=float,
        default=10.0,
        help="Sliding window length for HR curve estimates.",
    )
    # HR curve window step.
    parser.add_argument(
        "--hr-step-sec",
        type=float,
        default=1.0,
        help="Sliding window step for HR curve estimates.",
    )
    return parser.parse_args()


def main() -> None:
    # Run metadata export and optional ROI/RGB trace extraction.
    args = parse_args()
    out_dir = Path(args.out)
    # Create the output directory if it does not exist.
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_video_metadata(args.video)
    metadata_dict = metadata.to_dict()

    json_path = out_dir / "video_metadata.json"
    csv_path = out_dir / "video_metadata.csv"

    # Write compact metadata for later pipeline stages.
    json_path.write_text(
        json.dumps(metadata_dict, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    # Pandas writes the same metadata as a one-row CSV.
    pd.DataFrame([metadata_dict]).to_csv(csv_path, index=False)

    if args.model:
        # Process landmarks, ROIs, RGB traces, and quality files.
        process_video_rgb_trace(
            video_path=args.video,
            out_dir=out_dir,
            model_path=args.model,
            metadata=metadata,
            debug_every=args.debug_every,
            max_frames=args.max_frames,
        )

    if args.run_rppg:
        # Process CHROM/POS and HR from rgb_trace.csv and valid_segments.csv.
        process_rppg_from_outputs(
            out_dir=out_dir,
            fps=metadata.fps,
            config=RPPGConfig(
                roi=args.rppg_roi,
                bandpass_low_hz=args.bandpass_low,
                bandpass_high_hz=args.bandpass_high,
                hr_estimator=args.hr_estimator,
                hr_window_sec=args.hr_window_sec,
                hr_step_sec=args.hr_step_sec,
            ),
        )

    print(json.dumps(metadata_dict, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
