# Command-line entry point for processing one video.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rppg_pipeline.rgb_trace import process_video_rgb_trace
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

    print(json.dumps(metadata_dict, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
