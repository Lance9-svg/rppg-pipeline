# Command-line entry point for processing one video.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rppg_pipeline.video import read_video_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rPPG MVP pipeline on a single video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--out",
        required=True,
        help="Directory for generated result files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_video_metadata(args.video)
    metadata_dict = metadata.to_dict()

    json_path = out_dir / "video_metadata.json"
    csv_path = out_dir / "video_metadata.csv"

    json_path.write_text(
        json.dumps(metadata_dict, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    pd.DataFrame([metadata_dict]).to_csv(csv_path, index=False)

    print(json.dumps(metadata_dict, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
