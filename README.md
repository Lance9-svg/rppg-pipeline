# rPPG Pipeline

Quality-aware remote photoplethysmography (rPPG) pipeline MVP.

This project starts with a single-video workflow and will grow toward video
quality assessment for uncontrolled face videos.

## MVP Goals

The first version focuses on five end-to-end steps:

1. Read one input video.
2. Detect a face and extract forehead / cheek ROIs.
3. Convert ROI pixels into RGB time series.
4. Run CHROM and POS rPPG methods.
5. Export heart-rate curves and runtime results.

## Current Scaffold

The initial scaffold includes:

- `rppg_pipeline.run_single`: command-line entry point for one video.
- `rppg_pipeline.video`: OpenCV-based video metadata reader.
- `requirements.txt`: Python dependencies for the MVP.
- `.gitignore`: ignores local environments, raw data, results, and caches.

At this stage, `run_single` reads video metadata and writes:

- `video_metadata.json`
- `video_metadata.csv`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m rppg_pipeline.run_single `
  --video data/sample_video.mp4 `
  --out results/sample
```

UBFC-rPPG subject example:

```powershell
.\.venv\Scripts\python.exe -m rppg_pipeline.run_single `
  --video D:\DA\Project\Dataset\subject1\vid.avi `
  --out results/subject1_metadata
```

Verified metadata from the local UBFC-rPPG `subject1` video:

```text
fps: 29.264106
frame_count: 1547
duration_sec: 52.8634
width: 640
height: 480
```

## Planned Outputs

The complete MVP should produce:

- `rgb_trace.csv`
- `rppg_signal.csv`
- `hr_results.csv`
- `runtime_results.csv`
- `hr_curve.png`

## Notes

- Raw videos and generated results are intentionally ignored by Git.
- Do not report true heart-rate accuracy for videos without PPG ground truth.
- For uncontrolled videos, report quality and signal usability separately from
  heart-rate accuracy.
