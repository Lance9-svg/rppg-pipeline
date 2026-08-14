# Original rPPG baseline runner

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rppg_pipeline.dataset import (
    build_reference_windows,
    discover_subjects,
    load_ground_truth,
)
from rppg_pipeline.degradation import FORMAL_CONDITIONS
from rppg_pipeline.rppg import build_candidates, build_original_candidates
from rppg_pipeline.video import FaceLandmarker, extract_rgb_trace


# Build all condition candidates
def build_subject_candidates(
    subject: str,
    references: pd.DataFrame,
    condition_traces: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames = [
        build_candidates(subject, references, condition_traces[condition], condition)
        for condition in FORMAL_CONDITIONS
    ]
    return pd.concat(frames, ignore_index=True)


# Run the original baseline
def run_original_baseline(
    dataset_root: str | Path,
    face_model: str | Path,
    output: str | Path,
    subject_names: tuple[str, ...] | None = None,
) -> dict[str, int]:
    subjects = discover_subjects(dataset_root)
    if subject_names is not None:
        selected = set(subject_names)
        subjects = [subject for subject in subjects if subject.name in selected]
    output_path = Path(output)
    trace_path = output_path / "traces"
    trace_path.mkdir(parents=True, exist_ok=True)

    reference_frames = []
    candidate_frames = []
    for index, subject in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] {subject.name}", flush=True)
        ground_truth = load_ground_truth(subject.ground_truth_path)
        references = build_reference_windows(subject.name, ground_truth)
        with FaceLandmarker(face_model) as detector:
            trace = extract_rgb_trace(subject.video_path, detector)
        trace.to_csv(trace_path / f"{subject.name}.csv", index=False)
        candidates = build_original_candidates(subject.name, references, trace)
        reference_frames.append(references)
        candidate_frames.append(candidates)

    reference_table = pd.concat(reference_frames, ignore_index=True)
    candidate_table = pd.concat(candidate_frames, ignore_index=True)
    metrics = build_original_metrics(candidate_table)
    reference_table.to_csv(output_path / "reference_windows.csv", index=False)
    candidate_table.to_csv(output_path / "original_candidates.csv", index=False)
    metrics.to_csv(output_path / "original_metrics.csv", index=False)
    return {
        "subject_count": len(subjects),
        "reference_window_count": len(reference_table),
        "candidate_count": len(candidate_table),
    }


# Summarize the original baseline
def build_original_metrics(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, roi), group in candidates.groupby(["method", "roi"], sort=True):
        reference = group[group["reference_valid"]]
        evaluated = reference[
            reference["window_status"].eq("ok")
            & np.isfinite(reference["rppg_hr_bpm"])
        ]
        errors = evaluated["signed_error_bpm"].to_numpy(dtype=float)
        rows.append(
            {
                "condition": "original",
                "method": method,
                "roi": roi,
                "n_reference_windows": len(reference),
                "n_evaluated_windows": len(evaluated),
                "availability_rate": (
                    len(evaluated) / len(reference) if len(reference) else np.nan
                ),
                "mae_bpm": (
                    float(np.mean(np.abs(errors))) if len(errors) else np.nan
                ),
                "rmse_bpm": (
                    float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan
                ),
                "bias_bpm": float(np.mean(errors)) if len(errors) else np.nan,
                "within_5_bpm_rate": (
                    float(np.mean(np.abs(errors) <= 5.0))
                    if len(errors)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
