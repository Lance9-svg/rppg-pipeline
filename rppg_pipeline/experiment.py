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
from rppg_pipeline.degradation import (
    FORMAL_CONDITIONS,
    downsample_trace,
    effective_sample_rate,
)
from rppg_pipeline.rppg import ROIS, build_candidates, build_original_candidates
from rppg_pipeline.video import FaceLandmarker, extract_rgb_trace

CONDITION_FACTOR = {
    "original": "none",
    "fps15": "frame_rate",
    "fps10": "frame_rate",
    "roi_shift_3": "roi_shift",
    "roi_shift_5": "roi_shift",
}
CONDITION_SEVERITY = {
    "original": np.nan,
    "fps15": 15.0,
    "fps10": 10.0,
    "roi_shift_3": 0.03,
    "roi_shift_5": 0.05,
}
TARGET_FPS = {"fps15": 15.0, "fps10": 10.0}


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


# Build five condition traces
def build_condition_traces(
    video_path: str | Path,
    face_model: str | Path,
    references: pd.DataFrame,
    original_trace: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    original = original_trace.copy()
    original["roi_shift_fraction"] = 0.0
    original["roi_shift_x_px"] = 0.0
    original["roi_shift_y_px"] = 0.0
    for roi in ROIS:
        original[f"{roi}_retention_ratio"] = 1.0

    with FaceLandmarker(face_model) as detector:
        roi_shift_3 = extract_rgb_trace(
            video_path,
            detector,
            roi_shift_fraction=0.03,
        )
    with FaceLandmarker(face_model) as detector:
        roi_shift_5 = extract_rgb_trace(
            video_path,
            detector,
            roi_shift_fraction=0.05,
        )
    return {
        "original": original,
        "fps15": downsample_trace(original, references, 15.0),
        "fps10": downsample_trace(original, references, 10.0),
        "roi_shift_3": roi_shift_3,
        "roi_shift_5": roi_shift_5,
    }


# Audit condition strength by window and ROI
def build_condition_audit(
    references: pd.DataFrame,
    condition_traces: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    original = condition_traces["original"]
    for reference in references.itertuples(index=False):
        start = float(reference.start_time_sec)
        end = float(reference.end_time_sec)
        source = original[
            (original["time_sec"] >= start) & (original["time_sec"] < end)
        ]
        for condition in FORMAL_CONDITIONS:
            trace = condition_traces[condition]
            selected = trace[
                (trace["time_sec"] >= start) & (trace["time_sec"] < end)
            ]
            time_sec = selected["time_sec"].to_numpy(dtype=float)
            face_width = selected["face_width_px"].to_numpy(dtype=float)
            shift_fraction = np.hypot(
                selected["roi_shift_x_px"].to_numpy(dtype=float),
                selected["roi_shift_y_px"].to_numpy(dtype=float),
            ) / face_width
            finite_shift = shift_fraction[np.isfinite(shift_fraction)]
            configured_shift = selected["roi_shift_fraction"].to_numpy(
                dtype=float
            )
            configured_shift = configured_shift[np.isfinite(configured_shift)]
            for roi in ROIS:
                retention = selected[f"{roi}_retention_ratio"].to_numpy(
                    dtype=float
                )
                auditable = np.isfinite(shift_fraction) & np.isfinite(retention)
                finite_retention = retention[np.isfinite(retention)]
                rows.append(
                    {
                        "subject": reference.subject,
                        "condition": condition,
                        "window_id": int(reference.window_id),
                        "roi": roi,
                        "degradation_factor": CONDITION_FACTOR[condition],
                        "degradation_severity": CONDITION_SEVERITY[condition],
                        "target_fps": TARGET_FPS.get(condition, np.nan),
                        "effective_fps": effective_sample_rate(time_sec),
                        "source_frame_count": len(source),
                        "retained_frame_count": len(selected),
                        "roi_shift_fraction": (
                            float(np.mean(configured_shift))
                            if len(configured_shift)
                            else np.nan
                        ),
                        "auditable_frame_count": int(np.count_nonzero(auditable)),
                        "auditable_frame_fraction": (
                            float(np.mean(auditable)) if len(auditable) else 0.0
                        ),
                        "max_roi_shift_fraction": (
                            float(np.max(finite_shift))
                            if len(finite_shift)
                            else np.nan
                        ),
                        "mean_roi_retention_ratio": (
                            float(np.mean(finite_retention))
                            if len(finite_retention)
                            else np.nan
                        ),
                        "min_roi_retention_ratio": (
                            float(np.min(finite_retention))
                            if len(finite_retention)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


# Check stored baseline references
def verify_reference_windows(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    baseline = baseline.sort_values("window_id").reset_index(drop=True)
    current = current.sort_values("window_id").reset_index(drop=True)
    keys = ["subject", "window_id", "reference_valid"]
    numbers = [
        "start_time_sec",
        "end_time_sec",
        "reference_hr_bpm",
    ]
    same_keys = len(baseline) == len(current) and baseline[keys].equals(
        current[keys]
    )
    same_numbers = len(baseline) == len(current) and np.allclose(
        baseline[numbers],
        current[numbers],
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    )
    if not same_keys or not same_numbers:
        raise ValueError("baseline reference windows do not match current ground truth")


# Run the five condition experiment
def run_degradation_experiment(
    dataset_root: str | Path,
    face_model: str | Path,
    baseline_output: str | Path,
    output: str | Path,
    subject_names: tuple[str, ...] | None = None,
) -> dict[str, int]:
    subjects = discover_subjects(dataset_root)
    if subject_names is not None:
        selected = set(subject_names)
        subjects = [subject for subject in subjects if subject.name in selected]
    output_path = Path(output)
    baseline_path = Path(baseline_output)
    baseline_references = pd.read_csv(baseline_path / "reference_windows.csv")
    trace_path = output_path / "traces"
    for condition in FORMAL_CONDITIONS:
        (trace_path / condition).mkdir(parents=True, exist_ok=True)

    reference_frames = []
    candidate_frames = []
    audit_frames = []
    for index, subject in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] {subject.name}", flush=True)
        ground_truth = load_ground_truth(subject.ground_truth_path)
        current_references = build_reference_windows(subject.name, ground_truth)
        references = baseline_references[
            baseline_references["subject"].eq(subject.name)
        ].copy()
        verify_reference_windows(references, current_references)
        original_trace = pd.read_csv(
            baseline_path / "traces" / f"{subject.name}.csv"
        )
        condition_traces = build_condition_traces(
            subject.video_path,
            face_model,
            references,
            original_trace,
        )
        for condition, trace in condition_traces.items():
            trace.to_csv(
                trace_path / condition / f"{subject.name}.csv",
                index=False,
            )
        audit = build_condition_audit(references, condition_traces)
        candidates = build_subject_candidates(
            subject.name,
            references,
            condition_traces,
        ).merge(
            audit,
            on=["subject", "condition", "window_id", "roi"],
            how="left",
        )
        reference_frames.append(references)
        candidate_frames.append(candidates)
        audit_frames.append(audit)

    reference_table = pd.concat(reference_frames, ignore_index=True)
    candidate_table = pd.concat(candidate_frames, ignore_index=True)
    audit_table = pd.concat(audit_frames, ignore_index=True)
    metrics = build_metrics(candidate_table)
    reference_table.to_csv(output_path / "reference_windows.csv", index=False)
    candidate_table.to_csv(output_path / "degradation_candidates.csv", index=False)
    metrics.to_csv(output_path / "degradation_metrics.csv", index=False)
    audit_table.to_csv(output_path / "condition_audit.csv", index=False)
    return {
        "subject_count": len(subjects),
        "reference_window_count": len(reference_table),
        "candidate_count": len(candidate_table),
    }


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
    return build_metrics(candidates)


# Summarize condition metrics
def build_metrics(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, method, roi), group in candidates.groupby(
        ["condition", "method", "roi"],
        sort=True,
    ):
        reference = group[group["reference_valid"]]
        evaluated = reference[
            reference["window_status"].eq("ok")
            & np.isfinite(reference["rppg_hr_bpm"])
        ]
        errors = evaluated["signed_error_bpm"].to_numpy(dtype=float)
        rows.append(
            {
                "condition": condition,
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
