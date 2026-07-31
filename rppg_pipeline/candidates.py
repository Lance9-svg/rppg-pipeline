"""Build the formal candidate table from frozen Phase 1 and Phase 2 rows."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANDIDATE_KEY = ("subject", "window_id", "roi", "method")
WINDOW_KEY = ("subject", "window_id")
FORMAL_METHODS = ("CHROM", "POS")
FORMAL_ROIS = (
    "full_face_inner",
    "forehead",
    "left_cheek",
    "right_cheek",
    "cheeks_mean",
)
ERROR_THRESHOLDS_BPM = (3.0, 5.0, 10.0)

_PHASE1_REQUIRED = {
    *WINDOW_KEY,
    "start_time_sec",
    "end_time_sec",
    "window_center_time_sec",
    "duration_sec",
    "time_hr_bpm",
    "frequency_hr_bpm",
    "reference_category",
    "eligible_primary",
}
_PHASE2_REQUIRED = {
    *CANDIDATE_KEY,
    "start_time_sec",
    "end_time_sec",
    "window_center_time_sec",
    "duration_sec",
    "reference_hr_bpm",
    "reference_frequency_hr_bpm",
    "reference_category",
    "eligible_primary",
    "window_status",
    "rppg_hr_bpm",
    "signed_error_bpm",
    "absolute_error_bpm",
}


def build_candidate_table(
    phase1_windows: pd.DataFrame,
    phase2_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return validated candidates without recomputing frozen rPPG estimates."""
    phase1 = phase1_windows.copy()
    candidates = phase2_rows.copy()
    _require_columns(phase1, _PHASE1_REQUIRED, "Phase 1 windows")
    _require_columns(candidates, _PHASE2_REQUIRED, "Phase 2 rows")
    _validate_source_rows(phase1, candidates)

    reference_hr = _numeric(candidates, "reference_hr_bpm")
    rppg_hr = _numeric(candidates, "rppg_hr_bpm")
    absolute_error = _numeric(candidates, "absolute_error_bpm")
    primary = (
        candidates["eligible_primary"].eq(True)  # noqa: E712
        & candidates["window_status"].eq("ok")
        & np.isfinite(reference_hr)
        & np.isfinite(rppg_hr)
        & np.isfinite(absolute_error)
    )
    candidates["primary_analysis_eligible"] = primary

    for threshold in ERROR_THRESHOLDS_BPM:
        column = f"error_gt_{int(threshold)}bpm"
        labels = pd.Series(pd.NA, index=candidates.index, dtype="boolean")
        labels.loc[primary] = absolute_error.loc[primary].gt(threshold)
        candidates[column] = labels

    candidates = candidates.sort_values(list(CANDIDATE_KEY)).reset_index(drop=True)
    validate_candidate_table(candidates, phase1)
    return candidates


def validate_candidate_table(
    candidates: pd.DataFrame,
    phase1_windows: pd.DataFrame,
) -> None:
    """Reject incomplete or internally inconsistent formal candidate rows."""
    _require_columns(candidates, _PHASE2_REQUIRED, "Candidate table")
    _require_columns(
        candidates,
        {
            "primary_analysis_eligible",
            "error_gt_3bpm",
            "error_gt_5bpm",
            "error_gt_10bpm",
        },
        "Candidate table",
    )
    _validate_source_rows(phase1_windows, candidates)

    reference_hr = _numeric(candidates, "reference_hr_bpm")
    rppg_hr = _numeric(candidates, "rppg_hr_bpm")
    absolute_error = _numeric(candidates, "absolute_error_bpm")
    expected_primary = (
        candidates["eligible_primary"].eq(True)  # noqa: E712
        & candidates["window_status"].eq("ok")
        & np.isfinite(reference_hr)
        & np.isfinite(rppg_hr)
        & np.isfinite(absolute_error)
    )
    if not candidates["primary_analysis_eligible"].equals(expected_primary):
        raise ValueError("Primary analysis eligibility is inconsistent")

    for threshold in ERROR_THRESHOLDS_BPM:
        column = f"error_gt_{int(threshold)}bpm"
        actual = candidates[column].astype("boolean")
        if not actual.loc[~expected_primary].isna().all():
            raise ValueError(f"{column} must be missing outside the primary analysis")
        expected = absolute_error.loc[expected_primary].gt(threshold).astype("boolean")
        if not actual.loc[expected_primary].equals(expected):
            raise ValueError(f"{column} is inconsistent with absolute_error_bpm")


def _validate_source_rows(
    phase1_windows: pd.DataFrame,
    phase2_rows: pd.DataFrame,
) -> None:
    _validate_identifiers(phase1_windows, WINDOW_KEY, "Phase 1 windows")
    _validate_identifiers(phase2_rows, CANDIDATE_KEY, "Phase 2 rows")
    if phase1_windows.duplicated(list(WINDOW_KEY)).any():
        raise ValueError("Duplicate Phase 1 window keys")
    if phase2_rows.duplicated(list(CANDIDATE_KEY)).any():
        raise ValueError("Duplicate candidate keys")

    methods = set(phase2_rows["method"])
    if not methods.issubset(FORMAL_METHODS):
        unexpected_methods = sorted(methods - set(FORMAL_METHODS))
        raise ValueError(f"Unexpected method values: {unexpected_methods}")
    rois = set(phase2_rows["roi"])
    if not rois.issubset(FORMAL_ROIS):
        raise ValueError(f"Unexpected ROI values: {sorted(rois - set(FORMAL_ROIS))}")

    phase1_keys = set(phase1_windows.loc[:, list(WINDOW_KEY)].itertuples(False, None))
    phase2_window_keys = set(
        phase2_rows.loc[:, list(WINDOW_KEY)].itertuples(False, None)
    )
    extra_windows = phase2_window_keys - phase1_keys
    if extra_windows:
        raise ValueError(
            f"Phase 2 candidate windows not present in Phase 1: {sorted(extra_windows)}"
        )

    expected_keys = {
        (*window_key, roi, method)
        for window_key in phase1_keys
        for roi in FORMAL_ROIS
        for method in FORMAL_METHODS
    }
    actual_keys = set(phase2_rows.loc[:, list(CANDIDATE_KEY)].itertuples(False, None))
    missing_keys = expected_keys - actual_keys
    if missing_keys:
        raise ValueError(f"Missing candidate keys: {len(missing_keys)}")

    _validate_reference_alignment(phase1_windows, phase2_rows)
    _validate_status_and_errors(phase2_rows)


def _validate_reference_alignment(
    phase1_windows: pd.DataFrame,
    phase2_rows: pd.DataFrame,
) -> None:
    reference_columns = [
        *WINDOW_KEY,
        "start_time_sec",
        "end_time_sec",
        "window_center_time_sec",
        "duration_sec",
        "time_hr_bpm",
        "frequency_hr_bpm",
        "reference_category",
        "eligible_primary",
    ]
    merged = phase2_rows.merge(
        phase1_windows.loc[:, reference_columns],
        on=list(WINDOW_KEY),
        how="left",
        validate="many_to_one",
        suffixes=("", "_phase1"),
    )
    numeric_pairs = (
        ("start_time_sec", "start_time_sec_phase1"),
        ("end_time_sec", "end_time_sec_phase1"),
        ("window_center_time_sec", "window_center_time_sec_phase1"),
        ("duration_sec", "duration_sec_phase1"),
        ("reference_hr_bpm", "time_hr_bpm"),
        ("reference_frequency_hr_bpm", "frequency_hr_bpm"),
    )
    for candidate_column, reference_column in numeric_pairs:
        if not np.isclose(
            _numeric(merged, candidate_column),
            _numeric(merged, reference_column),
            equal_nan=True,
        ).all():
            raise ValueError(f"Phase 1 alignment mismatch in {candidate_column}")

    if not merged["reference_category"].equals(merged["reference_category_phase1"]):
        raise ValueError("Phase 1 alignment mismatch in reference_category")
    if not merged["eligible_primary"].equals(merged["eligible_primary_phase1"]):
        raise ValueError("Phase 1 alignment mismatch in eligible_primary")


def _validate_status_and_errors(rows: pd.DataFrame) -> None:
    reference_hr = _numeric(rows, "reference_hr_bpm")
    rppg_hr = _numeric(rows, "rppg_hr_bpm")
    signed_error = _numeric(rows, "signed_error_bpm")
    absolute_error = _numeric(rows, "absolute_error_bpm")
    estimable = rows["window_status"].eq("ok")
    any_finite_estimate = (
        np.isfinite(rppg_hr)
        | np.isfinite(signed_error)
        | np.isfinite(absolute_error)
    )
    all_finite_estimate = (
        np.isfinite(rppg_hr)
        & np.isfinite(signed_error)
        & np.isfinite(absolute_error)
    )
    if ((~estimable) & any_finite_estimate).any():
        raise ValueError("Non-estimable candidates have finite HR or error values")
    if (estimable & ~all_finite_estimate).any():
        raise ValueError("Estimable candidates are missing finite HR or error values")

    evaluated = estimable & np.isfinite(reference_hr)
    expected_signed = rppg_hr.loc[evaluated] - reference_hr.loc[evaluated]
    if not np.isclose(
        signed_error.loc[evaluated], expected_signed, equal_nan=False
    ).all():
        raise ValueError("Candidate rows contain inconsistent error values")
    if not np.isclose(
        absolute_error.loc[evaluated], expected_signed.abs(), equal_nan=False
    ).all():
        raise ValueError("Candidate rows contain inconsistent error values")


def _validate_identifiers(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{label} contains missing {column}")
        blank = frame[column].astype(str).str.strip().eq("")
        if frame[column].dtype == object and blank.any():
            raise ValueError(f"{label} contains blank {column}")


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="raise")
