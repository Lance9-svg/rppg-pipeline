from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rppg_pipeline.candidates import build_candidate_table

ROIS = (
    "full_face_inner",
    "forehead",
    "left_cheek",
    "right_cheek",
    "cheeks_mean",
)
METHODS = ("CHROM", "POS")


def test_build_candidate_table_preserves_keys_and_builds_primary_targets() -> None:
    phase1, phase2 = _candidate_inputs()

    candidates = build_candidate_table(phase1, phase2)

    assert len(candidates) == 20
    assert not candidates.duplicated(["subject", "window_id", "roi", "method"]).any()
    assert candidates["primary_analysis_eligible"].sum() == 9
    assert candidates["error_gt_3bpm"].dtype == "boolean"
    assert candidates["error_gt_5bpm"].dtype == "boolean"
    assert candidates["error_gt_10bpm"].dtype == "boolean"
    assert candidates.loc[candidates["window_id"].eq(2), "error_gt_5bpm"].isna().all()
    assert "spectral_entropy" in candidates.columns
    assert candidates["spectral_entropy"].notna().all()
    assert list(candidates["method"].head(2)) == ["CHROM", "POS"]

    at_three = _candidate(candidates, 1, "full_face_inner", "POS")
    at_five = _candidate(candidates, 1, "forehead", "POS")
    over_five = _candidate(candidates, 1, "left_cheek", "CHROM")
    at_ten = _candidate(candidates, 1, "right_cheek", "CHROM")
    over_ten = _candidate(candidates, 1, "right_cheek", "POS")
    failed = _candidate(candidates, 1, "cheeks_mean", "POS")

    assert bool(at_three["error_gt_3bpm"]) is False
    assert bool(at_five["error_gt_5bpm"]) is False
    assert bool(over_five["error_gt_5bpm"]) is True
    assert bool(at_ten["error_gt_10bpm"]) is False
    assert bool(over_ten["error_gt_10bpm"]) is True
    assert bool(failed["primary_analysis_eligible"]) is False
    assert pd.isna(failed["error_gt_5bpm"])


def test_build_candidate_table_rejects_duplicate_candidate_key() -> None:
    phase1, phase2 = _candidate_inputs()
    phase2 = pd.concat([phase2, phase2.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate candidate keys"):
        build_candidate_table(phase1, phase2)


def test_build_candidate_table_rejects_missing_candidate_key() -> None:
    phase1, phase2 = _candidate_inputs()
    phase2 = phase2.iloc[1:].reset_index(drop=True)

    with pytest.raises(ValueError, match="Missing candidate keys"):
        build_candidate_table(phase1, phase2)


def test_build_candidate_table_rejects_unknown_roi() -> None:
    phase1, phase2 = _candidate_inputs()
    phase2.loc[0, "roi"] = "nose"

    with pytest.raises(ValueError, match="Unexpected ROI values"):
        build_candidate_table(phase1, phase2)


def test_build_candidate_table_rejects_unknown_phase1_window() -> None:
    phase1, phase2 = _candidate_inputs()
    phase2.loc[0, "window_id"] = 999

    with pytest.raises(ValueError, match="not present in Phase 1"):
        build_candidate_table(phase1, phase2)


def test_build_candidate_table_rejects_status_with_finite_estimate() -> None:
    phase1, phase2 = _candidate_inputs()
    failed = phase2["window_status"].ne("ok")
    phase2.loc[failed, "rppg_hr_bpm"] = 90.0
    phase2.loc[failed, "signed_error_bpm"] = 20.0
    phase2.loc[failed, "absolute_error_bpm"] = 20.0

    with pytest.raises(ValueError, match="Non-estimable candidates have finite"):
        build_candidate_table(phase1, phase2)


def test_build_candidate_table_rejects_non_estimable_row_with_only_finite_hr() -> None:
    phase1, phase2 = _candidate_inputs()
    failed = phase2["window_status"].ne("ok")
    phase2.loc[failed, "rppg_hr_bpm"] = 90.0

    with pytest.raises(ValueError, match="Non-estimable candidates have finite"):
        build_candidate_table(phase1, phase2)


def test_build_candidate_table_rejects_inconsistent_error() -> None:
    phase1, phase2 = _candidate_inputs()
    phase2.loc[0, "absolute_error_bpm"] = 99.0

    with pytest.raises(ValueError, match="inconsistent error values"):
        build_candidate_table(phase1, phase2)


def _candidate_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    phase1 = pd.DataFrame(
        [
            _phase1_row(1, 70.0, "concordant", True),
            _phase1_row(2, 75.0, "uncertain", False),
        ]
    )
    errors = [2.0, 3.0, 4.0, 5.0, 5.1, 7.0, 10.0, 10.1, 12.0, np.nan]
    rows: list[dict[str, object]] = []
    for window_id, reference_hr, category, eligible in (
        (1, 70.0, "concordant", True),
        (2, 75.0, "uncertain", False),
    ):
        for index, (roi, method) in enumerate(
            (roi, method) for roi in ROIS for method in METHODS
        ):
            error = errors[index] if window_id == 1 else float(index + 1)
            status = "ok" if np.isfinite(error) else "insufficient_valid_frames"
            rppg_hr = reference_hr + error if np.isfinite(error) else np.nan
            rows.append(
                {
                    "subject": "subject1",
                    "window_id": window_id,
                    "roi": roi,
                    "method": method,
                    "start_time_sec": float(window_id - 1),
                    "end_time_sec": float(window_id + 9),
                    "window_center_time_sec": float(window_id + 4),
                    "duration_sec": 10.0,
                    "reference_hr_bpm": reference_hr,
                    "reference_frequency_hr_bpm": reference_hr + 0.2,
                    "reference_category": category,
                    "eligible_primary": eligible,
                    "window_status": status,
                    "rppg_hr_bpm": rppg_hr,
                    "signed_error_bpm": error,
                    "absolute_error_bpm": error,
                    "spectral_entropy": 0.25 + (index / 100.0),
                }
            )
    return phase1, pd.DataFrame(rows)


def _phase1_row(
    window_id: int,
    time_hr_bpm: float,
    category: str,
    eligible: bool,
) -> dict[str, object]:
    return {
        "subject": "subject1",
        "window_id": window_id,
        "start_time_sec": float(window_id - 1),
        "end_time_sec": float(window_id + 9),
        "window_center_time_sec": float(window_id + 4),
        "duration_sec": 10.0,
        "time_hr_bpm": time_hr_bpm,
        "frequency_hr_bpm": time_hr_bpm + 0.2,
        "reference_category": category,
        "eligible_primary": eligible,
    }


def _candidate(
    frame: pd.DataFrame,
    window_id: int,
    roi: str,
    method: str,
) -> pd.Series:
    selected = frame[
        frame["window_id"].eq(window_id)
        & frame["roi"].eq(roi)
        & frame["method"].eq(method)
    ]
    assert len(selected) == 1
    return selected.iloc[0]
