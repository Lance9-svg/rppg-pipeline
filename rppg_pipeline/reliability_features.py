"""Build and audit the leakage-controlled reliability feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rppg_pipeline.candidates import CANDIDATE_KEY

CONTEXT_FEATURES = ("method", "roi")
QUALITY_FEATURES = (
    "spectral_peak_power_fraction",
    "spectral_entropy",
    "p95_bbox_center_jump",
    "p95_bbox_area_change",
    "valid_frame_fraction",
    "brightness_std",
)
AGREEMENT_FEATURES = (
    "pos_chrom_abs_diff_bpm",
    "regional_abs_diff_from_median_bpm",
)
TARGET_COLUMNS = (
    "unreliable_3bpm",
    "unreliable_5bpm",
    "unreliable_10bpm",
)
MODEL_INPUTS = {
    "M0": (),
    "M1": CONTEXT_FEATURES,
    "M2": (*CONTEXT_FEATURES, *QUALITY_FEATURES),
    "M3": (*CONTEXT_FEATURES, *AGREEMENT_FEATURES),
    "M4": (*CONTEXT_FEATURES, *QUALITY_FEATURES, *AGREEMENT_FEATURES),
}
FORMAL_FEATURE_COLUMNS = (
    *CANDIDATE_KEY,
    *QUALITY_FEATURES,
    *AGREEMENT_FEATURES,
    *TARGET_COLUMNS,
)

_FIELD_METADATA = {
    "subject": (
        "group",
        "identifier",
        "Original UBFC subject identifier used only for grouped validation.",
        "Not permitted.",
        "dataset",
    ),
    "window_id": (
        "key",
        "index",
        "Reference-window identifier within one subject.",
        "Not permitted.",
        "phase1",
    ),
    "roi": (
        "context_feature",
        "category",
        "Facial signal region used for the candidate estimate.",
        "Not permitted.",
        "phase2",
    ),
    "method": (
        "context_feature",
        "category",
        "Classical rPPG method used for the candidate estimate.",
        "Not permitted.",
        "phase2",
    ),
    "spectral_peak_power_fraction": (
        "quality_feature",
        "fraction",
        "Dominant-peak power divided by total in-band spectral power.",
        "Spectral estimate was unavailable.",
        "phase2",
    ),
    "spectral_entropy": (
        "quality_feature",
        "unitless",
        "Normalized entropy of the in-band rPPG spectrum.",
        "Spectral estimate was unavailable.",
        "phase2",
    ),
    "p95_bbox_center_jump": (
        "quality_feature",
        "frame fraction",
        "95th percentile absolute frame-to-frame face-centre displacement.",
        "Source motion measurements were unavailable.",
        "phase2",
    ),
    "p95_bbox_area_change": (
        "quality_feature",
        "fraction",
        "95th percentile absolute frame-to-frame face-area change.",
        "Source motion measurements were unavailable.",
        "phase2",
    ),
    "valid_frame_fraction": (
        "quality_feature",
        "fraction",
        "Fraction of window frames with finite RGB and a valid ROI.",
        "No source frames were available.",
        "phase2",
    ),
    "brightness_std": (
        "quality_feature",
        "pixel intensity",
        "Standard deviation of mean ROI brightness within the window.",
        "Brightness measurements were unavailable.",
        "phase2",
    ),
    "pos_chrom_abs_diff_bpm": (
        "agreement_feature",
        "bpm",
        "Absolute POS-versus-CHROM HR difference for the same ROI and window.",
        "Either method lacked an estimable HR.",
        "phase2",
    ),
    "regional_abs_diff_from_median_bpm": (
        "agreement_feature",
        "bpm",
        "Absolute deviation from the same-method regional median HR.",
        "The candidate or regional median HR was unavailable.",
        "phase2",
    ),
    "unreliable_3bpm": (
        "target",
        "boolean",
        "True when absolute candidate error is strictly greater than 3 bpm.",
        "Candidate was not eligible for finite-error modelling.",
        "phase3",
    ),
    "unreliable_5bpm": (
        "target",
        "boolean",
        "True when absolute candidate error is strictly greater than 5 bpm.",
        "Candidate was not eligible for finite-error modelling.",
        "phase3",
    ),
    "unreliable_10bpm": (
        "target",
        "boolean",
        "True when absolute candidate error is strictly greater than 10 bpm.",
        "Candidate was not eligible for finite-error modelling.",
        "phase3",
    ),
}


def build_reliability_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep eligible rows and only the fields allowed in Phase 4."""
    required = {"eligible_model", *FORMAL_FEATURE_COLUMNS}
    _require_columns(candidates, required, "Candidate table")
    if candidates.duplicated(list(CANDIDATE_KEY)).any():
        raise ValueError("Duplicate reliability feature keys")

    eligible = candidates["eligible_model"].eq(True)  # noqa: E712
    features = candidates.loc[eligible, list(FORMAL_FEATURE_COLUMNS)].copy()
    if features.loc[:, list(CANDIDATE_KEY)].isna().any().any():
        raise ValueError("Reliability feature keys must not be missing")
    for target in TARGET_COLUMNS:
        features[target] = features[target].astype("boolean")
        if features[target].isna().any():
            raise ValueError(f"Eligible reliability rows contain missing {target}")
    features = features.sort_values(list(CANDIDATE_KEY)).reset_index(drop=True)
    return features


def select_model_input(features: pd.DataFrame, model: str) -> pd.DataFrame:
    """Select the frozen input fields for one reliability model."""
    expected = _model_columns(model)
    _require_columns(features, set(expected), "Reliability feature table")
    selected = features.loc[:, list(expected)].copy()
    validate_model_input(selected, model)
    return selected


def validate_model_input(features: pd.DataFrame, model: str) -> None:
    """Reject model input that differs from the frozen allowlist."""
    expected = _model_columns(model)
    actual = tuple(features.columns)
    if actual != expected:
        raise ValueError(
            f"{model} input fields must exactly match {list(expected)}; "
            f"received {list(actual)}"
        )


def build_feature_dictionary(features: pd.DataFrame) -> pd.DataFrame:
    """Describe each field in the formal reliability table."""
    _validate_formal_columns(features)
    rows = []
    for field in features.columns:
        role, unit, definition, missing_meaning, source_stage = _FIELD_METADATA[field]
        rows.append(
            {
                "field": field,
                "role": role,
                "dtype": str(features[field].dtype),
                "unit": unit,
                "definition": definition,
                "missing_meaning": missing_meaning,
                "source_stage": source_stage,
            }
        )
    return pd.DataFrame(rows)


def build_feature_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and variation without altering values."""
    dictionary = build_feature_dictionary(features).set_index("field")
    rows = []
    for field in features.columns:
        series = features[field]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        numeric = pd.api.types.is_numeric_dtype(series.dtype)
        finite_values = np.array([], dtype=float)
        nonfinite_count = 0
        if numeric:
            values = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype=float,
                na_value=np.nan,
            )
            nonfinite_count = int(np.isinf(values).sum())
            finite_values = values[np.isfinite(values)]
        quantiles = (
            np.quantile(finite_values, [0.0, 0.25, 0.5, 0.75, 1.0])
            if finite_values.size
            else [np.nan] * 5
        )
        rows.append(
            {
                "field": field,
                "role": dictionary.loc[field, "role"],
                "missing_count": missing_count,
                "missing_fraction": missing_count / len(features)
                if len(features)
                else np.nan,
                "nonfinite_count": nonfinite_count,
                "unique_nonmissing_count": unique_count,
                "is_constant": unique_count <= 1,
                "minimum": quantiles[0],
                "p25": quantiles[1],
                "median": quantiles[2],
                "p75": quantiles[3],
                "maximum": quantiles[4],
                "model_exclusion_reason": _model_exclusion_reason(
                    str(dictionary.loc[field, "role"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _model_columns(model: str) -> tuple[str, ...]:
    try:
        return MODEL_INPUTS[model]
    except KeyError as error:
        raise ValueError(f"Unknown reliability model: {model}") from error


def _validate_formal_columns(features: pd.DataFrame) -> None:
    actual = tuple(features.columns)
    if actual != FORMAL_FEATURE_COLUMNS:
        raise ValueError(
            "Reliability feature columns must match the frozen Phase 4 schema"
        )


def _model_exclusion_reason(role: str) -> str:
    if role == "group":
        return "Used only for subject-level grouping"
    if role == "key":
        return "Used only for row alignment and audit"
    if role == "target":
        return "Prediction target; forbidden from model input"
    return ""


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
