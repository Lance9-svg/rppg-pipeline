from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rppg_pipeline.reliability_features import (
    build_feature_audit,
    build_feature_dictionary,
    build_reliability_features,
    select_model_input,
    validate_model_input,
)


def test_build_reliability_features_keeps_eligible_rows_without_leakage() -> None:
    candidates = _candidate_rows()

    features = build_reliability_features(candidates)

    assert list(features[["subject", "window_id"]].itertuples(False, None)) == [
        ("subject1", 1),
        ("subject1", 2),
    ]
    assert features["brightness_std"].isna().sum() == 1
    assert {
        "reference_hr_bpm",
        "rppg_hr_bpm",
        "signed_error_bpm",
        "absolute_error_bpm",
        "start_time_sec",
        "eligible_model",
    }.isdisjoint(features.columns)
    assert features["unreliable_5bpm"].dtype == "boolean"


def test_build_reliability_features_rejects_duplicate_candidate_key() -> None:
    candidates = _candidate_rows()
    candidates.loc[1, ["window_id", "roi", "method"]] = [1, "forehead", "POS"]

    with pytest.raises(ValueError, match="Duplicate reliability feature keys"):
        build_reliability_features(candidates)


def test_model_inputs_exactly_match_predeclared_features() -> None:
    features = build_reliability_features(_candidate_rows())
    expected = {
        "M0": [],
        "M1": ["method", "roi"],
        "M2": [
            "method",
            "roi",
            "spectral_peak_power_fraction",
            "spectral_entropy",
            "p95_bbox_center_jump",
            "p95_bbox_area_change",
            "valid_frame_fraction",
            "brightness_std",
        ],
        "M3": [
            "method",
            "roi",
            "pos_chrom_abs_diff_bpm",
            "regional_abs_diff_from_median_bpm",
        ],
        "M4": [
            "method",
            "roi",
            "spectral_peak_power_fraction",
            "spectral_entropy",
            "p95_bbox_center_jump",
            "p95_bbox_area_change",
            "valid_frame_fraction",
            "brightness_std",
            "pos_chrom_abs_diff_bpm",
            "regional_abs_diff_from_median_bpm",
        ],
    }

    for model, expected_columns in expected.items():
        selected = select_model_input(features, model)
        assert selected.columns.tolist() == expected_columns


def test_validate_model_input_rejects_extra_or_reordered_fields() -> None:
    features = build_reliability_features(_candidate_rows())
    valid = select_model_input(features, "M4")
    with_leakage = valid.assign(reference_hr_bpm=70.0)

    with pytest.raises(ValueError, match="must exactly match"):
        validate_model_input(with_leakage, "M4")
    with pytest.raises(ValueError, match="must exactly match"):
        validate_model_input(valid.loc[:, list(reversed(valid.columns))], "M4")


def test_feature_dictionary_assigns_one_role_to_every_formal_field() -> None:
    features = build_reliability_features(_candidate_rows())

    dictionary = build_feature_dictionary(features)

    assert dictionary["field"].tolist() == features.columns.tolist()
    assert not dictionary["field"].duplicated().any()
    roles = dictionary.set_index("field")["role"]
    assert roles["subject"] == "group"
    assert roles["window_id"] == "key"
    assert roles["method"] == "context_feature"
    assert roles["spectral_entropy"] == "quality_feature"
    assert roles["pos_chrom_abs_diff_bpm"] == "agreement_feature"
    assert roles["unreliable_5bpm"] == "target"
    assert dictionary["definition"].str.strip().ne("").all()


def test_feature_audit_reports_missing_and_constant_values_without_imputation() -> None:
    features = build_reliability_features(_candidate_rows())

    audit = build_feature_audit(features).set_index("field")

    assert audit.loc["brightness_std", "missing_count"] == 1
    assert audit.loc["brightness_std", "missing_fraction"] == 0.5
    assert bool(audit.loc["valid_frame_fraction", "is_constant"])
    assert not bool(audit.loc["spectral_entropy", "is_constant"])
    assert features["brightness_std"].isna().sum() == 1


def _candidate_rows() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject": ["subject1", "subject1", "subject2"],
            "window_id": [1, 2, 1],
            "roi": ["forehead", "left_cheek", "right_cheek"],
            "method": ["POS", "CHROM", "POS"],
            "eligible_model": [True, True, False],
            "spectral_peak_power_fraction": [0.7, 0.6, 0.5],
            "spectral_entropy": [0.1, 0.2, 0.3],
            "p95_bbox_center_jump": [0.01, 0.02, 0.03],
            "p95_bbox_area_change": [0.02, 0.03, 0.04],
            "valid_frame_fraction": [1.0, 1.0, 0.9],
            "brightness_std": [np.nan, 2.0, 3.0],
            "pos_chrom_abs_diff_bpm": [1.0, 2.0, 3.0],
            "regional_abs_diff_from_median_bpm": [0.5, 1.5, 2.5],
            "unreliable_3bpm": pd.Series([False, True, pd.NA], dtype="boolean"),
            "unreliable_5bpm": pd.Series([False, True, pd.NA], dtype="boolean"),
            "unreliable_10bpm": pd.Series([False, False, pd.NA], dtype="boolean"),
            "reference_hr_bpm": [70.0, 72.0, 74.0],
            "rppg_hr_bpm": [72.0, 79.0, np.nan],
            "signed_error_bpm": [2.0, 7.0, np.nan],
            "absolute_error_bpm": [2.0, 7.0, np.nan],
            "start_time_sec": [0.0, 1.0, 0.0],
        }
    )
    return frame
