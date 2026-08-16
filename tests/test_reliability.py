from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_quality_logistic_model_uses_subject_grouped_oof_signal() -> None:
    from rppg_pipeline.reliability import build_reliability_outputs

    outputs = build_reliability_outputs(
        _reliability_candidates("quality"),
        n_bootstrap=20,
        seed=7,
    )

    assert outputs["route"] == "logistic"
    predictions = outputs["predictions"]
    assert len(predictions) == 200
    assert predictions.groupby("subject")["fold"].nunique().eq(1).all()
    assert predictions["baseline_risk"].between(0.0, 1.0).all()
    assert predictions["quality_risk"].between(0.0, 1.0).all()

    metrics = outputs["metrics"]
    roc_auc = metrics[
        metrics["record_type"].eq("performance")
        & metrics["metric"].eq("roc_auc")
    ].iloc[0]
    assert roc_auc["baseline_value"] == pytest.approx(0.5)
    assert roc_auc["quality_value"] > 0.99
    entropy = metrics[
        metrics["record_type"].eq("coefficient")
        & metrics["model"].eq("quality")
        & metrics["feature"].eq("spectral_entropy")
    ].iloc[0]
    assert entropy["coefficient_mean"] > 0.0


def test_models_ignore_condition_and_forbidden_outcome_columns() -> None:
    from rppg_pipeline.reliability import build_reliability_outputs

    outputs = build_reliability_outputs(
        _reliability_candidates("condition"),
        n_bootstrap=20,
        seed=7,
    )

    roc_auc = outputs["metrics"]
    roc_auc = roc_auc[
        roc_auc["record_type"].eq("performance")
        & roc_auc["metric"].eq("roc_auc")
    ].iloc[0]
    assert roc_auc["baseline_value"] == pytest.approx(0.5)
    assert roc_auc["quality_value"] == pytest.approx(0.5)


def test_run_reliability_reads_once_and_writes_two_formal_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rppg_pipeline import reliability

    candidate_path = tmp_path / "degradation_candidates.csv"
    _reliability_candidates("quality").to_csv(candidate_path, index=False)
    output = tmp_path / "formal"
    real_read_csv = pd.read_csv
    read_paths: list[Path] = []

    def read_csv(path, *args, **kwargs):
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(reliability.pd, "read_csv", read_csv)

    summary = reliability.run_reliability_model(
        candidate_path,
        output,
        n_bootstrap=20,
        seed=7,
    )

    assert read_paths == [candidate_path]
    assert summary["route"] == "logistic"
    assert summary["prediction_count"] == 200
    assert sorted(path.name for path in output.iterdir()) == [
        "reliability_metrics.csv",
        "reliability_predictions.csv",
    ]
    predictions = real_read_csv(output / "reliability_predictions.csv")
    assert len(predictions) == 200
    assert not predictions.duplicated(
        ["subject", "condition", "window_id", "roi", "method"]
    ).any()


def test_route_gate_uses_all_three_logistic_requirements() -> None:
    from rppg_pipeline.reliability import select_reliability_route

    assert select_reliability_route(_gate_frame()) == "logistic"
    assert select_reliability_route(_gate_frame(event_count=99)) == "ridge"
    assert select_reliability_route(_gate_frame(affected_subjects=9)) == "ridge"
    assert select_reliability_route(_gate_frame(single_class_fold=0)) == "ridge"


def test_ridge_fallback_predicts_continuous_error_with_grouped_oof() -> None:
    from rppg_pipeline.reliability import build_reliability_outputs

    outputs = build_reliability_outputs(
        _ridge_candidates(),
        n_bootstrap=20,
        seed=7,
    )

    assert outputs["route"] == "ridge"
    predictions = outputs["predictions"]
    assert len(predictions) == 100
    assert predictions.groupby("subject")["fold"].nunique().eq(1).all()
    assert predictions[
        [
            "absolute_error_bpm",
            "baseline_predicted_absolute_error_bpm",
            "quality_predicted_absolute_error_bpm",
        ]
    ].notna().all().all()

    metrics = outputs["metrics"]
    mae = metrics[
        metrics["record_type"].eq("performance")
        & metrics["metric"].eq("mae_bpm")
    ].iloc[0]
    assert mae["route"] == "ridge"
    assert mae["quality_value"] < mae["baseline_value"]
    entropy = metrics[
        metrics["record_type"].eq("coefficient")
        & metrics["model"].eq("quality")
        & metrics["feature"].eq("spectral_entropy")
    ].iloc[0]
    assert entropy["coefficient_mean"] > 0.0


def _reliability_candidates(signal: str) -> pd.DataFrame:
    rows = []
    for subject_index in range(1, 11):
        for window_id in range(20):
            unreliable = window_id % 2 == 1
            condition = (
                "fps10"
                if signal == "condition" and unreliable
                else "original"
            )
            entropy = (
                0.9
                if signal == "quality" and unreliable
                else 0.1
            )
            absolute_error = 10.0 if unreliable else 1.0
            rows.append(
                {
                    "subject": f"subject{subject_index}",
                    "condition": condition,
                    "window_id": window_id,
                    "roi": "full_face_inner",
                    "method": "POS",
                    "reference_hr_bpm": 70.0,
                    "reference_valid": True,
                    "window_status": "ok",
                    "rppg_hr_bpm": 70.0 + absolute_error,
                    "signed_error_bpm": absolute_error,
                    "absolute_error_bpm": absolute_error,
                    "valid_frame_fraction": 1.0,
                    "max_missing_gap_sec": 0.04,
                    "mean_brightness": 120.0,
                    "brightness_std": 2.0,
                    "mean_overexposure_ratio": 0.0,
                    "spectral_peak_fraction": 0.5,
                    "spectral_entropy": entropy,
                    "pos_chrom_diff_bpm": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _gate_frame(
    event_count: int = 100,
    affected_subjects: int = 10,
    single_class_fold: int | None = None,
) -> pd.DataFrame:
    rows = []
    for subject_index in range(1, 11):
        for window_id in range(20):
            rows.append(
                {
                    "subject": f"subject{subject_index}",
                    "fold": (subject_index - 1) % 5,
                    "unreliable": False,
                    "window_id": window_id,
                }
            )
    frame = pd.DataFrame(rows)
    eligible = frame[
        frame["subject"].isin(
            [f"subject{index}" for index in range(1, affected_subjects + 1)]
        )
    ].sort_values(["window_id", "subject"])
    event_indices = eligible.index[:event_count]
    frame.loc[event_indices, "unreliable"] = True
    if single_class_fold is not None:
        frame.loc[frame["fold"].eq(single_class_fold), "unreliable"] = True
    return frame


def _ridge_candidates() -> pd.DataFrame:
    frame = _reliability_candidates("quality").iloc[::2].copy()
    sequence = frame["window_id"].to_numpy(dtype=float) / 20.0
    frame["spectral_entropy"] = sequence
    frame["absolute_error_bpm"] = 1.0 + 4.0 * sequence
    frame["signed_error_bpm"] = frame["absolute_error_bpm"]
    frame["rppg_hr_bpm"] = 70.0 + frame["absolute_error_bpm"]
    return frame.reset_index(drop=True)
