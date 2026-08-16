from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_bootstrap_pairs_errors_but_keeps_all_valid_slots_for_availability() -> None:
    from rppg_pipeline.statistics import build_degradation_bootstrap

    candidates = pd.DataFrame(
        [
            _candidate("subject1", "original", "ok", 1.0),
            _candidate("subject1", "fps15", "ok", 3.0),
            _candidate("subject2", "original", "ok", 100.0),
            _candidate("subject2", "fps15", "low_video_coverage", np.nan),
        ]
    )

    result = build_degradation_bootstrap(candidates, n_bootstrap=200, seed=7)

    assert len(result) == 5
    mae = result[result["metric"].eq("mae_bpm")].iloc[0]
    assert mae["n_observations"] == 1
    assert mae["original_value"] == pytest.approx(1.0)
    assert mae["degraded_value"] == pytest.approx(3.0)
    assert mae["delta"] == pytest.approx(2.0)
    assert mae["ci_lower"] == pytest.approx(2.0)
    assert mae["ci_upper"] == pytest.approx(2.0)

    availability = result[result["metric"].eq("availability_rate")].iloc[0]
    assert availability["n_observations"] == 2
    assert availability["original_value"] == pytest.approx(1.0)
    assert availability["degraded_value"] == pytest.approx(0.5)
    assert availability["delta"] == pytest.approx(-0.5)


def test_run_statistics_reads_once_and_writes_two_formal_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rppg_pipeline import statistics

    candidate_path = tmp_path / "degradation_candidates.csv"
    _formal_candidates().to_csv(candidate_path, index=False)
    output = tmp_path / "formal"
    real_read_csv = pd.read_csv
    read_paths: list[Path] = []

    def read_csv(path, *args, **kwargs):
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(statistics.pd, "read_csv", read_csv)

    summary = statistics.run_degradation_statistics(
        candidate_path,
        output,
        n_bootstrap=20,
        seed=7,
    )

    assert read_paths == [candidate_path]
    assert summary == {"metric_count": 30, "bootstrap_count": 120}
    assert sorted(path.name for path in output.iterdir()) == [
        "degradation_bootstrap.csv",
        "degradation_metrics.csv",
    ]
    assert len(real_read_csv(output / "degradation_metrics.csv")) == 30
    assert len(real_read_csv(output / "degradation_bootstrap.csv")) == 120


def test_bootstrap_rejects_a_missing_paired_candidate_key() -> None:
    from rppg_pipeline.statistics import build_degradation_bootstrap

    candidates = pd.DataFrame(
        [
            _candidate("subject1", "original", "ok", 1.0),
            _candidate("subject1", "fps15", "ok", 3.0),
            _candidate("subject2", "original", "ok", 2.0),
        ]
    )

    with pytest.raises(ValueError, match="paired candidate keys differ"):
        build_degradation_bootstrap(candidates, n_bootstrap=20, seed=7)


def test_bootstrap_rejects_reference_validity_disagreement() -> None:
    from rppg_pipeline.statistics import build_degradation_bootstrap

    candidates = pd.DataFrame(
        [
            _candidate("subject1", "original", "ok", 1.0),
            _candidate("subject1", "fps15", "ok", 3.0),
        ]
    )
    candidates.loc[candidates["condition"].eq("fps15"), "reference_valid"] = False

    with pytest.raises(ValueError, match="reference validity differs"):
        build_degradation_bootstrap(candidates, n_bootstrap=20, seed=7)


def test_bootstrap_rejects_duplicate_candidate_keys() -> None:
    from rppg_pipeline.statistics import build_degradation_bootstrap

    candidates = pd.DataFrame(
        [
            _candidate("subject1", "original", "ok", 1.0),
            _candidate("subject1", "fps15", "ok", 3.0),
            _candidate("subject1", "fps15", "ok", 3.0),
        ]
    )

    with pytest.raises(ValueError, match="duplicate candidate keys"):
        build_degradation_bootstrap(candidates, n_bootstrap=20, seed=7)


def test_run_statistics_rejects_an_incomplete_formal_condition(
    tmp_path: Path,
) -> None:
    from rppg_pipeline.statistics import run_degradation_statistics

    candidates = _formal_candidates()
    candidates = candidates[candidates["condition"].ne("fps15")]
    candidate_path = tmp_path / "degradation_candidates.csv"
    candidates.to_csv(candidate_path, index=False)
    output = tmp_path / "formal"

    with pytest.raises(ValueError, match="expected 30 metric rows"):
        run_degradation_statistics(candidate_path, output, n_bootstrap=20, seed=7)

    assert not list(output.glob("*.csv"))


def _candidate(
    subject: str,
    condition: str,
    status: str,
    signed_error_bpm: float,
    roi: str = "full_face_inner",
    method: str = "POS",
) -> dict[str, object]:
    reference_hr = 70.0
    available = status == "ok" and np.isfinite(signed_error_bpm)
    return {
        "subject": subject,
        "condition": condition,
        "window_id": 1,
        "roi": roi,
        "method": method,
        "reference_hr_bpm": reference_hr,
        "reference_valid": True,
        "window_status": status,
        "rppg_hr_bpm": reference_hr + signed_error_bpm if available else np.nan,
        "signed_error_bpm": signed_error_bpm,
        "absolute_error_bpm": abs(signed_error_bpm),
    }


def _formal_candidates() -> pd.DataFrame:
    rows = []
    conditions = ("original", "fps15", "fps10", "roi_shift_3", "roi_shift_5")
    rois = ("full_face_inner", "forehead", "cheeks_mean")
    methods = ("POS", "CHROM")
    for subject_index, subject in enumerate(("subject1", "subject2"), start=1):
        for condition_index, condition in enumerate(conditions):
            for roi in rois:
                for method in methods:
                    rows.append(
                        _candidate(
                            subject,
                            condition,
                            "ok",
                            float(subject_index + condition_index),
                            roi,
                            method,
                        )
                    )
    return pd.DataFrame(rows)
