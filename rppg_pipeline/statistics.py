# Degradation statistics

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rppg_pipeline.degradation import FORMAL_CONDITIONS, ROIS
from rppg_pipeline.experiment import build_metrics
from rppg_pipeline.rppg import METHODS

DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260815
PAIR_KEY = ["subject", "window_id", "roi", "method"]
PAIR_VALUES = [
    "reference_valid",
    "window_status",
    "rppg_hr_bpm",
    "signed_error_bpm",
]
BOOTSTRAP_METRICS = (
    "mae_bpm",
    "rmse_bpm",
    "bias_bpm",
    "within_5_bpm_rate",
    "availability_rate",
)


# Build paired bootstrap comparisons
def build_degradation_bootstrap(
    candidates: pd.DataFrame,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    subjects = sorted(candidates["subject"].unique())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(subjects),
        np.full(len(subjects), 1.0 / len(subjects)),
        size=n_bootstrap,
    )
    present_conditions = set(candidates["condition"])
    present_groups = set(
        candidates.loc[candidates["condition"].eq("original"), ["roi", "method"]]
        .itertuples(index=False, name=None)
    )
    rows = []
    for condition in FORMAL_CONDITIONS:
        if condition == "original" or condition not in present_conditions:
            continue
        for roi in ROIS:
            for method in METHODS:
                if (roi, method) not in present_groups:
                    continue
                paired = _pair_candidates(candidates, condition, roi, method)
                subject_stats = _subject_statistics(paired, subjects)
                rows.extend(
                    _bootstrap_rows(
                        condition,
                        roi,
                        method,
                        subject_stats,
                        weights,
                        n_bootstrap,
                        seed,
                    )
                )
    return pd.DataFrame(rows)


# Pair one degradation with original
def _pair_candidates(
    candidates: pd.DataFrame,
    condition: str,
    roi: str,
    method: str,
) -> pd.DataFrame:
    group = candidates[
        candidates["condition"].isin(("original", condition))
        & candidates["roi"].eq(roi)
        & candidates["method"].eq(method)
    ]
    original = group[group["condition"].eq("original")][PAIR_KEY + PAIR_VALUES]
    degraded = group[group["condition"].eq(condition)][PAIR_KEY + PAIR_VALUES]
    if original.duplicated(PAIR_KEY).any() or degraded.duplicated(PAIR_KEY).any():
        raise ValueError(
            f"duplicate candidate keys for original versus {condition}, {roi}, {method}"
        )
    original_keys = set(original[PAIR_KEY].itertuples(index=False, name=None))
    degraded_keys = set(degraded[PAIR_KEY].itertuples(index=False, name=None))
    if original_keys != degraded_keys:
        raise ValueError(
            f"paired candidate keys differ for original versus {condition}, {roi}, "
            f"{method}: original={len(original_keys)}, degraded={len(degraded_keys)}"
        )
    paired = original.merge(
        degraded,
        on=PAIR_KEY,
        suffixes=("_original", "_degraded"),
        validate="one_to_one",
    )
    validity_matches = paired["reference_valid_original"].astype(bool).eq(
        paired["reference_valid_degraded"].astype(bool)
    )
    if not validity_matches.all():
        raise ValueError(
            f"reference validity differs for original versus {condition}, {roi}, "
            f"{method}"
        )
    return paired


# Reduce paired rows by subject
def _subject_statistics(
    paired: pd.DataFrame,
    subjects: list[str],
) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        group = paired[paired["subject"].eq(subject)]
        reference_valid = (
            group["reference_valid_original"].astype(bool)
            & group["reference_valid_degraded"].astype(bool)
        )
        original_available = group["window_status_original"].eq("ok") & np.isfinite(
            group["rppg_hr_bpm_original"]
        )
        degraded_available = group["window_status_degraded"].eq("ok") & np.isfinite(
            group["rppg_hr_bpm_degraded"]
        )
        paired_available = reference_valid & original_available & degraded_available
        original_error = group.loc[
            paired_available, "signed_error_bpm_original"
        ].to_numpy(dtype=float)
        degraded_error = group.loc[
            paired_available, "signed_error_bpm_degraded"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "subject": subject,
                "paired_n": len(original_error),
                "original_abs_sum": np.abs(original_error).sum(),
                "degraded_abs_sum": np.abs(degraded_error).sum(),
                "original_square_sum": np.square(original_error).sum(),
                "degraded_square_sum": np.square(degraded_error).sum(),
                "original_error_sum": original_error.sum(),
                "degraded_error_sum": degraded_error.sum(),
                "original_within_5": np.count_nonzero(
                    np.abs(original_error) <= 5.0
                ),
                "degraded_within_5": np.count_nonzero(
                    np.abs(degraded_error) <= 5.0
                ),
                "reference_n": np.count_nonzero(reference_valid),
                "original_available": np.count_nonzero(
                    reference_valid & original_available
                ),
                "degraded_available": np.count_nonzero(
                    reference_valid & degraded_available
                ),
            }
        )
    return pd.DataFrame(rows)


# Build five metric rows
def _bootstrap_rows(
    condition: str,
    roi: str,
    method: str,
    subject_stats: pd.DataFrame,
    weights: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    definitions = {
        "mae_bpm": ("original_abs_sum", "degraded_abs_sum", "paired_n", False),
        "rmse_bpm": (
            "original_square_sum",
            "degraded_square_sum",
            "paired_n",
            True,
        ),
        "bias_bpm": ("original_error_sum", "degraded_error_sum", "paired_n", False),
        "within_5_bpm_rate": (
            "original_within_5",
            "degraded_within_5",
            "paired_n",
            False,
        ),
        "availability_rate": (
            "original_available",
            "degraded_available",
            "reference_n",
            False,
        ),
    }
    rows = []
    for metric in BOOTSTRAP_METRICS:
        original_column, degraded_column, denominator_column, square_root = definitions[
            metric
        ]
        denominator = subject_stats[denominator_column].to_numpy(dtype=float)
        original = subject_stats[original_column].to_numpy(dtype=float)
        degraded = subject_stats[degraded_column].to_numpy(dtype=float)
        original_value = _observed_ratio(original, denominator, square_root)
        degraded_value = _observed_ratio(degraded, denominator, square_root)
        original_bootstrap = _weighted_ratio(
            weights, original, denominator, square_root
        )
        degraded_bootstrap = _weighted_ratio(
            weights, degraded, denominator, square_root
        )
        differences = degraded_bootstrap - original_bootstrap
        finite = differences[np.isfinite(differences)]
        ci_lower, ci_upper = (
            np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        )
        rows.append(
            {
                "condition": condition,
                "roi": roi,
                "method": method,
                "metric": metric,
                "n_subjects": len(subject_stats),
                "n_observations": int(denominator.sum()),
                "original_value": original_value,
                "degraded_value": degraded_value,
                "delta": degraded_value - original_value,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": seed,
            }
        )
    return rows


# Calculate one observed ratio
def _observed_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    square_root: bool,
) -> float:
    total = float(denominator.sum())
    if not total:
        return np.nan
    value = float(numerator.sum() / total)
    return float(np.sqrt(value)) if square_root else value


# Calculate bootstrap ratios
def _weighted_ratio(
    weights: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    square_root: bool,
) -> np.ndarray:
    totals = weights @ denominator
    values = np.divide(
        weights @ numerator,
        totals,
        out=np.full(len(weights), np.nan),
        where=totals > 0,
    )
    return np.sqrt(values) if square_root else values


# Run formal degradation statistics
def run_degradation_statistics(
    candidates_path: str | Path,
    output: str | Path,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, int]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(candidates_path)
    metrics = build_metrics(candidates)
    expected_metric_count = len(FORMAL_CONDITIONS) * len(ROIS) * len(METHODS)
    if len(metrics) != expected_metric_count:
        raise ValueError(
            f"expected {expected_metric_count} metric rows, found {len(metrics)}"
        )
    bootstrap = build_degradation_bootstrap(candidates, n_bootstrap, seed)
    expected_bootstrap_count = (
        (len(FORMAL_CONDITIONS) - 1)
        * len(ROIS)
        * len(METHODS)
        * len(BOOTSTRAP_METRICS)
    )
    if len(bootstrap) != expected_bootstrap_count:
        raise ValueError(
            f"expected {expected_bootstrap_count} bootstrap rows, "
            f"found {len(bootstrap)}"
        )
    metrics.to_csv(output_path / "degradation_metrics.csv", index=False)
    bootstrap.to_csv(output_path / "degradation_bootstrap.csv", index=False)
    return {"metric_count": len(metrics), "bootstrap_count": len(bootstrap)}


# Parse statistics arguments
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    summary = run_degradation_statistics(
        args.candidates,
        args.output,
        args.bootstrap_replicates,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
