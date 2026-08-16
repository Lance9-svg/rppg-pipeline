# Interpretable reliability model

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from rppg_pipeline.degradation import ROIS
from rppg_pipeline.rppg import METHODS

QUALITY_FEATURES = (
    "valid_frame_fraction",
    "max_missing_gap_sec",
    "mean_brightness",
    "brightness_std",
    "mean_overexposure_ratio",
    "spectral_peak_fraction",
    "spectral_entropy",
    "pos_chrom_diff_bpm",
)
CANDIDATE_KEY = ["subject", "condition", "window_id", "roi", "method"]
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
FOLD_COUNT = 5
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260815


# Build OOF predictions and metrics
def build_reliability_outputs(
    candidates: pd.DataFrame,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    eligible = candidates[
        candidates["reference_valid"].astype(bool)
        & candidates["window_status"].eq("ok")
        & np.isfinite(candidates["rppg_hr_bpm"])
        & np.isfinite(candidates["absolute_error_bpm"])
    ].copy()
    eligible.reset_index(drop=True, inplace=True)
    eligible["unreliable"] = eligible["absolute_error_bpm"].gt(5.0)
    fold_map = assign_subject_folds(eligible["subject"].unique())
    eligible["fold"] = eligible["subject"].map(fold_map)
    route = select_reliability_route(eligible)

    baseline_prediction = np.empty(len(eligible), dtype=float)
    quality_prediction = np.empty(len(eligible), dtype=float)
    coefficient_rows = []
    for fold in range(FOLD_COUNT):
        train = eligible[eligible["fold"].ne(fold)]
        test = eligible[eligible["fold"].eq(fold)]
        test_index = test.index.to_numpy(dtype=int)
        target_column = "unreliable" if route == "logistic" else "absolute_error_bpm"
        target = train[target_column].to_numpy(dtype=float)
        for model, include_quality, output in (
            ("baseline", False, baseline_prediction),
            ("quality", True, quality_prediction),
        ):
            train_design, feature_names, parameters = _fit_design(
                train, include_quality
            )
            test_design = _transform_design(test, parameters)
            if route == "logistic":
                intercept, coefficients = _fit_logistic(train_design, target)
                prediction = expit(intercept + test_design @ coefficients)
            else:
                intercept, coefficients = _fit_ridge(train_design, target)
                prediction = intercept + test_design @ coefficients
            output[test_index] = prediction
            for feature, value in zip(
                ("intercept", *feature_names),
                (intercept, *coefficients),
                strict=True,
            ):
                coefficient_rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "feature": feature,
                        "coefficient": float(value),
                    }
                )

    predictions = eligible[CANDIDATE_KEY].copy()
    predictions["fold"] = eligible["fold"].to_numpy(dtype=int)
    if route == "logistic":
        predictions["unreliable"] = eligible["unreliable"].to_numpy(dtype=bool)
        predictions["baseline_risk"] = baseline_prediction
        predictions["quality_risk"] = quality_prediction
    else:
        predictions["absolute_error_bpm"] = eligible[
            "absolute_error_bpm"
        ].to_numpy(dtype=float)
        predictions["baseline_predicted_absolute_error_bpm"] = baseline_prediction
        predictions["quality_predicted_absolute_error_bpm"] = quality_prediction
    metrics = _build_reliability_metrics(
        predictions,
        pd.DataFrame(coefficient_rows),
        n_bootstrap,
        seed,
        route,
    )
    return {"route": route, "predictions": predictions, "metrics": metrics}


# Assign fixed subject folds
def assign_subject_folds(subjects) -> dict[str, int]:
    ordered = sorted((str(subject) for subject in subjects), key=_subject_sort_key)
    return {subject: index % FOLD_COUNT for index, subject in enumerate(ordered)}


# Choose one formal route
def select_reliability_route(eligible: pd.DataFrame) -> str:
    unreliable = eligible["unreliable"].astype(bool)
    enough_events = int(unreliable.sum()) >= 100
    enough_subjects = eligible.loc[unreliable, "subject"].nunique() >= 10
    both_classes = eligible.groupby("fold")["unreliable"].agg(
        lambda values: values.any() and (~values.astype(bool)).any()
    )
    return (
        "logistic"
        if enough_events and enough_subjects and both_classes.all()
        else "ridge"
    )


# Sort numeric subject identifiers
def _subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", subject)
    return (int(match.group(1)), subject) if match else (2**31 - 1, subject)


# Impute missing quality features
def _impute_quality(numeric: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(numeric), numeric, medians)


# Fit fold-local feature processing
def _fit_design(
    train: pd.DataFrame,
    include_quality: bool,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    observed_rois = set(train["roi"])
    observed_methods = set(train["method"])
    roi_categories = [roi for roi in ROIS if roi in observed_rois]
    method_categories = [method for method in METHODS if method in observed_methods]
    parameters: dict[str, object] = {
        "roi_categories": roi_categories,
        "method_categories": method_categories,
        "include_quality": include_quality,
    }
    # reference levels: full_face_inner and POS
    feature_names = [f"roi_{roi}" for roi in roi_categories[1:]]
    feature_names.extend(f"method_{method}" for method in method_categories[1:])
    columns = _categorical_columns(train, roi_categories, method_categories)
    if include_quality:
        numeric = train[list(QUALITY_FEATURES)].to_numpy(dtype=float)
        medians = np.nanmedian(numeric, axis=0)
        imputed = _impute_quality(numeric, medians)
        # numeric coefficients are per fold-local 1-SD change
        means = np.mean(imputed, axis=0)
        scales = np.std(imputed, axis=0)
        # zero-variance features keep scale 1.0 and a zero coefficient
        scales[scales <= 1e-12] = 1.0
        columns.extend(((imputed - means) / scales).T)
        feature_names.extend(QUALITY_FEATURES)
        parameters.update(
            {
                "medians": medians,
                "means": means,
                "scales": scales,
            }
        )
    design = np.column_stack(columns) if columns else np.empty((len(train), 0))
    return design, feature_names, parameters


# Apply fold-local feature processing
def _transform_design(
    frame: pd.DataFrame,
    parameters: dict[str, object],
) -> np.ndarray:
    roi_categories = parameters["roi_categories"]
    method_categories = parameters["method_categories"]
    columns = _categorical_columns(frame, roi_categories, method_categories)
    if parameters["include_quality"]:
        numeric = frame[list(QUALITY_FEATURES)].to_numpy(dtype=float)
        medians = parameters["medians"]
        imputed = _impute_quality(numeric, medians)
        columns.extend(
            ((imputed - parameters["means"]) / parameters["scales"]).T
        )
    return np.column_stack(columns) if columns else np.empty((len(frame), 0))


# Encode ROI and method
def _categorical_columns(
    frame: pd.DataFrame,
    roi_categories: list[str],
    method_categories: list[str],
) -> list[np.ndarray]:
    columns = [frame["roi"].eq(roi).to_numpy(dtype=float) for roi in roi_categories[1:]]
    columns.extend(
        frame["method"].eq(method).to_numpy(dtype=float)
        for method in method_categories[1:]
    )
    return columns


# Fit one L2 logistic regression
def _fit_logistic(design: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    prevalence = float(np.clip(np.mean(labels), 1e-6, 1.0 - 1e-6))
    initial = np.zeros(design.shape[1] + 1, dtype=float)
    initial[0] = np.log(prevalence / (1.0 - prevalence))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        linear = intercept + design @ coefficients
        residual = expit(linear) - labels
        loss = np.logaddexp(0.0, linear).sum() - labels @ linear
        loss += 0.5 * coefficients @ coefficients
        gradient = np.concatenate(
            ([residual.sum()], design.T @ residual + coefficients)
        )
        return float(loss), gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"logistic regression did not converge: {result.message}")
    return float(result.x[0]), result.x[1:]


# Fit one L2 ridge regression
def _fit_ridge(design: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    augmented = np.column_stack((np.ones(len(design)), design))
    penalty = np.eye(augmented.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    system = augmented.T @ augmented + penalty
    response = augmented.T @ target
    try:
        parameters = np.linalg.solve(system, response)
    except np.linalg.LinAlgError:
        parameters = np.linalg.lstsq(system, response, rcond=None)[0]
    return float(parameters[0]), parameters[1:]


# Build performance and coefficient rows
def _build_reliability_metrics(
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    route: str,
) -> pd.DataFrame:
    performance = (
        _classification_performance_rows(predictions, n_bootstrap, seed)
        if route == "logistic"
        else _regression_performance_rows(predictions, n_bootstrap, seed)
    )
    coefficient_rows = []
    groups = coefficients.groupby(["model", "feature"], sort=False)
    for (model, feature), group in groups:
        values = group["coefficient"].to_numpy(dtype=float)
        coefficient_rows.append(
            {
                "record_type": "coefficient",
                "route": route,
                "model": model,
                "feature": feature,
                "coefficient_mean": float(np.mean(values)),
                "coefficient_std": float(np.std(values)),
                "coefficient_min": float(np.min(values)),
                "coefficient_max": float(np.max(values)),
                "positive_fold_count": int(np.count_nonzero(values > 0.0)),
                "fold_count": len(values),
            }
        )
    metrics = pd.concat(
        [pd.DataFrame(performance), pd.DataFrame(coefficient_rows)],
        ignore_index=True,
    )
    columns = (
        "record_type",
        "route",
        "model",
        "metric",
        "coverage",
        "feature",
        "baseline_value",
        "quality_value",
        "delta",
        "ci_lower",
        "ci_upper",
        "coefficient_mean",
        "coefficient_std",
        "coefficient_min",
        "coefficient_max",
        "positive_fold_count",
        "fold_count",
        "n_records",
        "n_subjects",
        "bootstrap_replicates",
        "bootstrap_seed",
    )
    return metrics.reindex(columns=columns)


# Build subject-level bootstrap weights
def _subject_bootstrap_weights(
    n_subjects: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n_subjects,
        np.full(n_subjects, 1.0 / n_subjects),
        size=n_bootstrap,
    )


# Build bootstrap difference rows for one route
def _bootstrap_performance_rows(
    predictions: pd.DataFrame,
    target: np.ndarray,
    baseline: np.ndarray,
    quality: np.ndarray,
    value_function,
    specifications: list[tuple[str, float]],
    route: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    baseline_layout = _risk_layout(baseline)
    quality_layout = _risk_layout(quality)
    unit_weights = np.ones(len(predictions), dtype=float)
    baseline_value = value_function(target, baseline, unit_weights, baseline_layout)
    quality_value = value_function(target, quality, unit_weights, quality_layout)

    subjects = sorted(predictions["subject"].unique(), key=_subject_sort_key)
    subject_index = predictions["subject"].map(
        {subject: index for index, subject in enumerate(subjects)}
    ).to_numpy(dtype=int)
    subject_weights = _subject_bootstrap_weights(
        len(subjects), n_bootstrap, seed
    )
    differences = np.empty((n_bootstrap, len(baseline_value)), dtype=float)
    for index, weights in enumerate(subject_weights):
        row_weights = weights[subject_index]
        baseline_sample = value_function(target, baseline, row_weights, baseline_layout)
        quality_sample = value_function(target, quality, row_weights, quality_layout)
        differences[index] = quality_sample - baseline_sample

    rows = []
    for index, (metric, coverage) in enumerate(specifications):
        finite = differences[:, index]
        finite = finite[np.isfinite(finite)]
        lower, upper = np.percentile(finite, [2.5, 97.5])
        rows.append(
            {
                "record_type": "performance",
                "route": route,
                "model": "quality_vs_baseline",
                "metric": metric,
                "coverage": coverage,
                "baseline_value": baseline_value[index],
                "quality_value": quality_value[index],
                "delta": quality_value[index] - baseline_value[index],
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_records": len(predictions),
                "n_subjects": len(subjects),
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": seed,
            }
        )
    return rows


# Compare OOF classification performance
def _classification_performance_rows(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    labels = predictions["unreliable"].to_numpy(dtype=float)
    baseline = predictions["baseline_risk"].to_numpy(dtype=float)
    quality = predictions["quality_risk"].to_numpy(dtype=float)
    specifications = [("aucpr", np.nan), ("roc_auc", np.nan), ("brier_score", np.nan)]
    specifications.extend(("coverage_error", coverage) for coverage in COVERAGES)
    return _bootstrap_performance_rows(
        predictions,
        labels,
        baseline,
        quality,
        _classification_values,
        specifications,
        "logistic",
        n_bootstrap,
        seed,
    )


# Compare OOF regression performance
def _regression_performance_rows(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    target = predictions["absolute_error_bpm"].to_numpy(dtype=float)
    baseline = predictions[
        "baseline_predicted_absolute_error_bpm"
    ].to_numpy(dtype=float)
    quality = predictions[
        "quality_predicted_absolute_error_bpm"
    ].to_numpy(dtype=float)
    specifications = [("mae_bpm", np.nan), ("rmse_bpm", np.nan)]
    specifications.extend(("coverage_error", coverage) for coverage in COVERAGES)
    return _bootstrap_performance_rows(
        predictions,
        target,
        baseline,
        quality,
        _regression_values,
        specifications,
        "ridge",
        n_bootstrap,
        seed,
    )


# Precompute fixed risk ordering
def _risk_layout(risk: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    layouts = {}
    for name, order in (
        ("ascending", np.argsort(risk, kind="stable")),
        ("descending", np.argsort(-risk, kind="stable")),
    ):
        ordered_risk = risk[order]
        starts = np.r_[0, np.flatnonzero(ordered_risk[1:] != ordered_risk[:-1]) + 1]
        layouts[name] = (order, starts)
    return layouts


# Calculate classification metrics
def _classification_values(
    labels: np.ndarray,
    risk: np.ndarray,
    weights: np.ndarray,
    layout: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    total_weight = float(weights.sum())
    brier = float(np.sum(weights * np.square(risk - labels)) / total_weight)
    ascending_total, ascending_positive = _group_weights(
        labels, weights, layout["ascending"]
    )
    descending_total, descending_positive = _group_weights(
        labels, weights, layout["descending"]
    )
    roc_auc = _weighted_roc_auc(ascending_total, ascending_positive)
    aucpr = _weighted_average_precision(descending_total, descending_positive)
    coverage_error = [
        _weighted_coverage_error(
            ascending_total,
            ascending_positive,
            coverage,
        )
        for coverage in COVERAGES
    ]
    return np.asarray([aucpr, roc_auc, brier, *coverage_error], dtype=float)


# Calculate regression metrics
def _regression_values(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    layout: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    total_weight = float(weights.sum())
    residual = prediction - target
    mae = float(np.sum(weights * np.abs(residual)) / total_weight)
    rmse = float(np.sqrt(np.sum(weights * np.square(residual)) / total_weight))
    ascending_total, ascending_error = _group_weights(
        target, weights, layout["ascending"]
    )
    coverage_error = [
        _weighted_coverage_error(ascending_total, ascending_error, coverage)
        for coverage in COVERAGES
    ]
    return np.asarray([mae, rmse, *coverage_error], dtype=float)


# Aggregate weighted labels by tied risk
def _group_weights(
    labels: np.ndarray,
    weights: np.ndarray,
    layout: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    order, starts = layout
    ordered_weights = weights[order]
    total = np.add.reduceat(ordered_weights, starts)
    positive = np.add.reduceat(ordered_weights * labels[order], starts)
    keep = total > 0
    return total[keep], positive[keep]


# Calculate tie-aware ROC-AUC
def _weighted_roc_auc(total: np.ndarray, positive: np.ndarray) -> float:
    negative = total - positive
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    if not positive_total or not negative_total:
        return np.nan
    negative_below = np.cumsum(negative) - negative
    concordant = np.sum(positive * (negative_below + 0.5 * negative))
    return float(concordant / (positive_total * negative_total))


# Calculate tie-aware average precision
def _weighted_average_precision(total: np.ndarray, positive: np.ndarray) -> float:
    positive_total = float(positive.sum())
    if not positive_total:
        return np.nan
    cumulative_total = np.cumsum(total)
    cumulative_positive = np.cumsum(positive)
    precision = cumulative_positive / cumulative_total
    return float(np.sum((positive / positive_total) * precision))


# Calculate error after risk-based retention
def _weighted_coverage_error(
    total: np.ndarray,
    positive: np.ndarray,
    coverage: float,
) -> float:
    retained = max(1, int(np.floor(total.sum() * coverage)))
    cumulative_total = np.cumsum(total)
    index = int(np.searchsorted(cumulative_total, retained, side="left"))
    total_before = cumulative_total[index] - total[index]
    positive_before = np.sum(positive[:index])
    partial = retained - total_before
    partial_positive = partial * positive[index] / total[index]
    return float((positive_before + partial_positive) / retained)


# Run formal reliability modeling
def run_reliability_model(
    candidates_path: str | Path,
    output: str | Path,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(candidates_path)
    outputs = build_reliability_outputs(candidates, n_bootstrap, seed)
    predictions = outputs["predictions"]
    metrics = outputs["metrics"]
    predictions.to_csv(output_path / "reliability_predictions.csv", index=False)
    metrics.to_csv(output_path / "reliability_metrics.csv", index=False)
    return {
        "route": outputs["route"],
        "prediction_count": len(predictions),
        "metric_count": len(metrics),
    }


# Parse reliability arguments
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
    summary = run_reliability_model(
        args.candidates,
        args.output,
        args.bootstrap_replicates,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
