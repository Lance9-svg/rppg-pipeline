"""Formal window-aligned POS/CHROM processing for Phase 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import signal

DEFAULT_ROIS = (
    "full_face_inner",
    "forehead",
    "left_cheek",
    "right_cheek",
    "cheeks_mean",
)
DEFAULT_METHODS = ("CHROM", "POS")


@dataclass(frozen=True)
class StandardRPPGConfig:
    """Frozen processing choices for the formal Phase 2 experiment."""

    rois: tuple[str, ...] = DEFAULT_ROIS
    methods: tuple[str, ...] = DEFAULT_METHODS
    bandpass_low_hz: float = 0.7
    bandpass_high_hz: float = 4.0
    filter_order: int = 3
    internal_window_sec: float = 1.6
    chrom_overlap_fraction: float = 0.5
    nfft: int = 4096
    min_video_coverage_fraction: float = 0.95
    min_valid_frame_fraction: float = 0.80
    max_interpolation_gap_sec: float = 0.50

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UniformRGBTrace:
    """Uniformly sampled RGB trace for one facial region."""

    time_sec: np.ndarray
    rgb: np.ndarray
    sample_rate_hz: float


def prepare_uniform_rgb_trace(
    trace_df: pd.DataFrame,
    roi: str,
) -> UniformRGBTrace:
    """Interpolate the valid RGB samples onto the video's uniform time grid."""
    required = [
        "time_sec",
        f"{roi}_r_mean",
        f"{roi}_g_mean",
        f"{roi}_b_mean",
        f"{roi}_roi_valid",
    ]
    missing = [column for column in required if column not in trace_df.columns]
    if missing:
        raise ValueError(f"Missing RGB trace columns for {roi}: {missing}")

    time_sec = trace_df["time_sec"].to_numpy(dtype=float)
    finite_time = np.isfinite(time_sec)
    time_sec = time_sec[finite_time]
    if len(time_sec) < 3 or np.any(np.diff(time_sec) <= 0):
        raise ValueError("RGB timestamps must be finite and strictly increasing")

    sample_interval = float(np.median(np.diff(time_sec)))
    if not np.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError("Could not derive a positive RGB sampling interval")
    sample_rate_hz = 1.0 / sample_interval

    rgb = trace_df.loc[
        finite_time,
        [
            f"{roi}_r_mean",
            f"{roi}_g_mean",
            f"{roi}_b_mean",
        ],
    ].to_numpy(dtype=float)
    roi_valid = _as_bool_array(
        trace_df.loc[finite_time, f"{roi}_roi_valid"].to_numpy()
    )
    usable = roi_valid & np.isfinite(rgb).all(axis=1)
    if np.count_nonzero(usable) < 3:
        raise ValueError(f"Too few valid RGB samples for ROI: {roi}")

    n_uniform = int(np.floor((time_sec[-1] - time_sec[0]) * sample_rate_hz)) + 1
    uniform_time = time_sec[0] + np.arange(n_uniform) / sample_rate_hz
    uniform_rgb = np.column_stack(
        [
            np.interp(uniform_time, time_sec[usable], rgb[usable, channel])
            for channel in range(3)
        ]
    )
    return UniformRGBTrace(
        time_sec=uniform_time,
        rgb=uniform_rgb,
        sample_rate_hz=sample_rate_hz,
    )


def extract_standard_bvp(
    rgb: np.ndarray,
    sample_rate_hz: float,
    method: str,
    config: StandardRPPGConfig | None = None,
) -> np.ndarray:
    """Extract and commonly band-pass one formal POS or CHROM signal."""
    config = config or StandardRPPGConfig()
    method_upper = method.upper()
    if method_upper == "POS":
        raw_bvp = pos_overlap_add(
            rgb,
            sample_rate_hz,
            window_sec=config.internal_window_sec,
        )
    elif method_upper == "CHROM":
        raw_bvp = chrom_overlap_add(
            rgb,
            sample_rate_hz,
            config=config,
        )
    else:
        raise ValueError(f"Unsupported formal rPPG method: {method}")
    return preprocess_bvp(raw_bvp, sample_rate_hz, config)


def pos_overlap_add(
    rgb: np.ndarray,
    sample_rate_hz: float,
    window_sec: float = 1.6,
) -> np.ndarray:
    """Implement the POS temporal-normalization and overlap-add algorithm."""
    values = _validate_rgb(rgb)
    window_n = max(3, int(round(window_sec * sample_rate_hz)))
    if len(values) < window_n:
        raise ValueError("RGB trace is shorter than the POS internal window")

    pulse = np.zeros(len(values), dtype=float)
    projection = np.array(
        [
            [0.0, 1.0, -1.0],
            [-2.0, 1.0, 1.0],
        ]
    )
    for end in range(window_n - 1, len(values)):
        start = end - window_n + 1
        color_window = values[start : end + 1].T
        channel_means = np.mean(color_window, axis=1)
        normalized = color_window / channel_means[:, np.newaxis]
        projected = projection @ normalized
        alpha = _safe_std_ratio(projected[0], projected[1])
        candidate = projected[0] + alpha * projected[1]
        pulse[start : end + 1] += candidate - float(np.mean(candidate))
    return pulse


def chrom_overlap_add(
    rgb: np.ndarray,
    sample_rate_hz: float,
    config: StandardRPPGConfig | None = None,
) -> np.ndarray:
    """Implement CHROM using filtered projections and overlap-add."""
    config = config or StandardRPPGConfig()
    values = _validate_rgb(rgb)
    window_n = max(3, int(round(config.internal_window_sec * sample_rate_hz)))
    if len(values) < window_n:
        raise ValueError("RGB trace is shorter than the CHROM internal window")

    overlap = float(np.clip(config.chrom_overlap_fraction, 0.0, 0.99))
    step_n = max(1, int(round(window_n * (1.0 - overlap))))
    starts = list(range(0, len(values) - window_n + 1, step_n))
    final_start = len(values) - window_n
    if starts[-1] != final_start:
        starts.append(final_start)

    pulse = np.zeros(len(values), dtype=float)
    weights = np.zeros(len(values), dtype=float)
    taper = np.hanning(window_n + 2)[1:-1]
    for start in starts:
        end = start + window_n
        normalized = values[start:end] / np.mean(values[start:end], axis=0) - 1.0
        red = normalized[:, 0]
        green = normalized[:, 1]
        blue = normalized[:, 2]
        x_signal = 3.0 * red - 2.0 * green
        y_signal = 1.5 * red + green - 1.5 * blue
        x_filtered = _bandpass(x_signal, sample_rate_hz, config)
        y_filtered = _bandpass(y_signal, sample_rate_hz, config)
        alpha = _safe_std_ratio(x_filtered, y_filtered)
        candidate = x_filtered - alpha * y_filtered
        candidate = (candidate - float(np.mean(candidate))) * taper
        pulse[start:end] += candidate
        weights[start:end] += taper

    covered = weights > np.finfo(float).eps
    pulse[covered] /= weights[covered]
    return pulse


def preprocess_bvp(
    values: np.ndarray,
    sample_rate_hz: float,
    config: StandardRPPGConfig | None = None,
) -> np.ndarray:
    """Apply the common detrending and zero-phase pulse-band filter."""
    config = config or StandardRPPGConfig()
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or len(samples) < 3:
        raise ValueError("BVP input must be a one-dimensional signal")
    if not np.isfinite(samples).all():
        raise ValueError("BVP input contains NaN or infinite values")
    detrended = signal.detrend(samples, type="linear")
    return _bandpass(detrended, sample_rate_hz, config)


def estimate_spectral_hr(
    values: np.ndarray,
    sample_rate_hz: float,
    config: StandardRPPGConfig | None = None,
) -> dict[str, float]:
    """Estimate rPPG HR and reference-independent spectral quality features."""
    config = config or StandardRPPGConfig()
    samples = np.asarray(values, dtype=float)
    if (
        samples.ndim != 1
        or len(samples) < 3
        or len(samples) > config.nfft
        or not np.isfinite(samples).all()
        or float(np.std(samples)) <= 1e-12
    ):
        return _empty_spectral_estimate()

    centered = samples - float(np.mean(samples))
    windowed = centered * np.hanning(len(centered))
    frequencies, power = signal.periodogram(
        windowed,
        fs=sample_rate_hz,
        window="boxcar",
        detrend=False,
        nfft=config.nfft,
        scaling="density",
    )
    band = (
        (frequencies >= config.bandpass_low_hz)
        & (frequencies <= config.bandpass_high_hz)
    )
    band_indices = np.flatnonzero(band)
    if len(band_indices) < 3:
        return _empty_spectral_estimate()

    band_power = power[band_indices]
    peak_index = int(band_indices[np.argmax(band_power)])
    if peak_index <= 0 or peak_index >= len(power) - 1:
        return _empty_spectral_estimate()

    log_power = np.log(np.maximum(power, np.finfo(float).tiny))
    denominator = (
        log_power[peak_index - 1]
        - 2.0 * log_power[peak_index]
        + log_power[peak_index + 1]
    )
    delta = 0.0
    if np.isfinite(denominator) and abs(denominator) > 1e-12:
        delta = 0.5 * (
            log_power[peak_index - 1] - log_power[peak_index + 1]
        ) / denominator
        delta = float(np.clip(delta, -1.0, 1.0))

    frequency_step = float(frequencies[1] - frequencies[0])
    peak_hz = float(frequencies[peak_index] + delta * frequency_step)
    total_band_power = float(np.sum(band_power))
    peak_neighborhood = band & (np.abs(frequencies - peak_hz) <= 0.10)
    concentrated_power = float(np.sum(power[peak_neighborhood]))
    peak_power_fraction = (
        concentrated_power / total_band_power if total_band_power > 0 else np.nan
    )

    positive_band_power = band_power[band_power > 0]
    median_band_power = (
        float(np.median(positive_band_power))
        if len(positive_band_power)
        else np.nan
    )
    peak_to_median_db = (
        10.0 * np.log10(float(power[peak_index]) / median_band_power)
        if np.isfinite(median_band_power) and median_band_power > 0
        else np.nan
    )
    probability = (
        band_power / total_band_power
        if total_band_power > 0
        else np.full(len(band_power), np.nan)
    )
    finite_probability = probability[np.isfinite(probability) & (probability > 0)]
    spectral_entropy = (
        -float(np.sum(finite_probability * np.log(finite_probability)))
        / np.log(len(band_power))
        if len(finite_probability) and len(band_power) > 1
        else np.nan
    )
    return {
        "rppg_hr_bpm": peak_hz * 60.0,
        "frequency_peak_hz": peak_hz,
        "frequency_peak_bin_hz": float(frequencies[peak_index]),
        "frequency_peak_power": float(power[peak_index]),
        "frequency_interpolation_delta": delta,
        "total_band_power": total_band_power,
        "spectral_peak_power_fraction": peak_power_fraction,
        "spectral_peak_to_median_db": float(peak_to_median_db),
        "spectral_entropy": spectral_entropy,
        "rppg_signal_std": float(np.std(samples)),
    }


def process_subject_windows(
    subject: str,
    reference_windows: pd.DataFrame,
    trace_df: pd.DataFrame,
    config: StandardRPPGConfig | None = None,
) -> pd.DataFrame:
    """Create one Phase 2 row per reference window, ROI, and method."""
    config = config or StandardRPPGConfig()
    required_reference = [
        "window_id",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "time_hr_bpm",
        "frequency_hr_bpm",
        "reference_category",
        "eligible_primary",
    ]
    missing = [
        column
        for column in required_reference
        if column not in reference_windows.columns
    ]
    if missing:
        raise ValueError(f"Missing Phase 1 reference columns: {missing}")

    rows: list[dict[str, object]] = []
    for roi in config.rois:
        uniform = prepare_uniform_rgb_trace(trace_df, roi)
        bvp_by_method = {
            method: extract_standard_bvp(
                uniform.rgb,
                uniform.sample_rate_hz,
                method,
                config,
            )
            for method in config.methods
        }
        for reference in reference_windows.itertuples(index=False):
            quality = build_window_quality(
                trace_df,
                roi,
                float(reference.start_time_sec),
                float(reference.duration_sec),
                uniform.sample_rate_hz,
                config,
            )
            bvp_mask = (
                (uniform.time_sec >= float(reference.start_time_sec))
                & (uniform.time_sec <= float(reference.end_time_sec))
            )
            for method in config.methods:
                row = _base_result_row(subject, roi, method, reference, quality)
                if quality["window_status"] != "ok":
                    row.update(_empty_spectral_estimate())
                else:
                    window_bvp = bvp_by_method[method][bvp_mask]
                    if len(window_bvp) < 3:
                        row["window_status"] = "insufficient_bvp_samples"
                        row.update(_empty_spectral_estimate())
                    else:
                        row.update(
                            estimate_spectral_hr(
                                window_bvp,
                                uniform.sample_rate_hz,
                                config,
                            )
                        )
                        if not np.isfinite(float(row["rppg_hr_bpm"])):
                            row["window_status"] = "estimation_failed"
                _append_error_columns(row)
                rows.append(row)

    results = pd.DataFrame(rows)
    return add_reference_independent_agreement_features(results)


def build_window_quality(
    trace_df: pd.DataFrame,
    roi: str,
    start_time_sec: float,
    duration_sec: float,
    sample_rate_hz: float,
    config: StandardRPPGConfig | None = None,
) -> dict[str, object]:
    """Summarize source-only quality and decide if a window can be estimated."""
    config = config or StandardRPPGConfig()
    end_boundary = start_time_sec + duration_sec
    in_window = (
        (trace_df["time_sec"] >= start_time_sec)
        & (trace_df["time_sec"] < end_boundary)
    )
    window_df = trace_df.loc[in_window]
    expected_frames = max(1, int(round(duration_sec * sample_rate_hz)))
    coverage_fraction = min(1.0, len(window_df) / expected_frames)

    rgb_columns = [
        f"{roi}_r_mean",
        f"{roi}_g_mean",
        f"{roi}_b_mean",
    ]
    rgb = window_df[rgb_columns].to_numpy(dtype=float)
    roi_valid = _as_bool_array(window_df[f"{roi}_roi_valid"].to_numpy())
    valid = roi_valid & np.isfinite(rgb).all(axis=1)
    valid_frame_fraction = float(np.mean(valid)) if len(valid) else 0.0
    valid_times = window_df.loc[valid, "time_sec"].to_numpy(dtype=float)
    gap_points = np.concatenate(
        [
            np.array([start_time_sec]),
            valid_times,
            np.array([end_boundary]),
        ]
    )
    max_gap_sec = (
        float(np.max(np.diff(gap_points))) if len(gap_points) >= 2 else duration_sec
    )

    status = "ok"
    if coverage_fraction < config.min_video_coverage_fraction:
        status = "insufficient_video_coverage"
    elif valid_frame_fraction < config.min_valid_frame_fraction:
        status = "insufficient_valid_frames"
    elif max_gap_sec > config.max_interpolation_gap_sec:
        status = "interpolation_gap_too_large"

    quality: dict[str, object] = {
        "window_status": status,
        "n_trace_frames": len(window_df),
        "n_valid_frames": int(np.count_nonzero(valid)),
        "video_coverage_fraction": coverage_fraction,
        "valid_frame_fraction": valid_frame_fraction,
        "max_interpolation_gap_sec": max_gap_sec,
        "sample_rate_hz": sample_rate_hz,
    }
    quality.update(_source_quality_features(window_df, roi))
    return quality


def add_reference_independent_agreement_features(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Add POS/CHROM and cross-region agreement without reading reference HR."""
    enriched = results.copy()
    valid_hr = enriched["rppg_hr_bpm"].where(
        enriched["window_status"].eq("ok")
    )
    enriched["_valid_hr"] = valid_hr

    method_pivot = enriched.pivot_table(
        index=["subject", "window_id", "roi"],
        columns="method",
        values="_valid_hr",
        aggfunc="first",
    )
    if {"POS", "CHROM"}.issubset(method_pivot.columns):
        method_difference = (method_pivot["POS"] - method_pivot["CHROM"]).abs()
        enriched = enriched.merge(
            method_difference.rename("pos_chrom_abs_diff_bpm"),
            left_on=["subject", "window_id", "roi"],
            right_index=True,
            how="left",
        )
    else:
        enriched["pos_chrom_abs_diff_bpm"] = np.nan

    region_group = enriched.groupby(
        ["subject", "window_id", "method"],
        sort=False,
    )["_valid_hr"]
    enriched["regional_median_hr_bpm"] = region_group.transform("median")
    enriched["n_regions_available"] = region_group.transform("count").astype(int)
    enriched["regional_abs_diff_from_median_bpm"] = (
        enriched["_valid_hr"] - enriched["regional_median_hr_bpm"]
    ).abs()

    cheek_pivot = enriched.pivot_table(
        index=["subject", "window_id", "method"],
        columns="roi",
        values="_valid_hr",
        aggfunc="first",
    )
    if {"left_cheek", "right_cheek"}.issubset(cheek_pivot.columns):
        cheek_difference = (
            cheek_pivot["left_cheek"] - cheek_pivot["right_cheek"]
        ).abs()
        enriched = enriched.merge(
            cheek_difference.rename("left_right_cheek_abs_diff_bpm"),
            left_on=["subject", "window_id", "method"],
            right_index=True,
            how="left",
        )
    else:
        enriched["left_right_cheek_abs_diff_bpm"] = np.nan
    return enriched.drop(columns="_valid_hr")


def build_method_roi_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize primary-reference performance for each method and region."""
    rows = []
    for (method, roi), group in results.groupby(["method", "roi"], sort=True):
        primary = group[_as_bool_series(group["eligible_primary"])]
        evaluated = primary[
            primary["window_status"].eq("ok")
            & np.isfinite(primary["reference_hr_bpm"])
            & np.isfinite(primary["rppg_hr_bpm"])
        ]
        error = (
            evaluated["rppg_hr_bpm"].to_numpy(dtype=float)
            - evaluated["reference_hr_bpm"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "method": method,
                "roi": roi,
                "n_primary_windows": len(primary),
                "n_evaluated_windows": len(evaluated),
                "availability_rate": (
                    len(evaluated) / len(primary) if len(primary) else np.nan
                ),
                "mae_bpm": (
                    float(np.mean(np.abs(error))) if len(error) else np.nan
                ),
                "rmse_bpm": (
                    float(np.sqrt(np.mean(error**2))) if len(error) else np.nan
                ),
                "median_absolute_error_bpm": (
                    float(np.median(np.abs(error))) if len(error) else np.nan
                ),
                "bias_bpm": float(np.mean(error)) if len(error) else np.nan,
                "pearson_r": _correlation(
                    evaluated["rppg_hr_bpm"].to_numpy(dtype=float),
                    evaluated["reference_hr_bpm"].to_numpy(dtype=float),
                ),
                "within_5_bpm_rate": (
                    float(np.mean(np.abs(error) <= 5.0))
                    if len(error)
                    else np.nan
                ),
                "within_10_bpm_rate": (
                    float(np.mean(np.abs(error) <= 10.0))
                    if len(error)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_subject_qc(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize availability and primary MAE for every subject/method/ROI."""
    rows = []
    for (subject, method, roi), group in results.groupby(
        ["subject", "method", "roi"],
        sort=True,
    ):
        primary = group[_as_bool_series(group["eligible_primary"])]
        evaluated = primary[
            primary["window_status"].eq("ok")
            & np.isfinite(primary["absolute_error_bpm"])
        ]
        rows.append(
            {
                "subject": subject,
                "method": method,
                "roi": roi,
                "total_windows": len(group),
                "ok_windows": int(group["window_status"].eq("ok").sum()),
                "primary_windows": len(primary),
                "primary_evaluated_windows": len(evaluated),
                "primary_availability_rate": (
                    len(evaluated) / len(primary) if len(primary) else np.nan
                ),
                "primary_mae_bpm": (
                    float(evaluated["absolute_error_bpm"].mean())
                    if len(evaluated)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _base_result_row(
    subject: str,
    roi: str,
    method: str,
    reference,
    quality: dict[str, object],
) -> dict[str, object]:
    return {
        "subject": subject,
        "window_id": int(reference.window_id),
        "roi": roi,
        "method": method,
        "start_time_sec": float(reference.start_time_sec),
        "end_time_sec": float(reference.end_time_sec),
        "window_center_time_sec": float(reference.window_center_time_sec),
        "duration_sec": float(reference.duration_sec),
        "reference_hr_bpm": float(reference.time_hr_bpm),
        "reference_frequency_hr_bpm": float(reference.frequency_hr_bpm),
        "reference_category": str(reference.reference_category),
        "eligible_primary": bool(reference.eligible_primary),
        **quality,
    }


def _append_error_columns(row: dict[str, object]) -> None:
    reference_hr = float(row["reference_hr_bpm"])
    rppg_hr = float(row["rppg_hr_bpm"])
    if np.isfinite(reference_hr) and np.isfinite(rppg_hr):
        signed_error = rppg_hr - reference_hr
        row["signed_error_bpm"] = signed_error
        row["absolute_error_bpm"] = abs(signed_error)
    else:
        row["signed_error_bpm"] = np.nan
        row["absolute_error_bpm"] = np.nan


def _source_quality_features(
    window_df: pd.DataFrame,
    roi: str,
) -> dict[str, float]:
    return {
        "mean_face_area_ratio": _finite_mean(window_df, "face_area_ratio"),
        "median_bbox_center_jump": _finite_median(
            window_df,
            "bbox_center_jump",
            absolute=True,
        ),
        "p95_bbox_center_jump": _finite_percentile(
            window_df,
            "bbox_center_jump",
            95,
            absolute=True,
        ),
        "median_bbox_area_change": _finite_median(
            window_df,
            "bbox_area_change",
            absolute=True,
        ),
        "p95_bbox_area_change": _finite_percentile(
            window_df,
            "bbox_area_change",
            95,
            absolute=True,
        ),
        "mean_roi_fill_ratio": _finite_mean(
            window_df,
            f"{roi}_fill_ratio",
        ),
        "median_roi_pixels": _finite_median(
            window_df,
            f"{roi}_n_pixels",
        ),
        "mean_brightness": _finite_mean(
            window_df,
            f"{roi}_brightness",
        ),
        "brightness_std": _finite_std(
            window_df,
            f"{roi}_brightness",
        ),
        "mean_overexposure_ratio": _finite_mean(
            window_df,
            f"{roi}_overexposure_ratio",
        ),
    }


def _bandpass(
    values: np.ndarray,
    sample_rate_hz: float,
    config: StandardRPPGConfig,
) -> np.ndarray:
    if sample_rate_hz <= 2.0 * config.bandpass_high_hz:
        raise ValueError("RGB sampling rate is too low for the pulse band")
    sos = signal.butter(
        config.filter_order,
        [config.bandpass_low_hz, config.bandpass_high_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=float))


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("RGB input must have shape (n_samples, 3)")
    if not np.isfinite(values).all() or np.any(np.mean(values, axis=0) <= 0):
        raise ValueError("RGB values and channel means must be finite and positive")
    return values


def _safe_std_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denominator_std = float(np.std(denominator))
    if denominator_std <= 1e-12:
        return 0.0
    return float(np.std(numerator) / denominator_std)


def _empty_spectral_estimate() -> dict[str, float]:
    return {
        "rppg_hr_bpm": np.nan,
        "frequency_peak_hz": np.nan,
        "frequency_peak_bin_hz": np.nan,
        "frequency_peak_power": np.nan,
        "frequency_interpolation_delta": np.nan,
        "total_band_power": np.nan,
        "spectral_peak_power_fraction": np.nan,
        "spectral_peak_to_median_db": np.nan,
        "spectral_entropy": np.nan,
        "rppg_signal_std": np.nan,
    }


def _as_bool_array(values: np.ndarray) -> np.ndarray:
    if values.dtype == bool:
        return values
    return np.isin(
        np.char.lower(values.astype(str)),
        ["true", "1", "yes"],
    )


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _finite_values(
    frame: pd.DataFrame,
    column: str,
    absolute: bool = False,
) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    values = frame[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return np.abs(values) if absolute else values


def _finite_mean(frame: pd.DataFrame, column: str) -> float:
    values = _finite_values(frame, column)
    return float(np.mean(values)) if len(values) else np.nan


def _finite_median(
    frame: pd.DataFrame,
    column: str,
    absolute: bool = False,
) -> float:
    values = _finite_values(frame, column, absolute)
    return float(np.median(values)) if len(values) else np.nan


def _finite_percentile(
    frame: pd.DataFrame,
    column: str,
    percentile: float,
    absolute: bool = False,
) -> float:
    values = _finite_values(frame, column, absolute)
    return float(np.percentile(values, percentile)) if len(values) else np.nan


def _finite_std(frame: pd.DataFrame, column: str) -> float:
    values = _finite_values(frame, column)
    return float(np.std(values)) if len(values) else np.nan


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if (
        len(left) < 2
        or float(np.std(left)) <= 1e-12
        or float(np.std(right)) <= 1e-12
    ):
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])
