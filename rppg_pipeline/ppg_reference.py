"""Timestamp-aware contact-PPG reference construction for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import signal

from rppg_pipeline.ubfc import UBFCGroundTruth


@dataclass(frozen=True)
class PPGReferenceConfig:
    """Frozen Phase 1 settings for the contact-PPG reference."""

    bandpass_low_hz: float = 0.7
    bandpass_high_hz: float = 4.0
    filter_order: int = 3
    min_hr_bpm: float = 42.0
    max_hr_bpm: float = 240.0
    min_peaks: int = 5
    window_sec: float = 10.0
    step_sec: float = 1.0
    nfft: int = 4096
    prominence_mad_multiplier: float = 0.5
    concordant_tolerance_bpm: float = 5.0
    uncertain_tolerance_bpm: float = 10.0
    sensor_wrap_threshold_bpm: float = 42.0
    sensor_wrap_add_bpm: float = 128.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UniformGroundTruth:
    """Ground truth after duplicate-time aggregation and uniform resampling."""

    time_sec: np.ndarray
    ppg_trace: np.ndarray
    sensor_hr_raw: np.ndarray
    sensor_hr_qc: np.ndarray
    sample_rate_hz: float
    median_dt_sec: float
    original_sample_count: int
    unique_timestamp_count: int
    duplicate_timestamp_count: int
    sensor_wrap_corrected_sample_count: int


@dataclass(frozen=True)
class WindowExample:
    """Minimal PPG-only data needed to render one QC example."""

    subject: str
    category: str
    window_id: int
    time_sec: np.ndarray
    filtered_ppg: np.ndarray
    peak_indices: np.ndarray
    time_hr_bpm: float
    frequency_hr_bpm: float


@dataclass(frozen=True)
class SubjectReferenceResult:
    """All Phase 1 outputs derived from one subject's contact PPG."""

    uniform_ground_truth: UniformGroundTruth
    windows: pd.DataFrame
    examples: dict[str, WindowExample]


def prepare_uniform_ground_truth(
    ground_truth: UBFCGroundTruth,
    config: PPGReferenceConfig | None = None,
) -> UniformGroundTruth:
    """Sort, combine duplicate timestamps, and interpolate onto a uniform grid."""
    config = config or PPGReferenceConfig()
    ppg = np.asarray(ground_truth.ppg_trace, dtype=float)
    sensor_hr = np.asarray(ground_truth.sensor_hr, dtype=float)
    timestamps = np.asarray(ground_truth.timestamp_sec, dtype=float)
    if not (len(ppg) == len(sensor_hr) == len(timestamps)):
        raise ValueError("PPG, sensor HR, and timestamp lengths must match")

    original_count = len(ppg)
    finite_primary = np.isfinite(timestamps) & np.isfinite(ppg)
    timestamps = timestamps[finite_primary]
    ppg = ppg[finite_primary]
    sensor_hr = sensor_hr[finite_primary]
    if len(timestamps) < 3:
        raise ValueError("At least three finite timestamped PPG samples are required")

    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    ppg = ppg[order]
    sensor_hr = sensor_hr[order]
    sensor_hr_qc = sensor_hr.copy()
    sensor_wrap_mask = (
        np.isfinite(sensor_hr_qc)
        & (sensor_hr_qc > 0)
        & (sensor_hr_qc < config.sensor_wrap_threshold_bpm)
    )
    sensor_hr_qc[sensor_wrap_mask] += config.sensor_wrap_add_bpm

    unique_time, inverse, counts = np.unique(
        timestamps,
        return_inverse=True,
        return_counts=True,
    )
    ppg_unique = _group_mean(ppg, inverse, len(unique_time))
    sensor_raw_unique = _group_nanmean(sensor_hr, inverse, len(unique_time))
    sensor_qc_unique = _group_nanmean(sensor_hr_qc, inverse, len(unique_time))
    if len(unique_time) < 3:
        raise ValueError("At least three unique PPG timestamps are required")

    positive_dt = np.diff(unique_time)
    positive_dt = positive_dt[positive_dt > 0]
    if len(positive_dt) == 0:
        raise ValueError("PPG timestamps do not contain a positive interval")
    median_dt = float(np.median(positive_dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("Median PPG timestamp interval must be positive")

    grid_count = int(np.floor((unique_time[-1] - unique_time[0]) / median_dt)) + 1
    if grid_count < 3:
        raise ValueError("Uniform PPG grid would contain fewer than three samples")
    uniform_time = unique_time[0] + np.arange(grid_count, dtype=float) * median_dt
    uniform_ppg = np.interp(uniform_time, unique_time, ppg_unique)
    uniform_sensor_raw = _interpolate_optional(
        uniform_time,
        unique_time,
        sensor_raw_unique,
    )
    uniform_sensor_qc = _interpolate_optional(
        uniform_time,
        unique_time,
        sensor_qc_unique,
    )

    return UniformGroundTruth(
        time_sec=uniform_time,
        ppg_trace=uniform_ppg,
        sensor_hr_raw=uniform_sensor_raw,
        sensor_hr_qc=uniform_sensor_qc,
        sample_rate_hz=1.0 / median_dt,
        median_dt_sec=median_dt,
        original_sample_count=original_count,
        unique_timestamp_count=len(unique_time),
        duplicate_timestamp_count=int(np.sum(counts - 1)),
        sensor_wrap_corrected_sample_count=int(np.count_nonzero(sensor_wrap_mask)),
    )


def preprocess_ppg(
    values: np.ndarray,
    sample_rate_hz: float,
    config: PPGReferenceConfig,
) -> np.ndarray:
    """Linearly detrend and zero-phase band-pass filter one PPG window."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or len(samples) < 3:
        raise ValueError("PPG input must be a one-dimensional signal")
    if not np.isfinite(samples).all():
        raise ValueError("PPG input contains NaN or infinite values")
    if sample_rate_hz <= 2 * config.bandpass_high_hz:
        raise ValueError("PPG sampling rate is too low for the configured band-pass")

    detrended = signal.detrend(samples, type="linear")
    sos = signal.butter(
        config.filter_order,
        [
            config.bandpass_low_hz,
            config.bandpass_high_hz,
        ],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, detrended)


def estimate_time_domain_hr(
    filtered_ppg: np.ndarray,
    sample_rate_hz: float,
    config: PPGReferenceConfig,
) -> dict[str, object]:
    """Estimate contact-PPG HR from the median inter-beat interval."""
    values = np.asarray(filtered_ppg, dtype=float)
    robust_scale = _robust_mad_scale(values)
    prominence = config.prominence_mad_multiplier * robust_scale
    if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
        return _empty_time_result(prominence)
    min_distance = max(
        1,
        int(np.floor(sample_rate_hz * 60.0 / config.max_hr_bpm)),
    )
    if not np.isfinite(prominence) or prominence <= 0:
        return _empty_time_result(prominence)

    peak_indices, properties = signal.find_peaks(
        values,
        distance=min_distance,
        prominence=prominence,
    )
    if len(peak_indices) < config.min_peaks:
        return {
            **_empty_time_result(prominence),
            "n_peaks": len(peak_indices),
            "peak_indices": peak_indices,
        }

    refined_peak_locations = _refine_peak_locations(values, peak_indices)
    ibi_sec = np.diff(refined_peak_locations) / sample_rate_hz
    valid_ibi = np.isfinite(ibi_sec) & (ibi_sec > 0)
    ibi_sec = ibi_sec[valid_ibi]
    if len(ibi_sec) < config.min_peaks - 1:
        return {
            **_empty_time_result(prominence),
            "n_peaks": len(peak_indices),
            "n_valid_ibi": len(ibi_sec),
            "peak_indices": peak_indices,
        }

    median_ibi = float(np.median(ibi_sec))
    hr_bpm = 60.0 / median_ibi
    ibi_mean = float(np.mean(ibi_sec))
    ibi_cv = (
        float(np.std(ibi_sec, ddof=1) / ibi_mean)
        if len(ibi_sec) > 1 and ibi_mean > 0
        else np.nan
    )
    return {
        "time_hr_bpm": hr_bpm,
        "n_peaks": len(peak_indices),
        "n_valid_ibi": len(ibi_sec),
        "median_ibi_sec": median_ibi,
        "ibi_cv": ibi_cv,
        "prominence_threshold": prominence,
        "median_peak_prominence": float(np.median(properties["prominences"])),
        "peak_indices": peak_indices,
    }


def estimate_frequency_domain_hr(
    values: np.ndarray,
    sample_rate_hz: float,
    config: PPGReferenceConfig,
) -> dict[str, float]:
    """Estimate HR with a 4096-point Hann periodogram and peak interpolation."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or len(samples) < 3 or not np.isfinite(samples).all():
        return _empty_frequency_result()
    if float(np.std(samples)) <= 1e-12:
        return _empty_frequency_result()
    if len(samples) > config.nfft:
        raise ValueError("Formal frequency estimator requires window length <= nfft")

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
    if len(band_indices) == 0:
        return _empty_frequency_result()

    peak_index = int(band_indices[np.argmax(power[band_indices])])
    if peak_index <= 0 or peak_index >= len(power) - 1:
        return _empty_frequency_result()
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
    interpolated_hz = float(frequencies[peak_index] + delta * frequency_step)
    return {
        "frequency_hr_bpm": interpolated_hz * 60.0,
        "frequency_peak_hz": interpolated_hz,
        "frequency_peak_bin_hz": float(frequencies[peak_index]),
        "frequency_peak_power": float(power[peak_index]),
        "frequency_interpolation_delta": delta,
    }


def classify_reference(
    time_hr_bpm: float,
    frequency_hr_bpm: float,
    n_peaks: int,
    config: PPGReferenceConfig,
) -> tuple[str, float]:
    """Assign the frozen four-level time-frequency concordance category."""
    if (
        n_peaks < config.min_peaks
        or not np.isfinite(time_hr_bpm)
        or not np.isfinite(frequency_hr_bpm)
    ):
        return "insufficient", np.nan
    disagreement = abs(time_hr_bpm - frequency_hr_bpm)
    if disagreement <= config.concordant_tolerance_bpm:
        return "concordant", disagreement
    if disagreement <= config.uncertain_tolerance_bpm:
        return "uncertain", disagreement
    return "discordant", disagreement


def build_subject_reference(
    subject: str,
    ground_truth: UBFCGroundTruth,
    config: PPGReferenceConfig | None = None,
) -> SubjectReferenceResult:
    """Build all timestamp-aware PPG reference windows for one subject."""
    config = config or PPGReferenceConfig()
    uniform = prepare_uniform_ground_truth(ground_truth, config)
    window_n = max(3, int(round(config.window_sec * uniform.sample_rate_hz)))
    step_n = max(1, int(round(config.step_sec * uniform.sample_rate_hz)))
    if window_n > config.nfft:
        raise ValueError("PPG reference window is longer than the fixed nfft")

    rows: list[dict[str, object]] = []
    examples: dict[str, WindowExample] = {}
    window_id = 0
    for start in range(0, max(0, len(uniform.ppg_trace) - window_n + 1), step_n):
        end = start + window_n
        window_id += 1
        time_values = uniform.time_sec[start:end]
        ppg_values = uniform.ppg_trace[start:end]
        sensor_raw_values = uniform.sensor_hr_raw[start:end]
        sensor_qc_values = uniform.sensor_hr_qc[start:end]
        filtered = preprocess_ppg(ppg_values, uniform.sample_rate_hz, config)
        time_result = estimate_time_domain_hr(
            filtered,
            uniform.sample_rate_hz,
            config,
        )
        frequency_result = estimate_frequency_domain_hr(
            filtered,
            uniform.sample_rate_hz,
            config,
        )
        category, disagreement = classify_reference(
            float(time_result["time_hr_bpm"]),
            float(frequency_result["frequency_hr_bpm"]),
            int(time_result["n_peaks"]),
            config,
        )
        sensor_raw_finite = sensor_raw_values[np.isfinite(sensor_raw_values)]
        sensor_qc_finite = sensor_qc_values[np.isfinite(sensor_qc_values)]
        sensor_raw_mean = (
            float(np.mean(sensor_raw_finite)) if len(sensor_raw_finite) else np.nan
        )
        sensor_raw_median = (
            float(np.median(sensor_raw_finite))
            if len(sensor_raw_finite)
            else np.nan
        )
        sensor_qc_mean = (
            float(np.mean(sensor_qc_finite)) if len(sensor_qc_finite) else np.nan
        )
        sensor_qc_median = (
            float(np.median(sensor_qc_finite)) if len(sensor_qc_finite) else np.nan
        )
        time_hr = float(time_result["time_hr_bpm"])
        frequency_hr = float(frequency_result["frequency_hr_bpm"])
        rows.append(
            {
                "subject": subject,
                "window_id": window_id,
                "start_sample": start,
                "end_sample": end - 1,
                "start_time_sec": float(time_values[0]),
                "end_time_sec": float(time_values[-1]),
                "window_center_time_sec": float(
                    (time_values[0] + time_values[-1]) / 2.0
                ),
                "duration_sec": len(time_values) / uniform.sample_rate_hz,
                "n_samples": len(time_values),
                "sample_rate_hz": uniform.sample_rate_hz,
                "time_hr_bpm": time_hr,
                "frequency_hr_bpm": frequency_hr,
                "time_frequency_abs_diff_bpm": disagreement,
                "reference_category": category,
                "eligible_primary": category == "concordant",
                "n_peaks": int(time_result["n_peaks"]),
                "n_valid_ibi": int(time_result["n_valid_ibi"]),
                "median_ibi_sec": float(time_result["median_ibi_sec"]),
                "ibi_cv": float(time_result["ibi_cv"]),
                "prominence_threshold": float(
                    time_result["prominence_threshold"]
                ),
                "median_peak_prominence": float(
                    time_result["median_peak_prominence"]
                ),
                "frequency_peak_hz": float(
                    frequency_result["frequency_peak_hz"]
                ),
                "frequency_peak_bin_hz": float(
                    frequency_result["frequency_peak_bin_hz"]
                ),
                "frequency_peak_power": float(
                    frequency_result["frequency_peak_power"]
                ),
                "frequency_interpolation_delta": float(
                    frequency_result["frequency_interpolation_delta"]
                ),
                "sensor_hr_raw_mean_bpm": sensor_raw_mean,
                "sensor_hr_raw_median_bpm": sensor_raw_median,
                "sensor_hr_qc_mean_bpm": sensor_qc_mean,
                "sensor_hr_qc_median_bpm": sensor_qc_median,
                "sensor_wrap_corrected_fraction": float(
                    np.mean(
                        np.isfinite(sensor_raw_values)
                        & (sensor_raw_values > 0)
                        & (sensor_raw_values < config.sensor_wrap_threshold_bpm)
                    )
                ),
                "sensor_time_abs_diff_bpm": (
                    abs(sensor_qc_median - time_hr)
                    if np.isfinite(sensor_qc_median) and np.isfinite(time_hr)
                    else np.nan
                ),
                "sensor_frequency_abs_diff_bpm": (
                    abs(sensor_qc_median - frequency_hr)
                    if np.isfinite(sensor_qc_median) and np.isfinite(frequency_hr)
                    else np.nan
                ),
            }
        )
        if category not in examples:
            examples[category] = WindowExample(
                subject=subject,
                category=category,
                window_id=window_id,
                time_sec=time_values.copy(),
                filtered_ppg=filtered.copy(),
                peak_indices=np.asarray(time_result["peak_indices"], dtype=int),
                time_hr_bpm=time_hr,
                frequency_hr_bpm=frequency_hr,
            )

    return SubjectReferenceResult(
        uniform_ground_truth=uniform,
        windows=pd.DataFrame(rows),
        examples=examples,
    )


def build_subject_qc(
    subject: str,
    result: SubjectReferenceResult,
) -> dict[str, object]:
    """Summarize Phase 1 retention and reference categories for one subject."""
    windows = result.windows
    category_counts = windows["reference_category"].value_counts()
    total = len(windows)
    concordant = int(category_counts.get("concordant", 0))
    return {
        "subject": subject,
        "total_windows": total,
        "concordant_windows": concordant,
        "uncertain_windows": int(category_counts.get("uncertain", 0)),
        "discordant_windows": int(category_counts.get("discordant", 0)),
        "insufficient_windows": int(category_counts.get("insufficient", 0)),
        "primary_retention_rate": concordant / total if total else 0.0,
        "median_time_frequency_abs_diff_bpm": float(
            windows["time_frequency_abs_diff_bpm"].median()
        ),
        "median_sensor_time_abs_diff_bpm": float(
            windows["sensor_time_abs_diff_bpm"].median()
        ),
        "median_sensor_frequency_abs_diff_bpm": float(
            windows["sensor_frequency_abs_diff_bpm"].median()
        ),
    }


def _group_mean(
    values: np.ndarray,
    inverse: np.ndarray,
    group_count: int,
) -> np.ndarray:
    sums = np.bincount(inverse, weights=values, minlength=group_count)
    counts = np.bincount(inverse, minlength=group_count)
    return sums / counts


def _group_nanmean(
    values: np.ndarray,
    inverse: np.ndarray,
    group_count: int,
) -> np.ndarray:
    finite = np.isfinite(values)
    sums = np.bincount(
        inverse[finite],
        weights=values[finite],
        minlength=group_count,
    )
    counts = np.bincount(inverse[finite], minlength=group_count)
    means = np.full(group_count, np.nan, dtype=float)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    return means


def _interpolate_optional(
    uniform_time: np.ndarray,
    unique_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(values)
    if np.count_nonzero(finite) == 0:
        return np.full(len(uniform_time), np.nan, dtype=float)
    if np.count_nonzero(finite) == 1:
        return np.full(len(uniform_time), values[finite][0], dtype=float)
    return np.interp(uniform_time, unique_time[finite], values[finite])


def _robust_mad_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    return scale


def _refine_peak_locations(
    values: np.ndarray,
    peak_indices: np.ndarray,
) -> np.ndarray:
    """Refine discrete peak positions with local three-point interpolation."""
    refined = peak_indices.astype(float)
    for output_idx, peak_idx in enumerate(peak_indices):
        if peak_idx <= 0 or peak_idx >= len(values) - 1:
            continue
        left = float(values[peak_idx - 1])
        center = float(values[peak_idx])
        right = float(values[peak_idx + 1])
        denominator = left - 2.0 * center + right
        if np.isfinite(denominator) and abs(denominator) > 1e-12:
            delta = 0.5 * (left - right) / denominator
            refined[output_idx] += float(np.clip(delta, -1.0, 1.0))
    return refined


def _empty_time_result(prominence: float) -> dict[str, object]:
    return {
        "time_hr_bpm": np.nan,
        "n_peaks": 0,
        "n_valid_ibi": 0,
        "median_ibi_sec": np.nan,
        "ibi_cv": np.nan,
        "prominence_threshold": prominence,
        "median_peak_prominence": np.nan,
        "peak_indices": np.array([], dtype=int),
    }


def _empty_frequency_result() -> dict[str, float]:
    return {
        "frequency_hr_bpm": np.nan,
        "frequency_peak_hz": np.nan,
        "frequency_peak_bin_hz": np.nan,
        "frequency_peak_power": np.nan,
        "frequency_interpolation_delta": np.nan,
    }
