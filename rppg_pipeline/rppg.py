# POS CHROM and original baseline

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from rppg_pipeline.degradation import effective_sample_rate

ROIS = ("full_face_inner", "forehead", "cheeks_mean")
METHODS = ("POS", "CHROM")


# Estimate one rPPG heart rate
def estimate_hr(
    rgb: np.ndarray,
    sample_rate_hz: float,
    method: str,
) -> float:
    bvp = _extract_bvp(rgb, sample_rate_hz, method)
    return float(_spectral_features(bvp, sample_rate_hz)["rppg_hr_bpm"])


# Build original candidate rows
def build_original_candidates(
    subject: str,
    references: pd.DataFrame,
    trace: pd.DataFrame,
) -> pd.DataFrame:
    return build_candidates(subject, references, trace, "original")


# Build candidate rows for one condition
def build_candidates(
    subject: str,
    references: pd.DataFrame,
    trace: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reference in references.itertuples(index=False):
        selected = trace[
            (trace["time_sec"] >= float(reference.start_time_sec))
            & (trace["time_sec"] < float(reference.end_time_sec))
        ]
        sample_rate_hz = effective_sample_rate(
            selected["time_sec"].to_numpy(dtype=float)
        )
        for roi in ROIS:
            status, rgb, quality = _prepare_window(
                selected,
                roi,
                float(reference.start_time_sec),
                float(reference.end_time_sec),
                sample_rate_hz,
            )
            for method in METHODS:
                row = {
                    "subject": subject,
                    "condition": condition,
                    "window_id": int(reference.window_id),
                    "start_time_sec": float(reference.start_time_sec),
                    "end_time_sec": float(reference.end_time_sec),
                    "duration_sec": float(reference.duration_sec),
                    "roi": roi,
                    "method": method,
                    "reference_hr_bpm": float(reference.reference_hr_bpm),
                    "reference_valid": bool(reference.reference_valid),
                    "window_status": status,
                    **quality,
                }
                if status == "ok":
                    bvp = _extract_bvp(rgb, sample_rate_hz, method)
                    row.update(_spectral_features(bvp, sample_rate_hz))
                else:
                    row.update(_empty_spectral_features())
                estimate = float(row["rppg_hr_bpm"])
                reference_hr = float(row["reference_hr_bpm"])
                if np.isfinite(estimate):
                    row["signed_error_bpm"] = estimate - reference_hr
                    row["absolute_error_bpm"] = abs(estimate - reference_hr)
                else:
                    row["signed_error_bpm"] = np.nan
                    row["absolute_error_bpm"] = np.nan
                rows.append(row)
    candidates = pd.DataFrame(rows)
    return _add_method_agreement(candidates)


# Prepare one RGB window
def _prepare_window(
    trace: pd.DataFrame,
    roi: str,
    start_time_sec: float,
    end_time_sec: float,
    sample_rate_hz: float,
) -> tuple[str, np.ndarray, dict[str, float]]:
    times = trace["time_sec"].to_numpy(dtype=float)
    rgb = trace[
        [f"{roi}_r_mean", f"{roi}_g_mean", f"{roi}_b_mean"]
    ].to_numpy(dtype=float)
    valid = trace[f"{roi}_valid"].to_numpy(dtype=bool) & np.isfinite(rgb).all(axis=1)
    duration_sec = end_time_sec - start_time_sec
    expected_frames = (
        int(round(duration_sec * sample_rate_hz))
        if np.isfinite(sample_rate_hz)
        else 0
    )
    coverage = min(1.0, len(trace) / expected_frames) if expected_frames else 0.0
    valid_fraction = float(np.mean(valid)) if len(valid) else 0.0
    valid_times = times[valid]
    gap_points = np.concatenate(([start_time_sec], valid_times, [end_time_sec]))
    max_gap = float(np.max(np.diff(gap_points)))

    status = "ok"
    if coverage < 0.95:
        status = "low_video_coverage"
    elif valid_fraction < 0.80:
        status = "low_valid_fraction"
    elif max_gap > 0.50:
        status = "long_missing_gap"

    quality = {
        "sample_rate_hz": sample_rate_hz,
        "n_frames": len(trace),
        "valid_frame_fraction": valid_fraction,
        "max_missing_gap_sec": max_gap,
        "mean_face_area_ratio": _mean(trace, "face_area_ratio"),
        "mean_face_center_shift": _mean(trace, "face_center_shift"),
        "mean_face_area_change": _mean(trace, "face_area_change"),
        "mean_brightness": _mean(trace, f"{roi}_brightness"),
        "brightness_std": _std(trace, f"{roi}_brightness"),
        "mean_overexposure_ratio": _mean(
            trace,
            f"{roi}_overexposure_ratio",
        ),
    }
    if status != "ok":
        return status, np.empty((0, 3)), quality
    interpolated = np.column_stack(
        [np.interp(times, valid_times, rgb[valid, channel]) for channel in range(3)]
    )
    return status, interpolated, quality


# Extract POS or CHROM signal
def _extract_bvp(
    rgb: np.ndarray,
    sample_rate_hz: float,
    method: str,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=float)
    if method == "POS":
        pulse = _pos(values, sample_rate_hz)
    elif method == "CHROM":
        pulse = _chrom(values, sample_rate_hz)
    else:
        raise ValueError(f"Unknown rPPG method: {method}")
    return _bandpass(signal.detrend(pulse), sample_rate_hz)


# Extract POS signal
def _pos(rgb: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    window_size = max(3, int(round(1.6 * sample_rate_hz)))
    pulse = np.zeros(len(rgb), dtype=float)
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for end in range(window_size - 1, len(rgb)):
        start = end - window_size + 1
        window = rgb[start : end + 1].T
        normalized = window / np.mean(window, axis=1)[:, np.newaxis]
        projected = projection @ normalized
        alpha = _std_ratio(projected[0], projected[1])
        candidate = projected[0] + alpha * projected[1]
        pulse[start : end + 1] += candidate - float(np.mean(candidate))
    return pulse


# Extract CHROM signal
def _chrom(rgb: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    window_size = max(3, int(round(1.6 * sample_rate_hz)))
    step = max(1, window_size // 2)
    starts = list(range(0, len(rgb) - window_size + 1, step))
    if starts[-1] != len(rgb) - window_size:
        starts.append(len(rgb) - window_size)
    pulse = np.zeros(len(rgb), dtype=float)
    weights = np.zeros(len(rgb), dtype=float)
    taper = np.hanning(window_size + 2)[1:-1]
    for start in starts:
        end = start + window_size
        normalized = rgb[start:end] / np.mean(rgb[start:end], axis=0) - 1.0
        red, green, blue = normalized.T
        x_signal = _bandpass(3.0 * red - 2.0 * green, sample_rate_hz)
        y_signal = _bandpass(1.5 * red + green - 1.5 * blue, sample_rate_hz)
        candidate = x_signal - _std_ratio(x_signal, y_signal) * y_signal
        candidate = (candidate - float(np.mean(candidate))) * taper
        pulse[start:end] += candidate
        weights[start:end] += taper
    covered = weights > np.finfo(float).eps
    pulse[covered] /= weights[covered]
    return pulse


# Apply pulse bandpass
def _bandpass(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    sos = signal.butter(
        3,
        [0.7, 4.0],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    # Fit short CHROM windows
    pad_length = min(3 * (2 * len(sos) + 1), len(values) - 1)
    return signal.sosfiltfilt(sos, values, padlen=pad_length)


# Estimate spectral features
def _spectral_features(
    bvp: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, float]:
    windowed = (bvp - float(np.mean(bvp))) * np.hanning(len(bvp))
    frequencies, power = signal.periodogram(
        windowed,
        fs=sample_rate_hz,
        window="boxcar",
        detrend=False,
        nfft=4096,
        scaling="density",
    )
    band = (frequencies >= 0.7) & (frequencies <= 4.0)
    band_indices = np.flatnonzero(band)
    peak_index = int(band_indices[np.argmax(power[band_indices])])
    peak_hz = _refine_peak(frequencies, power, peak_index)
    band_power = power[band_indices]
    total_power = float(np.sum(band_power))
    probability = band_power / total_power
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-30))))
    entropy /= np.log(len(probability))
    peak_neighborhood = band & (np.abs(frequencies - peak_hz) <= 0.10)
    return {
        "rppg_hr_bpm": peak_hz * 60.0,
        "spectral_peak_fraction": float(np.sum(power[peak_neighborhood]) / total_power),
        "spectral_entropy": entropy,
        "rppg_signal_std": float(np.std(bvp)),
    }


# Refine spectral peak
def _refine_peak(
    frequencies: np.ndarray,
    power: np.ndarray,
    peak_index: int,
) -> float:
    log_power = np.log(np.maximum(power, np.finfo(float).tiny))
    denominator = (
        log_power[peak_index - 1]
        - 2.0 * log_power[peak_index]
        + log_power[peak_index + 1]
    )
    delta = 0.0
    if abs(denominator) > 1e-12:
        delta = 0.5 * (
            log_power[peak_index - 1] - log_power[peak_index + 1]
        ) / denominator
        delta = float(np.clip(delta, -1.0, 1.0))
    return float(frequencies[peak_index] + delta * (frequencies[1] - frequencies[0]))


# Add POS CHROM agreement
def _add_method_agreement(candidates: pd.DataFrame) -> pd.DataFrame:
    pivot = candidates.pivot(
        index=["subject", "condition", "window_id", "roi"],
        columns="method",
        values="rppg_hr_bpm",
    )
    difference = (pivot["POS"] - pivot["CHROM"]).abs()
    return candidates.merge(
        difference.rename("pos_chrom_diff_bpm"),
        left_on=["subject", "condition", "window_id", "roi"],
        right_index=True,
        how="left",
    )


# Build empty spectral features
def _empty_spectral_features() -> dict[str, float]:
    return {
        "rppg_hr_bpm": np.nan,
        "spectral_peak_fraction": np.nan,
        "spectral_entropy": np.nan,
        "rppg_signal_std": np.nan,
    }


# Calculate standard deviation ratio
def _std_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    scale = float(np.std(denominator))
    return float(np.std(numerator) / scale) if scale > 1e-12 else 0.0


# Calculate finite mean
def _mean(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


# Calculate finite standard deviation
def _std(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if len(values) else np.nan
