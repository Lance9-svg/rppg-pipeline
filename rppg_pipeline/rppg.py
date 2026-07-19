# CHROM/POS rPPG signal processing and heart-rate export.

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RPPGConfig:
    # ROI column prefix used as the RGB source.
    roi: str = "cheeks_mean"
    # Lower bandpass cutoff in Hz.
    bandpass_low_hz: float = 0.7
    # Upper bandpass cutoff in Hz.
    bandpass_high_hz: float = 4.0
    # Butterworth filter order.
    filter_order: int = 3
    # HR estimator used for segment and window estimates.
    hr_estimator: str = "welch"
    # Sliding window length for HR curve points.
    hr_window_sec: float = 10.0
    # Sliding window step for HR curve points.
    hr_step_sec: float = 1.0

    def to_dict(self) -> dict[str, object]:
        # Export processing parameters for runtime files.
        return asdict(self)


@dataclass(frozen=True)
class RGBSegment:
    # Segment id copied from valid_segments.csv.
    segment_id: int
    # ROI name copied from valid_segments.csv.
    roi: str
    # Original frame indices included in this segment.
    frame_idx: np.ndarray
    # Original timestamps in seconds.
    time_sec: np.ndarray
    # RGB values as columns R, G, B.
    rgb: np.ndarray
    # First frame in the valid segment.
    start_frame: int
    # Last frame in the valid segment.
    end_frame: int
    # Segment duration in seconds.
    duration_sec: float


def process_rppg_from_outputs(
    out_dir: str | Path,
    fps: float,
    config: RPPGConfig | None = None,
) -> None:
    # Run Milestone 3 from existing rgb_trace.csv and valid_segments.csv.
    config = config or RPPGConfig()
    if fps <= 0:
        raise ValueError(f"FPS must be positive for rPPG processing, got {fps}")

    out_path = Path(out_dir)
    runtime_rows = []
    total_start = time.perf_counter()

    load_start = time.perf_counter()
    trace_df = pd.read_csv(out_path / "rgb_trace.csv")
    segments_df = pd.read_csv(out_path / "valid_segments.csv")
    _append_runtime(runtime_rows, "load_inputs", load_start)

    select_start = time.perf_counter()
    segments = select_rgb_segments(trace_df, segments_df, config.roi)
    _append_runtime(runtime_rows, "select_valid_segments", select_start)
    if not segments:
        raise ValueError(f"No long-enough valid segments found for ROI: {config.roi}")

    signal_rows: list[dict[str, object]] = []
    segment_signals: list[dict[str, object]] = []

    for segment_item in segments:
        normalized_rgb = mean_normalize_rgb(segment_item.rgb)

        chrom_start = time.perf_counter()
        chrom_raw = chrom_signal(segment_item.rgb)
        _append_runtime(runtime_rows, "chrom", chrom_start)

        pos_start = time.perf_counter()
        pos_raw = pos_signal(segment_item.rgb)
        _append_runtime(runtime_rows, "pos", pos_start)

        filter_start = time.perf_counter()
        chrom_filtered = bandpass_filter(
            chrom_raw,
            fps=fps,
            low_hz=config.bandpass_low_hz,
            high_hz=config.bandpass_high_hz,
            order=config.filter_order,
        )
        pos_filtered = bandpass_filter(
            pos_raw,
            fps=fps,
            low_hz=config.bandpass_low_hz,
            high_hz=config.bandpass_high_hz,
            order=config.filter_order,
        )
        _append_runtime(runtime_rows, "filtering", filter_start)

        segment_signals.extend(
            [
                {
                    "segment": segment_item,
                    "method": "CHROM",
                    "raw": chrom_raw,
                    "filtered": chrom_filtered,
                },
                {
                    "segment": segment_item,
                    "method": "POS",
                    "raw": pos_raw,
                    "filtered": pos_filtered,
                },
            ]
        )

        signal_rows.extend(
            _signal_rows(
                segment_item,
                normalized_rgb,
                chrom_raw,
                chrom_filtered,
                pos_raw,
                pos_filtered,
            )
        )

    hr_start = time.perf_counter()
    hr_rows = build_hr_results(segment_signals, fps, config)
    _append_runtime(runtime_rows, "hr_estimation", hr_start)

    write_start = time.perf_counter()
    pd.DataFrame(signal_rows).to_csv(out_path / "rppg_signal.csv", index=False)
    hr_df = pd.DataFrame(hr_rows)
    hr_df.to_csv(out_path / "hr_results.csv", index=False)
    _append_runtime(runtime_rows, "write_signal_results", write_start)

    plot_start = time.perf_counter()
    plot_hr_curve(hr_df, out_path / "hr_curve.png")
    _append_runtime(runtime_rows, "plot_hr_curve", plot_start)

    runtime_rows.append(
        {
            "stage": "total",
            "runtime_sec": time.perf_counter() - total_start,
            **config.to_dict(),
        }
    )
    pd.DataFrame(runtime_rows).to_csv(out_path / "runtime_results.csv", index=False)


def select_rgb_segments(
    trace_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    roi: str = "cheeks_mean",
) -> list[RGBSegment]:
    # Select long-enough valid RGB chunks for one ROI.
    required_cols = [
        "frame_idx",
        "time_sec",
        f"{roi}_r_mean",
        f"{roi}_g_mean",
        f"{roi}_b_mean",
        f"{roi}_roi_valid",
    ]
    missing = [column for column in required_cols if column not in trace_df.columns]
    if missing:
        raise ValueError(f"Missing RGB trace columns: {missing}")

    roi_segments = segments_df[
        (segments_df["roi"] == roi) & _as_bool_series(segments_df["is_long_enough"])
    ]
    selected = []
    for row in roi_segments.itertuples(index=False):
        start_frame = int(row.start_frame)
        end_frame = int(row.end_frame)
        chunk = trace_df[
            (trace_df["frame_idx"] >= start_frame)
            & (trace_df["frame_idx"] <= end_frame)
            & _as_bool_series(trace_df[f"{roi}_roi_valid"])
        ].copy()
        rgb = chunk[[f"{roi}_r_mean", f"{roi}_g_mean", f"{roi}_b_mean"]].to_numpy(
            dtype=float
        )
        finite = np.isfinite(rgb).all(axis=1)
        chunk = chunk.loc[finite]
        rgb = rgb[finite]
        if len(chunk) == 0:
            continue
        selected.append(
            RGBSegment(
                segment_id=int(row.segment_id),
                roi=roi,
                frame_idx=chunk["frame_idx"].to_numpy(dtype=int),
                time_sec=chunk["time_sec"].to_numpy(dtype=float),
                rgb=rgb,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_sec=float(row.duration_sec),
            )
        )
    return selected


def mean_normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    # Normalize each RGB channel by its mean and subtract 1.
    values = np.asarray(rgb, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("RGB input must have shape (n_samples, 3)")
    means = np.mean(values, axis=0)
    if np.any(~np.isfinite(means)) or np.any(means <= 0):
        raise ValueError("RGB channel means must be positive and finite")
    return values / means - 1.0


def chrom_signal(rgb: np.ndarray) -> np.ndarray:
    # Compute CHROM projection from mean-normalized RGB channels.
    normalized = mean_normalize_rgb(rgb)
    red = normalized[:, 0]
    green = normalized[:, 1]
    blue = normalized[:, 2]
    x_signal = 3.0 * red - 2.0 * green
    y_signal = 1.5 * red + green - 1.5 * blue
    alpha = _safe_std_ratio(x_signal, y_signal)
    return _center_signal(x_signal - alpha * y_signal)


def pos_signal(rgb: np.ndarray) -> np.ndarray:
    # Compute POS projection from mean-normalized RGB channels.
    normalized = mean_normalize_rgb(rgb)
    red = normalized[:, 0]
    green = normalized[:, 1]
    blue = normalized[:, 2]
    s1 = green - blue
    s2 = -2.0 * red + green + blue
    alpha = _safe_std_ratio(s1, s2)
    return _center_signal(s1 + alpha * s2)


def bandpass_filter(
    values: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    order: int = 3,
) -> np.ndarray:
    # Apply a Butterworth bandpass filter to one rPPG signal.
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1:
        raise ValueError("Bandpass input must be one-dimensional")
    if not np.isfinite(samples).all():
        raise ValueError("Bandpass input contains NaN or infinite values")
    nyquist = fps / 2.0
    high = min(high_hz, nyquist * 0.99)
    if low_hz <= 0 or high <= low_hz:
        raise ValueError("Invalid bandpass frequency range")
    sos = signal.butter(
        order,
        [low_hz / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    try:
        # sosfiltfilt applies zero-phase forward/backward filtering.
        return signal.sosfiltfilt(sos, samples)
    except ValueError:
        # sosfilt is a fallback for very short smoke-test signals.
        return signal.sosfilt(sos, samples)


def estimate_heart_rate(
    values: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
    estimator: str = "welch",
) -> dict[str, float]:
    # Estimate heart rate from the strongest spectral peak in the HR band.
    samples = _center_signal(np.asarray(values, dtype=float))
    if len(samples) < 3:
        return _empty_hr_estimate()
    if estimator == "welch":
        freqs, power = signal.welch(samples, fs=fps, nperseg=len(samples))
    elif estimator == "fft":
        windowed = samples * np.hanning(len(samples))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / fps)
        power = np.abs(np.fft.rfft(windowed)) ** 2
    else:
        raise ValueError(f"Unsupported HR estimator: {estimator}")

    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return _empty_hr_estimate()
    band_freqs = freqs[band]
    band_power = power[band]
    peak_idx = int(np.argmax(band_power))
    peak_hz = float(band_freqs[peak_idx])
    return {
        "hr_bpm": peak_hz * 60.0,
        "peak_frequency_hz": peak_hz,
        "peak_power": float(band_power[peak_idx]),
    }


def build_hr_results(
    segment_signals: list[dict[str, object]],
    fps: float,
    config: RPPGConfig,
) -> list[dict[str, object]]:
    # Build segment-level and sliding-window HR estimates.
    rows = []
    window_n = max(3, int(round(config.hr_window_sec * fps)))
    step_n = max(1, int(round(config.hr_step_sec * fps)))
    for item in segment_signals:
        segment_item: RGBSegment = item["segment"]
        method = str(item["method"])
        filtered = np.asarray(item["filtered"], dtype=float)
        rows.append(
            _hr_row(
                segment_item,
                method,
                "segment",
                int(segment_item.frame_idx[0]),
                int(segment_item.frame_idx[-1]),
                float(segment_item.time_sec[0]),
                float(segment_item.time_sec[-1]),
                filtered,
                fps,
                config,
            )
        )
        for start in range(0, max(0, len(filtered) - window_n + 1), step_n):
            end = start + window_n
            rows.append(
                _hr_row(
                    segment_item,
                    method,
                    "window",
                    int(segment_item.frame_idx[start]),
                    int(segment_item.frame_idx[end - 1]),
                    float(segment_item.time_sec[start]),
                    float(segment_item.time_sec[end - 1]),
                    filtered[start:end],
                    fps,
                    config,
                )
            )
    return rows


def plot_hr_curve(hr_df: pd.DataFrame, out_path: str | Path) -> None:
    # Save a simple HR curve plot for CHROM and POS window estimates.
    fig, ax = plt.subplots(figsize=(10, 4))
    curve_df = hr_df[hr_df["estimate_type"] == "window"]
    if curve_df.empty:
        curve_df = hr_df[hr_df["estimate_type"] == "segment"]
    for method, group in curve_df.groupby("method"):
        group = group.sort_values("window_center_time_sec")
        ax.plot(
            group["window_center_time_sec"],
            group["hr_bpm"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=method,
        )
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.set_title("rPPG Heart Rate Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _signal_rows(
    segment_item: RGBSegment,
    normalized_rgb: np.ndarray,
    chrom_raw: np.ndarray,
    chrom_filtered: np.ndarray,
    pos_raw: np.ndarray,
    pos_filtered: np.ndarray,
) -> list[dict[str, object]]:
    # Convert per-sample signals into CSV rows.
    rows = []
    for idx in range(len(segment_item.frame_idx)):
        rows.append(
            {
                "segment_id": segment_item.segment_id,
                "roi": segment_item.roi,
                "frame_idx": int(segment_item.frame_idx[idx]),
                "time_sec": float(segment_item.time_sec[idx]),
                "r": float(segment_item.rgb[idx, 0]),
                "g": float(segment_item.rgb[idx, 1]),
                "b": float(segment_item.rgb[idx, 2]),
                "r_norm": float(normalized_rgb[idx, 0]),
                "g_norm": float(normalized_rgb[idx, 1]),
                "b_norm": float(normalized_rgb[idx, 2]),
                "chrom_raw": float(chrom_raw[idx]),
                "chrom_filtered": float(chrom_filtered[idx]),
                "pos_raw": float(pos_raw[idx]),
                "pos_filtered": float(pos_filtered[idx]),
            }
        )
    return rows


def _hr_row(
    segment_item: RGBSegment,
    method: str,
    estimate_type: str,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
    values: np.ndarray,
    fps: float,
    config: RPPGConfig,
) -> dict[str, object]:
    # Build one HR estimate row for a segment or window.
    estimate = estimate_heart_rate(
        values,
        fps=fps,
        low_hz=config.bandpass_low_hz,
        high_hz=config.bandpass_high_hz,
        estimator=config.hr_estimator,
    )
    return {
        "segment_id": segment_item.segment_id,
        "roi": segment_item.roi,
        "method": method,
        "estimate_type": estimate_type,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": start_time,
        "end_time_sec": end_time,
        "window_center_time_sec": (start_time + end_time) / 2.0,
        "duration_sec": end_time - start_time + (1.0 / fps),
        "n_samples": len(values),
        **estimate,
    }


def _append_runtime(
    rows: list[dict[str, object]],
    stage: str,
    start_time: float,
) -> None:
    # Append elapsed seconds for one processing stage.
    rows.append(
        {
            "stage": stage,
            "runtime_sec": time.perf_counter() - start_time,
        }
    )


def _safe_std_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    # Compute std ratio while avoiding divide-by-zero.
    denom_std = float(np.std(denominator))
    if denom_std <= 1e-12:
        return 0.0
    return float(np.std(numerator) / denom_std)


def _center_signal(values: np.ndarray) -> np.ndarray:
    # Remove the mean from a one-dimensional signal.
    samples = np.asarray(values, dtype=float)
    return samples - float(np.mean(samples))


def _as_bool_series(series: pd.Series) -> pd.Series:
    # Convert pandas bool or string-bool columns to real booleans.
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _empty_hr_estimate() -> dict[str, float]:
    # Return NaN values when HR estimation is not possible.
    return {
        "hr_bpm": np.nan,
        "peak_frequency_hz": np.nan,
        "peak_power": np.nan,
    }
