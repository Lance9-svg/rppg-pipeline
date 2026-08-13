# UBFC data and reference windows

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


@dataclass(frozen=True)
class SubjectFiles:
    # Subject files
    name: str
    video_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class GroundTruth:
    # Contact PPG ground truth
    ppg: np.ndarray
    sensor_hr: np.ndarray
    time_sec: np.ndarray


# Find subjects by number
def discover_subjects(dataset_root: str | Path) -> list[SubjectFiles]:
    root = Path(dataset_root)
    subjects: list[tuple[int, SubjectFiles]] = []
    for subject_dir in root.iterdir():
        match = re.fullmatch(r"subject(\d+)", subject_dir.name)
        if not match or not subject_dir.is_dir():
            continue
        video_path = subject_dir / "vid.avi"
        ground_truth_path = subject_dir / "ground_truth.txt"
        if video_path.is_file() and ground_truth_path.is_file():
            subjects.append(
                (
                    int(match.group(1)),
                    SubjectFiles(
                        name=subject_dir.name,
                        video_path=video_path,
                        ground_truth_path=ground_truth_path,
                    ),
                )
            )
    return [subject for _, subject in sorted(subjects)]


# Read three-row ground truth
def load_ground_truth(path: str | Path) -> GroundTruth:
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 3:
        raise ValueError("UBFC ground truth must contain exactly three rows")

    ppg = np.fromstring(lines[0], sep=" ", dtype=float)
    sensor_hr = np.fromstring(lines[1], sep=" ", dtype=float)
    time_sec = np.fromstring(lines[2], sep=" ", dtype=float)
    if len(ppg) == 0 or not (len(ppg) == len(sensor_hr) == len(time_sec)):
        raise ValueError("UBFC ground-truth rows must have equal lengths")
    return GroundTruth(ppg=ppg, sensor_hr=sensor_hr, time_sec=time_sec)


# Build non-overlapping windows
def build_reference_windows(
    subject: str,
    ground_truth: GroundTruth,
) -> pd.DataFrame:
    time_sec = np.asarray(ground_truth.time_sec, dtype=float)
    ppg = np.asarray(ground_truth.ppg, dtype=float)
    sensor_hr = np.asarray(ground_truth.sensor_hr, dtype=float)
    if not (len(time_sec) == len(ppg) == len(sensor_hr)) or len(time_sec) < 3:
        raise ValueError("Ground-truth arrays must have equal usable lengths")

    median_dt = float(np.median(np.diff(time_sec)))
    sample_rate_hz = 1.0 / median_dt
    recording_start = float(time_sec[0])
    recording_end = float(time_sec[-1] + median_dt)
    window_sec = 10.0
    window_count = int(
        np.floor((recording_end - recording_start + median_dt * 0.1) / window_sec)
    )

    corrected_sensor_hr = sensor_hr.copy()
    wrapped = (
        np.isfinite(corrected_sensor_hr)
        & (corrected_sensor_hr > 0.0)
        & (corrected_sensor_hr < 42.0)
    )
    corrected_sensor_hr[wrapped] += 128.0

    rows: list[dict[str, object]] = []
    for index in range(window_count):
        start = recording_start + index * window_sec
        end = start + window_sec
        selected = (time_sec >= start) & (time_sec < end)
        time_hr, frequency_hr = _estimate_contact_hr(
            time_sec[selected],
            ppg[selected],
        )
        sensor_values = corrected_sensor_hr[selected]
        sensor_values = sensor_values[np.isfinite(sensor_values)]
        sensor_median = (
            float(np.median(sensor_values)) if len(sensor_values) else np.nan
        )
        ppg_disagreement = abs(time_hr - frequency_hr)
        sensor_disagreement = abs(time_hr - sensor_median)
        rows.append(
            {
                "subject": subject,
                "window_id": index + 1,
                "start_time_sec": start,
                "end_time_sec": end,
                "window_center_time_sec": start + window_sec / 2.0,
                "duration_sec": window_sec,
                "n_samples": int(np.count_nonzero(selected)),
                "sample_rate_hz": sample_rate_hz,
                "reference_hr_bpm": time_hr,
                "time_hr_bpm": time_hr,
                "frequency_hr_bpm": frequency_hr,
                "time_frequency_diff_bpm": ppg_disagreement,
                "sensor_hr_bpm": sensor_median,
                "reference_sensor_diff_bpm": sensor_disagreement,
                "reference_valid": bool(ppg_disagreement <= 5.0),
            }
        )
    return pd.DataFrame(rows)


# Estimate contact PPG heart rate
def _estimate_contact_hr(
    time_sec: np.ndarray,
    ppg: np.ndarray,
) -> tuple[float, float]:
    sample_rate_hz = 1.0 / float(np.median(np.diff(time_sec)))
    uniform_time = np.arange(
        time_sec[0],
        time_sec[-1] + 0.5 / sample_rate_hz,
        1.0 / sample_rate_hz,
    )
    uniform_ppg = np.interp(uniform_time, time_sec, ppg)
    detrended = signal.detrend(uniform_ppg)
    sos = signal.butter(
        3,
        [0.7, 4.0],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, detrended)
    robust_scale = 1.4826 * float(
        np.median(np.abs(filtered - np.median(filtered)))
    )
    peak_indices, _ = signal.find_peaks(
        filtered,
        distance=max(1, int(sample_rate_hz * 60.0 / 240.0)),
        prominence=0.5 * robust_scale,
    )
    refined_peaks = _refine_peak_locations(filtered, peak_indices)
    intervals_sec = np.diff(refined_peaks) / sample_rate_hz
    intervals_sec = intervals_sec[
        (intervals_sec >= 60.0 / 240.0) & (intervals_sec <= 60.0 / 42.0)
    ]
    time_hr = (
        60.0 / float(np.median(intervals_sec))
        if len(peak_indices) >= 5 and len(intervals_sec) >= 4
        else np.nan
    )

    windowed = (filtered - float(np.mean(filtered))) * np.hanning(len(filtered))
    frequencies, power = signal.periodogram(
        windowed,
        fs=sample_rate_hz,
        window="boxcar",
        detrend=False,
        nfft=4096,
        scaling="density",
    )
    band_indices = np.flatnonzero((frequencies >= 0.7) & (frequencies <= 4.0))
    peak_index = int(band_indices[np.argmax(power[band_indices])])
    peak_frequency = _interpolate_peak(frequencies, power, peak_index)
    return time_hr, peak_frequency * 60.0


# Refine pulse peaks
def _refine_peak_locations(
    values: np.ndarray,
    peak_indices: np.ndarray,
) -> np.ndarray:
    refined = peak_indices.astype(float)
    for output_index, peak_index in enumerate(peak_indices):
        if peak_index <= 0 or peak_index >= len(values) - 1:
            continue
        left = float(values[peak_index - 1])
        center = float(values[peak_index])
        right = float(values[peak_index + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            delta = 0.5 * (left - right) / denominator
            refined[output_index] += float(np.clip(delta, -1.0, 1.0))
    return refined


# Refine spectral peak
def _interpolate_peak(
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
