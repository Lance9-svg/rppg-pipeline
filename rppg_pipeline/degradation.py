# Controlled degradation

from __future__ import annotations

import numpy as np
import pandas as pd

ROIS = ("full_face_inner", "forehead", "cheeks_mean")
CONDITION_SETTINGS = {
    "original": ("none", np.nan, np.nan),
    "fps15": ("frame_rate", 15.0, 15.0),
    "fps10": ("frame_rate", 10.0, 10.0),
    "roi_shift_3": ("roi_shift", 0.03, np.nan),
    "roi_shift_5": ("roi_shift", 0.05, np.nan),
}
FORMAL_CONDITIONS = tuple(CONDITION_SETTINGS)


# Select nearest source frames
def select_frame_indices(
    time_sec: np.ndarray,
    start_time_sec: float,
    end_time_sec: float,
    target_fps: float,
) -> np.ndarray:
    times = np.asarray(time_sec, dtype=float)
    count = int(round((end_time_sec - start_time_sec) * target_fps))
    targets = start_time_sec + np.arange(count) / target_fps
    available = np.ones(len(times), dtype=bool)
    selected = []
    for target in targets:
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        nearest = int(candidates[np.argmin(np.abs(times[candidates] - target))])
        selected.append(nearest)
        available[nearest] = False
    return np.asarray(selected, dtype=int)


# Calculate whole-window rate
def effective_sample_rate(time_sec: np.ndarray) -> float:
    times = np.asarray(time_sec, dtype=float)
    if len(times) < 2:
        return np.nan
    return float((len(times) - 1) / (times[-1] - times[0]))


# Build periodic ROI offsets
def roi_shift_offsets(
    time_sec: np.ndarray,
    face_width_px: np.ndarray,
    shift_fraction: float,
) -> np.ndarray:
    times = np.asarray(time_sec, dtype=float)
    widths = np.asarray(face_width_px, dtype=float)
    phase = 2.0 * np.pi * np.mod(times, 10.0) / 10.0
    unit_offsets = np.column_stack(
        [
            np.sin(phase),
            0.5 * np.sin(2.0 * phase),
        ]
    )
    return unit_offsets * widths[:, np.newaxis] * shift_fraction


# Downsample one window
def downsample_window(
    trace: pd.DataFrame,
    start_time_sec: float,
    end_time_sec: float,
    target_fps: float,
) -> pd.DataFrame:
    window = trace[
        (trace["time_sec"] >= start_time_sec) & (trace["time_sec"] < end_time_sec)
    ]
    selected = select_frame_indices(
        window["time_sec"].to_numpy(dtype=float),
        start_time_sec,
        end_time_sec,
        target_fps,
    )
    return window.iloc[selected].reset_index(drop=True)


# Downsample all reference windows
def downsample_trace(
    trace: pd.DataFrame,
    references: pd.DataFrame,
    target_fps: float,
) -> pd.DataFrame:
    windows = [
        downsample_window(
            trace,
            float(reference.start_time_sec),
            float(reference.end_time_sec),
            target_fps,
        )
        for reference in references.itertuples(index=False)
    ]
    return pd.concat(windows, ignore_index=True)
