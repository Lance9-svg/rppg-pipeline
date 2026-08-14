# Controlled degradation

from __future__ import annotations

import numpy as np
import pandas as pd

FORMAL_CONDITIONS = (
    "original",
    "fps15",
    "fps10",
    "roi_shift_3",
    "roi_shift_5",
)


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
        nearest = int(candidates[np.argmin(np.abs(times[candidates] - target))])
        selected.append(nearest)
        available[nearest] = False
    return np.asarray(selected, dtype=int)


# Calculate whole-window rate
def effective_sample_rate(time_sec: np.ndarray) -> float:
    times = np.asarray(time_sec, dtype=float)
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
        (trace["time_sec"] >= start_time_sec)
        & (trace["time_sec"] < end_time_sec)
    ]
    selected = select_frame_indices(
        window["time_sec"].to_numpy(dtype=float),
        start_time_sec,
        end_time_sec,
        target_fps,
    )
    return window.iloc[selected].reset_index(drop=True)
