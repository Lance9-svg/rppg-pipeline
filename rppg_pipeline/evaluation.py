"""Legacy coarse-grid evaluation retained for the bias-audit experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from rppg_pipeline.rppg import bandpass_filter, estimate_heart_rate
from rppg_pipeline.ubfc import UBFCGroundTruth, read_ubfc_ground_truth

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EvaluationConfig:
    # Lower bandpass cutoff in Hz for contact PPG.
    bandpass_low_hz: float = 0.7
    # Upper bandpass cutoff in Hz for contact PPG.
    bandpass_high_hz: float = 4.0
    # Butterworth filter order for contact PPG.
    filter_order: int = 3
    # Spectral estimator used to derive PPG HR.
    hr_estimator: str = "welch"

    def to_dict(self) -> dict[str, object]:
        # Export evaluation parameters.
        return asdict(self)


def process_ubfc_evaluation(
    out_dir: str | Path,
    ground_truth_path: str | Path,
    fps: float,
    config: EvaluationConfig | None = None,
) -> None:
    # Evaluate existing rPPG HR results against UBFC contact PPG.
    config = config or EvaluationConfig()
    if fps <= 0:
        raise ValueError(f"FPS must be positive for evaluation, got {fps}")

    out_path = Path(out_dir)
    hr_path = out_path / "hr_results.csv"
    if not hr_path.exists():
        raise FileNotFoundError(f"Missing rPPG HR results: {hr_path}")

    hr_df = pd.read_csv(hr_path)
    ground_truth = read_ubfc_ground_truth(ground_truth_path)
    _validate_ground_truth_length(ground_truth, hr_df)

    ppg_hr_df = build_ppg_hr_results(hr_df, ground_truth, fps, config)
    evaluation_df = build_evaluation_results(hr_df, ppg_hr_df)
    metrics_df = build_evaluation_metrics(evaluation_df)

    ppg_hr_df.to_csv(out_path / "ppg_hr_results.csv", index=False)
    evaluation_df.to_csv(out_path / "evaluation_results.csv", index=False)
    metrics_df.to_csv(out_path / "evaluation_metrics.csv", index=False)
    plot_evaluation(evaluation_df, out_path / "evaluation_plot.png")


def build_ppg_hr_results(
    hr_df: pd.DataFrame,
    ground_truth: UBFCGroundTruth,
    fps: float,
    config: EvaluationConfig,
) -> pd.DataFrame:
    # Derive contact-PPG HR for each unique rPPG frame window.
    rows = []
    unique_windows = (
        hr_df[
            [
                "estimate_type",
                "start_frame",
                "end_frame",
                "start_time_sec",
                "end_time_sec",
                "window_center_time_sec",
            ]
        ]
        .drop_duplicates()
        .sort_values(["estimate_type", "start_frame", "end_frame"])
    )
    for window in unique_windows.itertuples(index=False):
        start_frame = int(window.start_frame)
        end_frame = int(window.end_frame)
        ppg_slice = ground_truth.ppg_trace[start_frame : end_frame + 1]
        sensor_slice = ground_truth.sensor_hr[start_frame : end_frame + 1]
        ppg_filtered = bandpass_filter(
            ppg_slice,
            fps=fps,
            low_hz=config.bandpass_low_hz,
            high_hz=config.bandpass_high_hz,
            order=config.filter_order,
        )
        estimate = estimate_heart_rate(
            ppg_filtered,
            fps=fps,
            low_hz=config.bandpass_low_hz,
            high_hz=config.bandpass_high_hz,
            estimator=config.hr_estimator,
        )
        rows.append(
            {
                "estimate_type": window.estimate_type,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": float(window.start_time_sec),
                "end_time_sec": float(window.end_time_sec),
                "window_center_time_sec": float(window.window_center_time_sec),
                "n_samples": len(ppg_slice),
                "ppg_hr_bpm": estimate["hr_bpm"],
                "ppg_peak_frequency_hz": estimate["peak_frequency_hz"],
                "ppg_peak_power": estimate["peak_power"],
                "sensor_hr_mean_bpm": float(np.mean(sensor_slice)),
                "sensor_hr_median_bpm": float(np.median(sensor_slice)),
                "gt_time_start_sec": float(ground_truth.timestamp_sec[start_frame]),
                "gt_time_end_sec": float(ground_truth.timestamp_sec[end_frame]),
                "evaluation_fps": fps,
                **config.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def build_evaluation_results(
    hr_df: pd.DataFrame,
    ppg_hr_df: pd.DataFrame,
) -> pd.DataFrame:
    # Join rPPG HR rows with matching PPG-derived HR rows.
    join_cols = ["estimate_type", "start_frame", "end_frame"]
    merged = hr_df.merge(
        ppg_hr_df,
        on=join_cols,
        how="inner",
        suffixes=("_rppg", "_ppg"),
    )
    merged = merged.rename(
        columns={
            "method": "rppg_method",
            "hr_bpm": "rppg_hr_bpm",
            "peak_frequency_hz": "rppg_peak_frequency_hz",
            "peak_power": "rppg_peak_power",
        }
    )
    merged["error_bpm"] = merged["rppg_hr_bpm"] - merged["ppg_hr_bpm"]
    merged["abs_error_bpm"] = np.abs(merged["error_bpm"])
    merged["squared_error_bpm"] = merged["error_bpm"] ** 2
    merged["sensor_error_bpm"] = (
        merged["rppg_hr_bpm"] - merged["sensor_hr_mean_bpm"]
    )
    merged["sensor_abs_error_bpm"] = np.abs(merged["sensor_error_bpm"])
    return merged[
        [
            "rppg_method",
            "estimate_type",
            "start_frame",
            "end_frame",
            "start_time_sec_rppg",
            "end_time_sec_rppg",
            "window_center_time_sec_rppg",
            "rppg_hr_bpm",
            "ppg_hr_bpm",
            "sensor_hr_mean_bpm",
            "sensor_hr_median_bpm",
            "error_bpm",
            "abs_error_bpm",
            "squared_error_bpm",
            "sensor_error_bpm",
            "sensor_abs_error_bpm",
            "rppg_peak_frequency_hz",
            "ppg_peak_frequency_hz",
            "rppg_peak_power",
            "ppg_peak_power",
            "n_samples_rppg",
            "n_samples_ppg",
            "gt_time_start_sec",
            "gt_time_end_sec",
            "evaluation_fps",
        ]
    ]


def build_evaluation_metrics(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    # Summarize HR errors by method and estimate type.
    rows = []
    for (method, estimate_type), group in evaluation_df.groupby(
        ["rppg_method", "estimate_type"]
    ):
        errors = group["error_bpm"].to_numpy(dtype=float)
        abs_errors = group["abs_error_bpm"].to_numpy(dtype=float)
        squared_errors = group["squared_error_bpm"].to_numpy(dtype=float)
        rppg_hr = group["rppg_hr_bpm"].to_numpy(dtype=float)
        ppg_hr = group["ppg_hr_bpm"].to_numpy(dtype=float)
        rows.append(
            {
                "rppg_method": method,
                "estimate_type": estimate_type,
                "n_windows": len(group),
                "mae_bpm": float(np.mean(abs_errors)),
                "rmse_bpm": float(np.sqrt(np.mean(squared_errors))),
                "bias_bpm": float(np.mean(errors)),
                "median_abs_error_bpm": float(np.median(abs_errors)),
                "max_abs_error_bpm": float(np.max(abs_errors)),
                "correlation": _correlation(rppg_hr, ppg_hr),
                "sensor_mae_bpm": float(np.mean(group["sensor_abs_error_bpm"])),
                "mean_rppg_hr_bpm": float(np.mean(rppg_hr)),
                "mean_ppg_hr_bpm": float(np.mean(ppg_hr)),
            }
        )
    return pd.DataFrame(rows)


def plot_evaluation(evaluation_df: pd.DataFrame, out_path: str | Path) -> None:
    # Plot rPPG HR curves against PPG-derived ground truth.
    curve_df = evaluation_df[evaluation_df["estimate_type"] == "window"].copy()
    if curve_df.empty:
        curve_df = evaluation_df[evaluation_df["estimate_type"] == "segment"].copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ppg_curve = (
        curve_df[
            [
                "window_center_time_sec_rppg",
                "ppg_hr_bpm",
            ]
        ]
        .drop_duplicates()
        .sort_values("window_center_time_sec_rppg")
    )
    ax.plot(
        ppg_curve["window_center_time_sec_rppg"],
        ppg_curve["ppg_hr_bpm"],
        color="black",
        linewidth=2.0,
        label="PPG ground truth",
    )
    for method, group in curve_df.groupby("rppg_method"):
        group = group.sort_values("window_center_time_sec_rppg")
        ax.plot(
            group["window_center_time_sec_rppg"],
            group["rppg_hr_bpm"],
            marker="o",
            linewidth=1.4,
            markersize=3,
            label=method,
        )
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.set_title("rPPG vs UBFC PPG Ground Truth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _validate_ground_truth_length(
    ground_truth: UBFCGroundTruth,
    hr_df: pd.DataFrame,
) -> None:
    # Ensure frame-index windows can index the UBFC arrays.
    max_frame = int(hr_df["end_frame"].max())
    sample_count = len(ground_truth.ppg_trace)
    if sample_count <= max_frame:
        raise ValueError(
            f"Ground truth has {sample_count} samples but needs frame {max_frame}"
        )
    if len(ground_truth.sensor_hr) != sample_count:
        raise ValueError("PPG and sensor HR sample counts differ")
    if len(ground_truth.timestamp_sec) != sample_count:
        raise ValueError("PPG and timestamp sample counts differ")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    # Compute Pearson correlation when at least two varying points exist.
    if len(left) < 2 or len(right) < 2:
        return np.nan
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])
