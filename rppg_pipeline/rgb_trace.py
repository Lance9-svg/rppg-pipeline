# Single-video ROI extraction and RGB trace export.

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rppg_pipeline.face_landmarks import (
    FaceLandmarkDetector,
    landmarks_to_pixels,
)
from rppg_pipeline.roi import (
    ROIConfig,
    ROIMeasurement,
    bgr_to_rgb,
    create_roi_masks,
    face_geometry,
    frame_timestamp,
    make_cheeks_mean,
    make_invalid_measurement,
    measure_roi,
)
from rppg_pipeline.video import VideoMetadata

# Fixed output order for all ROI rows and wide columns.
ROI_OUTPUT_ORDER = [
    "full_face_inner",
    "forehead",
    "left_cheek",
    "right_cheek",
    "cheeks_mean",
]


@dataclass(frozen=True)
class FrameDebugData:
    # Pixel landmarks used for drawing debug overlays.
    landmarks_px: np.ndarray | None
    # Face box used for debug visualization.
    bbox: tuple[int, int, int, int] | None
    # True when MediaPipe returned landmarks for the frame.
    landmark_available: bool


def process_video_rgb_trace(
    video_path: str | Path,
    out_dir: str | Path,
    model_path: str | Path,
    metadata: VideoMetadata,
    debug_every: int = 150,
    max_frames: int | None = None,
    config: ROIConfig | None = None,
) -> None:
    # Extract ROI RGB traces and quality outputs from one video.
    config = config or ROIConfig()
    if metadata.fps <= 0:
        raise ValueError("ROI processing requires fps > 0")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    debug_dir = out_path / "roi_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Long rows store one frame and one ROI per row.
    rows: list[dict[str, object]] = []
    # Frame debug data is cached for selected overlay images.
    frame_debug: dict[int, FrameDebugData] = {}
    # Per-frame metrics help choose useful debug frames.
    frame_metrics: list[dict[str, object]] = []

    # OpenCV VideoCapture decodes video frames sequentially.
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    prev_center: tuple[float, float] | None = None
    prev_area: float | None = None

    try:
        # Context manager closes the MediaPipe task after processing.
        with FaceLandmarkDetector(model_path) as detector:
            frame_idx = 0
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                # capture.read returns one BGR frame from OpenCV.
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                # Convert BGR to RGB before landmark and rPPG processing.
                frame_rgb = bgr_to_rgb(frame_bgr)
                height, width = frame_rgb.shape[:2]
                time_sec, timestamp_ms = frame_timestamp(frame_idx, metadata.fps)
                # MediaPipe detects landmarks using the video timestamp.
                result = detector.detect(frame_rgb, timestamp_ms)
                landmark_available = bool(result.face_landmarks)

                if landmark_available:
                    # Convert MediaPipe normalized landmarks to pixels.
                    landmarks_px = landmarks_to_pixels(
                        result.face_landmarks[0],
                        width,
                        height,
                    )
                    # Compute face geometry for quality metrics.
                    face = face_geometry(landmarks_px, (height, width), config)
                    center_jump = _center_jump(
                        face.bbox_center,
                        prev_center,
                        width,
                        height,
                    )
                    area_change = _area_change(face.bbox_area, prev_area)
                    prev_center = face.bbox_center
                    prev_area = face.bbox_area

                    # Create final ROI masks and their base hull masks.
                    masks, base_masks = create_roi_masks(
                        landmarks_px,
                        (height, width),
                        config,
                    )
                    # Measure RGB and quality metrics for each base ROI.
                    measurements = {
                        roi_name: measure_roi(
                            frame_rgb=frame_rgb,
                            roi_name=roi_name,
                            final_mask=masks[roi_name],
                            base_mask=base_masks[roi_name],
                            landmark_available=True,
                            face=face,
                            config=config,
                        )
                        for roi_name in (
                            "full_face_inner",
                            "forehead",
                            "left_cheek",
                            "right_cheek",
                        )
                    }
                    # cheeks_mean requires both cheeks to be valid.
                    measurements["cheeks_mean"] = make_cheeks_mean(
                        measurements["left_cheek"],
                        measurements["right_cheek"],
                    )
                    frame_debug[frame_idx] = FrameDebugData(
                        landmarks_px=landmarks_px,
                        bbox=face.bbox,
                        landmark_available=True,
                    )
                else:
                    # Missing landmarks make every ROI invalid for this frame.
                    face = _empty_face_geometry()
                    center_jump = np.nan
                    area_change = np.nan
                    measurements = {
                        roi_name: make_invalid_measurement(
                            roi_name,
                            "landmark_unavailable",
                        )
                        for roi_name in ROI_OUTPUT_ORDER
                    }
                    frame_debug[frame_idx] = FrameDebugData(
                        landmarks_px=None,
                        bbox=None,
                        landmark_available=False,
                    )

                # Store one long-format row per ROI.
                for roi_name in ROI_OUTPUT_ORDER:
                    rows.append(
                        _row_from_measurement(
                            frame_idx=frame_idx,
                            time_sec=time_sec,
                            timestamp_ms=timestamp_ms,
                            measurement=measurements[roi_name],
                            landmark_available=landmark_available,
                            face=face,
                            bbox_center_jump=center_jump,
                            bbox_area_change=area_change,
                        )
                    )

                # Store metrics used for choosing debug images.
                frame_metrics.append(
                    {
                        "frame_idx": frame_idx,
                        "landmark_available": landmark_available,
                        "bbox_center_jump": center_jump,
                        "cheeks_brightness": measurements[
                            "cheeks_mean"
                        ].brightness,
                    }
                )
                frame_idx += 1
    finally:
        capture.release()

    # Pandas DataFrame stores all ROI measurements for export.
    trace_long_df = pd.DataFrame(rows)
    trace_wide_df = build_wide_rgb_trace(trace_long_df)
    # Wide table is one row per frame for downstream algorithms.
    trace_wide_df.to_csv(out_path / "rgb_trace.csv", index=False)
    # Long table is one row per frame per ROI for diagnostics.
    trace_long_df.to_csv(out_path / "rgb_trace_long.csv", index=False)

    # Valid segments mark continuous usable chunks for CHROM/POS.
    segments_df = build_valid_segments(trace_long_df, metadata.fps, config)
    segments_df.to_csv(out_path / "valid_segments.csv", index=False)

    # Pick fixed and quality-driven frames for visual inspection.
    debug_indices = _select_debug_indices(
        frame_count=int(trace_long_df["frame_idx"].max() + 1)
        if len(trace_long_df)
        else 0,
        debug_every=debug_every,
        frame_metrics=frame_metrics,
        segments_df=segments_df,
    )
    _write_debug_frames(
        video_path=video_path,
        debug_dir=debug_dir,
        debug_indices=debug_indices,
        frame_debug=frame_debug,
        config=config,
    )

    # Keep one concise trace-quality summary for local visual verification.
    quality_summary = build_quality_summary(trace_long_df, segments_df, debug_indices)
    _write_json(out_path / "quality_summary.json", quality_summary)


def build_valid_segments(
    trace_df: pd.DataFrame,
    fps: float,
    config: ROIConfig,
) -> pd.DataFrame:
    # Convert per-frame ROI validity into continuous valid segments.
    min_frames = int(np.ceil(fps * config.min_valid_segment_duration_sec))
    segments = []
    segment_id = 0
    for roi_name, roi_df in trace_df.groupby("roi", sort=False):
        # Pandas groupby processes one ROI at a time.
        roi_df = roi_df.sort_values("frame_idx")
        active_start = None
        previous_frame = None
        previous_time = None

        for row in roi_df.itertuples(index=False):
            valid = bool(row.roi_valid)
            frame_idx = int(row.frame_idx)
            time_sec = float(row.time_sec)
            if valid and active_start is None:
                active_start = (frame_idx, time_sec)
            if not valid and active_start is not None:
                segment_id += 1
                _append_segment(
                    segments,
                    segment_id,
                    roi_name,
                    active_start,
                    previous_frame,
                    previous_time,
                    fps,
                    min_frames,
                    str(row.invalid_reason),
                )
                active_start = None
            previous_frame = frame_idx
            previous_time = time_sec

        if active_start is not None and previous_frame is not None:
            segment_id += 1
            _append_segment(
                segments,
                segment_id,
                roi_name,
                active_start,
                previous_frame,
                previous_time,
                fps,
                min_frames,
                "end_of_video",
            )

    return pd.DataFrame(
        segments,
        columns=[
            "segment_id",
            "roi",
            "start_frame",
            "end_frame",
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
            "n_frames",
            "is_long_enough",
            "invalid_reason_after_segment",
        ],
    )


def build_wide_rgb_trace(trace_long_df: pd.DataFrame) -> pd.DataFrame:
    # Convert long ROI rows to a one-row-per-frame table.
    frame_columns = [
        "frame_idx",
        "time_sec",
        "timestamp_ms",
        "landmark_available",
        "face_area_ratio",
        "touches_frame_border",
        "bbox_center_x",
        "bbox_center_y",
        "bbox_center_jump",
        "bbox_area_change",
    ]
    roi_columns = [
        "r_mean",
        "g_mean",
        "b_mean",
        "n_pixels",
        "fill_ratio",
        "brightness",
        "overexposure_ratio",
        "roi_valid",
        "invalid_reason",
    ]
    if trace_long_df.empty:
        return pd.DataFrame(columns=frame_columns)

    wide = (
        trace_long_df[frame_columns]
        # One shared frame metadata row is kept per frame.
        .drop_duplicates("frame_idx")
        .sort_values("frame_idx")
        .reset_index(drop=True)
    )
    for roi_name in ROI_OUTPUT_ORDER:
        roi_df = trace_long_df[trace_long_df["roi"] == roi_name]
        roi_df = roi_df[["frame_idx", *roi_columns]].copy()
        # Prefix ROI columns to avoid name collisions in wide format.
        roi_df = roi_df.rename(
            columns={column: f"{roi_name}_{column}" for column in roi_columns}
        )
        # Merge adds the ROI columns onto the frame-level table.
        wide = wide.merge(roi_df, on="frame_idx", how="left")
    return wide


def build_quality_summary(
    trace_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    debug_indices: list[int],
) -> dict[str, object]:
    # Build compact quality statistics for one processed video.
    frame_df = trace_df.drop_duplicates("frame_idx")
    landmark_available_rate = float(frame_df["landmark_available"].mean())

    valid_rate = (
        # Mean of boolean roi_valid gives valid-frame rate.
        trace_df.groupby("roi")["roi_valid"]
        .mean()
        .fillna(0.0)
        .astype(float)
        .to_dict()
    )
    longest_segments = (
        segments_df.groupby("roi")["duration_sec"].max().fillna(0.0).to_dict()
        if len(segments_df)
        else {}
    )
    long_enough_counts = (
        segments_df[segments_df["is_long_enough"]]
        .groupby("roi")["segment_id"]
        .count()
        .to_dict()
        if len(segments_df)
        else {}
    )
    median_pixels = (
        trace_df.groupby("roi")["n_pixels"].median().fillna(0).astype(float).to_dict()
    )
    invalid_counts = {
        roi: Counter(
            # Count each invalid reason token per ROI.
            reason
            for reasons in group["invalid_reason"].dropna().astype(str)
            for reason in reasons.split(";")
            if reason
        )
        for roi, group in trace_df.groupby("roi")
    }

    return {
        "landmark_available_rate": landmark_available_rate,
        "valid_frame_rate_by_roi": valid_rate,
        "longest_valid_segment_sec_by_roi": longest_segments,
        "long_enough_segment_count_by_roi": long_enough_counts,
        "mean_face_area_ratio": float(frame_df["face_area_ratio"].mean()),
        "median_roi_pixels_by_roi": median_pixels,
        "invalid_reason_counts_by_roi": {
            roi: dict(counts) for roi, counts in invalid_counts.items()
        },
        "debug_frame_indices": debug_indices,
    }


def _row_from_measurement(
    frame_idx: int,
    time_sec: float,
    timestamp_ms: int,
    measurement: ROIMeasurement,
    landmark_available: bool,
    face,
    bbox_center_jump: float,
    bbox_area_change: float,
) -> dict[str, object]:
    # Flatten one ROIMeasurement plus frame metrics into a CSV row.
    return {
        "frame_idx": frame_idx,
        "time_sec": time_sec,
        "timestamp_ms": timestamp_ms,
        "roi": measurement.roi,
        "r_mean": measurement.r_mean,
        "g_mean": measurement.g_mean,
        "b_mean": measurement.b_mean,
        "n_pixels": measurement.n_pixels,
        "fill_ratio": measurement.fill_ratio,
        "brightness": measurement.brightness,
        "overexposure_ratio": measurement.overexposure_ratio,
        "landmark_available": landmark_available,
        "roi_valid": measurement.roi_valid,
        "invalid_reason": measurement.invalid_reason,
        "face_area_ratio": face.face_area_ratio,
        "touches_frame_border": face.touches_frame_border,
        "bbox_center_x": face.bbox_center[0],
        "bbox_center_y": face.bbox_center[1],
        "bbox_center_jump": bbox_center_jump,
        "bbox_area_change": bbox_area_change,
    }


def _append_segment(
    segments: list[dict[str, object]],
    segment_id: int,
    roi_name: str,
    active_start: tuple[int, float],
    end_frame: int,
    end_time: float,
    fps: float,
    min_frames: int,
    invalid_reason: str,
) -> None:
    # Append one continuous valid segment record.
    start_frame, start_time = active_start
    n_frames = end_frame - start_frame + 1
    duration = n_frames / fps
    segments.append(
        {
            "segment_id": segment_id,
            "roi": roi_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_sec": start_time,
            "end_time_sec": end_time,
            "duration_sec": duration,
            "n_frames": n_frames,
            "is_long_enough": n_frames >= min_frames,
            "invalid_reason_after_segment": invalid_reason,
        }
    )


def _empty_face_geometry():
    # Return invalid geometry when no landmarks are available.
    from rppg_pipeline.roi import FaceGeometry

    return FaceGeometry(
        bbox=(0, 0, 0, 0),
        bbox_center=(np.nan, np.nan),
        bbox_area=0.0,
        face_area_ratio=0.0,
        touches_frame_border=True,
    )


def _center_jump(
    current_center: tuple[float, float],
    previous_center: tuple[float, float] | None,
    width: int,
    height: int,
) -> float:
    # Measure normalized face center movement between frames.
    if previous_center is None:
        return np.nan
    if not np.isfinite(current_center).all() or not np.isfinite(previous_center).all():
        return np.nan
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    return float(np.sqrt(dx * dx + dy * dy) / np.sqrt(width * height))


def _area_change(current_area: float, previous_area: float | None) -> float:
    # Measure relative face-box area change between frames.
    if previous_area is None or previous_area <= 0:
        return np.nan
    return float(abs(current_area - previous_area) / previous_area)


def _select_debug_indices(
    frame_count: int,
    debug_every: int,
    frame_metrics: list[dict[str, object]],
    segments_df: pd.DataFrame,
) -> list[int]:
    # Select debug frames from fixed intervals and quality extremes.
    if frame_count <= 0:
        return []
    indices = {0, frame_count - 1}
    if debug_every > 0:
        # Include periodic debug frames.
        indices.update(range(0, frame_count, debug_every))

    # Include the first frame where landmarks are missing.
    first_missing = next(
        (
            int(item["frame_idx"])
            for item in frame_metrics
            if not bool(item["landmark_available"])
        ),
        None,
    )
    if first_missing is not None:
        indices.add(first_missing)

    finite_jumps = [
        item for item in frame_metrics if np.isfinite(item["bbox_center_jump"])
    ]
    if finite_jumps:
        # Include the frame with largest head-motion jump.
        max_jump = max(finite_jumps, key=lambda item: item["bbox_center_jump"])
        indices.add(int(max_jump["frame_idx"]))

    finite_brightness = [
        item for item in frame_metrics if np.isfinite(item["cheeks_brightness"])
    ]
    if finite_brightness:
        # Include the darkest cheeks_mean frame.
        indices.add(
            int(
                min(
                    finite_brightness,
                    key=lambda item: item["cheeks_brightness"],
                )["frame_idx"]
            )
        )

    cheeks = segments_df[segments_df["roi"] == "cheeks_mean"]
    if len(cheeks):
        # Include one frame from the longest usable cheeks segment.
        longest = cheeks.sort_values("duration_sec", ascending=False).iloc[0]
        middle = int((int(longest["start_frame"]) + int(longest["end_frame"])) // 2)
        indices.add(middle)

    return sorted(idx for idx in indices if 0 <= idx < frame_count)


def _write_debug_frames(
    video_path: str | Path,
    debug_dir: Path,
    debug_indices: list[int],
    frame_debug: dict[int, FrameDebugData],
    config: ROIConfig,
) -> None:
    # Render selected frames with ROI overlays.
    if not debug_indices:
        return
    wanted = set(debug_indices)
    # Re-open the video so debug images match original frames.
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return
    try:
        frame_idx = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_idx in wanted:
                frame_rgb = bgr_to_rgb(frame_bgr)
                debug = frame_debug.get(frame_idx)
                rendered = _render_debug_frame(frame_rgb, debug, config)
                # Convert RGB back to BGR before cv2.imwrite.
                out_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(debug_dir / f"frame_{frame_idx:06d}.jpg"), out_bgr)
            frame_idx += 1
            if frame_idx > max(wanted):
                break
    finally:
        capture.release()


def _render_debug_frame(
    frame_rgb: np.ndarray,
    debug: FrameDebugData | None,
    config: ROIConfig,
) -> np.ndarray:
    # Draw ROI masks, landmarks, and face box on one RGB frame.
    image = frame_rgb.copy()
    if debug is None or debug.landmarks_px is None:
        cv2.putText(
            image,
            "landmarks unavailable",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
        return image

    height, width = image.shape[:2]
    masks, _ = create_roi_masks(debug.landmarks_px, (height, width), config)
    colors = {
        "full_face_inner": (0, 255, 255),
        "forehead": (255, 128, 0),
        "left_cheek": (0, 255, 0),
        "right_cheek": (0, 128, 255),
    }
    overlay = image.copy()
    for roi_name, color in colors.items():
        overlay[masks[roi_name] > 0] = color
    # addWeighted blends the ROI overlay with the original frame.
    image = cv2.addWeighted(overlay, 0.30, image, 0.70, 0)

    for x, y in debug.landmarks_px[::8]:
        if np.isfinite([x, y]).all():
            # Draw a sparse subset of landmarks for readability.
            cv2.circle(image, (int(round(x)), int(round(y))), 1, (255, 255, 255), -1)

    if debug.bbox is not None:
        x_min, y_min, x_max, y_max = debug.bbox
        # Draw face bounding box.
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

    y_pos = 24
    for roi_name, color in colors.items():
        cv2.putText(
            image,
            roi_name,
            (16, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        y_pos += 24
    return image


def _write_json(path: Path, payload: dict[str, object]) -> None:
    # Write a readable JSON file.
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
