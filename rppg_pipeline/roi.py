# Fixed landmark ROI masks and quality metrics.

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Fixed MediaPipe landmark IDs for the outer face boundary.
FACE_OVAL_IDS = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
# Fixed landmark IDs excluded from usable skin regions.
RIGHT_EYE_EXCLUDE_IDS = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
# Fixed landmark IDs excluded from usable skin regions.
LEFT_EYE_EXCLUDE_IDS = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
]
# Fixed landmark IDs excluded from usable skin regions.
MOUTH_EXCLUDE_IDS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
]
# Fixed landmark IDs for the forehead ROI hull.
FOREHEAD_IDS = [103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104]
# Fixed landmark IDs for the image-left cheek ROI hull.
LEFT_CHEEK_IDS = [
    234,
    93,
    132,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
    123,
    116,
    111,
    117,
    118,
    119,
    100,
    47,
]
# Fixed landmark IDs for the image-right cheek ROI hull.
RIGHT_CHEEK_IDS = [
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    352,
    345,
    340,
    346,
    347,
    348,
    329,
    277,
]

# ROI names mapped to their fixed landmark hulls.
ROI_LANDMARK_IDS = {
    "forehead": FOREHEAD_IDS,
    "left_cheek": LEFT_CHEEK_IDS,
    "right_cheek": RIGHT_CHEEK_IDS,
}


@dataclass(frozen=True)
class ROIConfig:
    # Scale for shrinking the face oval toward its centroid.
    inner_scale: float = 0.82
    # Pixel dilation applied to eye and mouth exclusion masks.
    exclude_dilate_px: int = 6
    # Minimum final ROI pixels required for validity.
    min_pixels: int = 50
    # Minimum final/base mask area ratio required for validity.
    min_fill_ratio: float = 0.10
    # Margin used to flag faces touching the frame border.
    border_margin_px: int = 2
    # Minimum continuous valid segment length.
    min_valid_segment_duration_sec: float = 10.0
    # Allowed gap between valid frames inside one segment.
    max_gap_frames: int = 0

    def to_dict(self) -> dict[str, object]:
        # Export ROI parameters for run metadata.
        return asdict(self)


@dataclass(frozen=True)
class FaceGeometry:
    # Face bounding box as x_min, y_min, x_max, y_max.
    bbox: tuple[int, int, int, int]
    # Bounding box center in pixel coordinates.
    bbox_center: tuple[float, float]
    # Bounding box area in pixels.
    bbox_area: float
    # Face box area divided by frame area.
    face_area_ratio: float
    # True when the face box is near the frame edge.
    touches_frame_border: bool


@dataclass(frozen=True)
class ROIMeasurement:
    # ROI name such as forehead or cheeks_mean.
    roi: str
    # Mean red channel value inside the ROI.
    r_mean: float
    # Mean green channel value inside the ROI.
    g_mean: float
    # Mean blue channel value inside the ROI.
    b_mean: float
    # Number of pixels used by the final ROI mask.
    n_pixels: int
    # Final ROI mask area divided by base ROI mask area.
    fill_ratio: float
    # Luma-style brightness computed from RGB.
    brightness: float
    # Share of ROI pixels with any channel >= 250.
    overexposure_ratio: float
    # True when the ROI passes hard quality checks.
    roi_valid: bool
    # Semicolon-separated reason string for invalid ROIs.
    invalid_reason: str


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    # OpenCV reads BGR, but rPPG algorithms use RGB order.
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def frame_timestamp(frame_idx: int, fps: float) -> tuple[float, int]:
    # Build a stable constant-frame-rate timebase from frame index.
    if fps <= 0:
        raise ValueError(f"FPS must be positive for video timestamps, got {fps}")
    time_sec = frame_idx / fps
    timestamp_ms = int(frame_idx * 1000 / fps)
    return time_sec, timestamp_ms


def landmark_sets_for_metadata() -> dict[str, list[int]]:
    # Store fixed landmark IDs for reproducibility.
    return {
        "face_oval_ids": FACE_OVAL_IDS,
        "right_eye_exclude_ids": RIGHT_EYE_EXCLUDE_IDS,
        "left_eye_exclude_ids": LEFT_EYE_EXCLUDE_IDS,
        "mouth_exclude_ids": MOUTH_EXCLUDE_IDS,
        "forehead_ids": FOREHEAD_IDS,
        "left_cheek_ids": LEFT_CHEEK_IDS,
        "right_cheek_ids": RIGHT_CHEEK_IDS,
    }


def face_geometry(
    landmarks_px: np.ndarray,
    frame_shape: tuple[int, int],
    config: ROIConfig,
) -> FaceGeometry:
    # Summarize face size, position, and border contact.
    height, width = frame_shape
    valid = landmarks_px[np.isfinite(landmarks_px).all(axis=1)]
    if valid.size == 0:
        return FaceGeometry((0, 0, 0, 0), (np.nan, np.nan), 0.0, 0.0, True)

    x_min = int(np.floor(np.clip(valid[:, 0].min(), 0, width - 1)))
    y_min = int(np.floor(np.clip(valid[:, 1].min(), 0, height - 1)))
    x_max = int(np.ceil(np.clip(valid[:, 0].max(), 0, width - 1)))
    y_max = int(np.ceil(np.clip(valid[:, 1].max(), 0, height - 1)))
    area = float(max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1))
    center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    touches = (
        x_min <= config.border_margin_px
        or y_min <= config.border_margin_px
        or x_max >= width - 1 - config.border_margin_px
        or y_max >= height - 1 - config.border_margin_px
    )
    return FaceGeometry(
        bbox=(x_min, y_min, x_max, y_max),
        bbox_center=center,
        bbox_area=area,
        face_area_ratio=area / float(width * height),
        touches_frame_border=touches,
    )


def create_roi_masks(
    landmarks_px: np.ndarray,
    frame_shape: tuple[int, int],
    config: ROIConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Build full-face and local ROI masks from fixed landmarks.
    height, width = frame_shape
    full_face_inner = _full_face_inner_mask(landmarks_px, (height, width), config)
    masks = {"full_face_inner": full_face_inner}
    base_masks = {"full_face_inner": full_face_inner.copy()}

    for roi_name, landmark_ids in ROI_LANDMARK_IDS.items():
        base = _convex_hull_mask(landmarks_px, landmark_ids, (height, width))
        # bitwise_and intersects each local ROI with full_face_inner.
        final = cv2.bitwise_and(base, full_face_inner)
        masks[roi_name] = final
        base_masks[roi_name] = base

    return masks, base_masks


def measure_roi(
    frame_rgb: np.ndarray,
    roi_name: str,
    final_mask: np.ndarray,
    base_mask: np.ndarray,
    landmark_available: bool,
    face: FaceGeometry,
    config: ROIConfig,
) -> ROIMeasurement:
    # Compute mean RGB and quality values for one ROI mask.
    base_pixels = int(np.count_nonzero(base_mask))
    n_pixels = int(np.count_nonzero(final_mask))
    fill_ratio = n_pixels / base_pixels if base_pixels > 0 else 0.0

    invalid_reasons = []
    if not landmark_available:
        invalid_reasons.append("landmark_unavailable")
    if n_pixels < config.min_pixels:
        invalid_reasons.append("too_few_pixels")
    if fill_ratio < config.min_fill_ratio:
        invalid_reasons.append("low_fill_ratio")
    if face.touches_frame_border:
        invalid_reasons.append("face_touches_border")

    if n_pixels == 0:
        return ROIMeasurement(
            roi=roi_name,
            r_mean=np.nan,
            g_mean=np.nan,
            b_mean=np.nan,
            n_pixels=0,
            fill_ratio=fill_ratio,
            brightness=np.nan,
            overexposure_ratio=np.nan,
            roi_valid=False,
            invalid_reason=";".join(invalid_reasons) or "empty_roi",
        )

    pixels = frame_rgb[final_mask > 0].astype(np.float32)
    # NumPy mean gives channel averages over ROI pixels.
    r_mean = float(np.mean(pixels[:, 0]))
    g_mean = float(np.mean(pixels[:, 1]))
    b_mean = float(np.mean(pixels[:, 2]))
    brightness = float(
        np.mean(0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2])
    )
    overexposure = float(np.mean(np.any(pixels >= 250, axis=1)))
    roi_valid = len(invalid_reasons) == 0
    return ROIMeasurement(
        roi=roi_name,
        r_mean=r_mean,
        g_mean=g_mean,
        b_mean=b_mean,
        n_pixels=n_pixels,
        fill_ratio=fill_ratio,
        brightness=brightness,
        overexposure_ratio=overexposure,
        roi_valid=roi_valid,
        invalid_reason=";".join(invalid_reasons),
    )


def make_invalid_measurement(roi_name: str, reason: str) -> ROIMeasurement:
    # Create a NaN-filled measurement for missing or invalid ROIs.
    return ROIMeasurement(
        roi=roi_name,
        r_mean=np.nan,
        g_mean=np.nan,
        b_mean=np.nan,
        n_pixels=0,
        fill_ratio=0.0,
        brightness=np.nan,
        overexposure_ratio=np.nan,
        roi_valid=False,
        invalid_reason=reason,
    )


def make_cheeks_mean(
    left: ROIMeasurement,
    right: ROIMeasurement,
) -> ROIMeasurement:
    # Average cheeks only when both cheek ROIs are valid.
    if not left.roi_valid or not right.roi_valid:
        return make_invalid_measurement(
            "cheeks_mean",
            "missing_left_or_right_cheek",
        )
    return ROIMeasurement(
        roi="cheeks_mean",
        r_mean=float(np.mean([left.r_mean, right.r_mean])),
        g_mean=float(np.mean([left.g_mean, right.g_mean])),
        b_mean=float(np.mean([left.b_mean, right.b_mean])),
        n_pixels=left.n_pixels + right.n_pixels,
        fill_ratio=float(np.mean([left.fill_ratio, right.fill_ratio])),
        brightness=float(np.mean([left.brightness, right.brightness])),
        overexposure_ratio=float(
            np.mean([left.overexposure_ratio, right.overexposure_ratio])
        ),
        roi_valid=True,
        invalid_reason="",
    )


def _points_for_ids(landmarks_px: np.ndarray, landmark_ids: list[int]) -> np.ndarray:
    # Select finite pixel points for a fixed landmark list.
    points = landmarks_px[np.array(landmark_ids, dtype=np.int32)]
    points = points[np.isfinite(points).all(axis=1)]
    return points


def _polygon_mask(
    points: np.ndarray,
    frame_shape: tuple[int, int],
    convex: bool,
) -> np.ndarray:
    # Rasterize a polygon or convex hull into a binary mask.
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    int_points = np.rint(points).astype(np.int32)
    int_points[:, 0] = np.clip(int_points[:, 0], 0, width - 1)
    int_points[:, 1] = np.clip(int_points[:, 1], 0, height - 1)
    if convex:
        # OpenCV convexHull orders points around the hull boundary.
        hull = cv2.convexHull(int_points)
        # fillConvexPoly fills the ROI mask.
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        # fillPoly keeps the face oval landmark order.
        cv2.fillPoly(mask, [int_points], 255)
    return mask


def _convex_hull_mask(
    landmarks_px: np.ndarray,
    landmark_ids: list[int],
    frame_shape: tuple[int, int],
) -> np.ndarray:
    # Build a convex ROI mask from selected landmarks.
    points = _points_for_ids(landmarks_px, landmark_ids)
    return _polygon_mask(points, frame_shape, convex=True)


def _full_face_inner_mask(
    landmarks_px: np.ndarray,
    frame_shape: tuple[int, int],
    config: ROIConfig,
) -> np.ndarray:
    # Build the inner face mask and remove eyes and mouth.
    face_points = _points_for_ids(landmarks_px, FACE_OVAL_IDS)
    if len(face_points) < 3:
        return np.zeros(frame_shape, dtype=np.uint8)

    centroid = np.mean(face_points, axis=0)
    # Shrink the face oval to avoid hair and background edges.
    inner_points = centroid + config.inner_scale * (face_points - centroid)
    full_mask = _polygon_mask(inner_points, frame_shape, convex=False)

    exclude_mask = np.zeros(frame_shape, dtype=np.uint8)
    for ids in (RIGHT_EYE_EXCLUDE_IDS, LEFT_EYE_EXCLUDE_IDS, MOUTH_EXCLUDE_IDS):
        exclude_mask = cv2.bitwise_or(
            exclude_mask,
            _convex_hull_mask(landmarks_px, ids, frame_shape),
        )

    if config.exclude_dilate_px > 0:
        size = config.exclude_dilate_px * 2 + 1
        # Ellipse kernel expands excluded facial feature masks.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        exclude_mask = cv2.dilate(exclude_mask, kernel)

    return cv2.bitwise_and(full_mask, cv2.bitwise_not(exclude_mask))
