"""UBFC dataset access used by the current research pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUBJECT_DIR_PATTERN = re.compile(r"^subject([1-9]\d*)$")


@dataclass(frozen=True)
class SubjectDataset:
    subject_id: int
    name: str
    directory: Path

    @property
    def video_path(self) -> Path:
        return self.directory / "vid.avi"

    @property
    def ground_truth_path(self) -> Path:
        return self.directory / "ground_truth.txt"


@dataclass(frozen=True)
class UBFCGroundTruth:
    ppg_trace: np.ndarray
    sensor_hr: np.ndarray
    timestamp_sec: np.ndarray


def discover_subjects(dataset_root: str | Path) -> list[SubjectDataset]:
    """Return exact subject<number> folders in original numeric order."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {root}")

    subjects = []
    for directory in root.iterdir():
        match = SUBJECT_DIR_PATTERN.fullmatch(directory.name)
        if directory.is_dir() and match:
            subjects.append(
                SubjectDataset(
                    subject_id=int(match.group(1)),
                    name=directory.name,
                    directory=directory,
                )
            )
    return sorted(subjects, key=lambda subject: subject.subject_id)


def select_subjects(
    subjects: Sequence[SubjectDataset],
    selected_ids: Sequence[int] | None,
) -> list[SubjectDataset]:
    """Select original subject IDs without renumbering them."""
    if not selected_ids:
        return list(subjects)
    wanted = set(selected_ids)
    available = {subject.subject_id for subject in subjects}
    missing = sorted(wanted - available)
    if missing:
        raise ValueError(f"Subject IDs not found: {missing}")
    return [subject for subject in subjects if subject.subject_id in wanted]


def read_ubfc_ground_truth(path: str | Path) -> UBFCGroundTruth:
    """Read contact PPG, stored sensor HR, and timestamps."""
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("UBFC ground truth must have shape (3, n_samples)")
    return UBFCGroundTruth(
        ppg_trace=values[0].astype(float),
        sensor_hr=values[1].astype(float),
        timestamp_sec=values[2].astype(float),
    )
