"""Reproducibility metadata for formal experiment runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RUNTIME_PACKAGES = (
    "mediapipe",
    "numpy",
    "opencv-contrib-python",
    "pandas",
    "scipy",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    stage: str,
    inputs: list[Path],
    config: dict[str, object],
    *,
    repository_root: Path,
    input_root: Path | None = None,
) -> dict[str, object]:
    """Describe the code, inputs, and configuration used for one run."""
    requested_root = Path(repository_root).resolve(strict=True)
    repository_root = Path(
        _git_output(requested_root, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    resolved_input_root = Path(input_root).resolve() if input_root else None
    input_records = []
    for path in inputs:
        resolved = Path(path).resolve(strict=True)
        display_name = _portable_input_name(
            resolved,
            repository_root.resolve(),
            resolved_input_root,
        )
        input_records.append(
            {
                "name": display_name,
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )

    return {
        "schema_version": 1,
        "stage": stage,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_output(repository_root, "status", "--porcelain")),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "configuration": config,
        "inputs": sorted(input_records, key=lambda item: str(item["name"])),
    }


def _portable_input_name(
    path: Path,
    repository_root: Path,
    input_root: Path | None,
) -> str:
    if input_root is not None:
        try:
            return path.relative_to(input_root).as_posix()
        except ValueError:
            pass
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.name


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_output(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
