"""Reproducibility metadata for formal experiment runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


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
) -> dict[str, object]:
    """Describe the code, inputs, and configuration used for one run."""
    repository_root = Path(_git_output("rev-parse", "--show-toplevel"))
    input_records = []
    for path in inputs:
        resolved = Path(path).resolve(strict=True)
        try:
            display_name = resolved.relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            display_name = resolved.name
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
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "python_version": platform.python_version(),
        "configuration": config,
        "inputs": sorted(input_records, key=lambda item: str(item["name"])),
    }


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
