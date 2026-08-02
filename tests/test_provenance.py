from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pytest

from rppg_pipeline.provenance import build_run_manifest, sha256_file


def test_sha256_file_reflects_file_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"abc")

    first_hash = sha256_file(artifact)
    artifact.write_bytes(b"abd")
    second_hash = sha256_file(artifact)

    assert first_hash == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    assert second_hash != first_hash


def test_sha256_file_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.csv")


def test_build_run_manifest_records_reproducible_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _initialise_git_repository(tmp_path)
    first = repo / "inputs" / "a.csv"
    second = repo / "inputs" / "b.csv"
    first.parent.mkdir()
    first.write_bytes(b"value\n1\n")
    second.write_bytes(b"value\n2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add inputs")
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.chdir(repo)

    manifest = build_run_manifest(
        "phase3",
        [second.resolve(), first.resolve()],
        {"methods": ["CHROM", "POS"], "threshold_bpm": 5.0},
        repository_root=repo,
    )

    created_at = datetime.fromisoformat(str(manifest["created_at_utc"]))
    assert created_at.tzinfo == UTC
    assert manifest["schema_version"] == 1
    assert manifest["stage"] == "phase3"
    assert manifest["git_commit"] == commit
    assert manifest["git_dirty"] is False
    assert str(manifest["python_version"]).startswith("3.")
    assert manifest["package_versions"] == {
        name: version(name)
        for name in (
            "mediapipe",
            "numpy",
            "opencv-contrib-python",
            "pandas",
            "scipy",
        )
    }
    assert manifest["configuration"] == {
        "methods": ["CHROM", "POS"],
        "threshold_bpm": 5.0,
    }
    assert [item["name"] for item in manifest["inputs"]] == [
        "inputs/a.csv",
        "inputs/b.csv",
    ]
    assert manifest["inputs"][0]["size_bytes"] == len(b"value\n1\n")
    assert manifest["inputs"][0]["sha256"] == sha256_file(first)
    assert str(repo.resolve()) not in str(manifest)


def test_build_run_manifest_reports_dirty_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _initialise_git_repository(tmp_path)
    artifact = repo / "input.csv"
    artifact.write_bytes(b"value\n1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add input")
    artifact.write_bytes(b"value\n2\n")
    monkeypatch.chdir(repo)

    manifest = build_run_manifest(
        "phase3-smoke",
        [artifact],
        {},
        repository_root=repo,
    )

    assert manifest["git_dirty"] is True


def test_build_run_manifest_rejects_missing_input(tmp_path: Path) -> None:
    repo = _initialise_git_repository(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_run_manifest(
            "phase3",
            [tmp_path / "missing.csv"],
            {},
            repository_root=repo,
        )


def test_build_run_manifest_uses_explicit_repository_outside_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _initialise_git_repository(tmp_path)
    artifact = repo / "input.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add input")
    expected_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    manifest = build_run_manifest(
        "experiment",
        [artifact],
        {},
        repository_root=repo,
    )

    assert manifest["git_commit"] == expected_commit
    assert manifest["git_dirty"] is False


def _initialise_git_repository(path: Path) -> Path:
    repo = path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    return repo


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
