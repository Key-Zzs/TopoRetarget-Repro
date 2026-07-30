#!/usr/bin/env python3
"""Archive verified compiled-branch evidence before non-force worktree removal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ArchiveSpec:
    """One clean worktree and the evidence that must be retained from it."""

    label: str
    worktree_name: str
    branch: str
    evidence_children: tuple[str, ...]


SPECS = (
    ArchiveSpec(
        "compiled-exact-sign",
        "TopoRetarget-Repro-compiled-exact-sign",
        "feature/compiled-exact-sign",
        ("assets", "experiments", "geometry", "patches", "reports"),
    ),
    ArchiveSpec(
        "compiled-kernel",
        "TopoRetarget-Repro-compiled-kernel",
        "develop/compiled-kernel",
        ("experiments", "patches", "reports"),
    ),
)
REGENERABLE_CHILDREN = ("build",)


def _git(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=worktree, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _clean_worktree(worktree: Path, expected_branch: str) -> tuple[str, str]:
    if not worktree.is_dir():
        raise RuntimeError(f"missing worktree: {worktree}")
    branch = _git(worktree, "branch", "--show-current")
    if branch != expected_branch:
        raise RuntimeError(f"unexpected branch for {worktree}: {branch!r}")
    for args in (
        ("status", "--short", "--untracked-files=all"),
        ("diff", "--name-status"),
        ("diff", "--cached", "--name-status"),
        ("diff", "--check"),
    ):
        if _git(worktree, *args):
            raise RuntimeError(
                f"worktree has unarchivable source state: {worktree} ({' '.join(args)})"
            )
    return branch, _git(worktree, "rev-parse", "HEAD")


def _inventory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _files(root):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _write_archive_metadata(
    destination: Path,
    *,
    spec: ArchiveSpec,
    source: Path,
    branch: str,
    commit: str,
    rows: list[dict[str, object]],
    remaining: list[dict[str, object]],
) -> None:
    archived_local = destination / ".local"
    verified = all(str(row["sha256"]) == _sha256(archived_local / str(row["path"])) for row in rows)
    payload = {
        "schema_version": "toporetarget.stage12.compiled_worktree_archive.v1",
        "archive_label": spec.label,
        "archived_unix_s": datetime.now(UTC).timestamp(),
        "source_worktree": str(source),
        "source_branch": branch,
        "source_commit": commit,
        "archived_paths": [f".local/{child}" for child in spec.evidence_children],
        "excluded_regenerable_paths": [f".local/{child}" for child in REGENERABLE_CHILDREN],
        "archived_file_count": len(rows),
        "archived_bytes": sum(int(row["bytes"]) for row in rows),
        "hash_verification_pass": verified,
        "source_evidence_remaining": {
            "file_count": len(remaining),
            "bytes": sum(int(row["bytes"]) for row in remaining),
            "paths": [str(row["path"]) for row in remaining],
            "regenerable_build_cache_only": all(
                str(row["path"]).startswith("build/") for row in remaining
            ),
        },
        "files": rows,
    }
    (destination / "archive_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (destination / "archive_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "archive_hashes.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "# Stage-12 compiled-worktree archive\n\n"
        f"- Source worktree: `{source}`\n"
        f"- Branch: `{branch}`\n"
        f"- Commit: `{commit}`\n"
        f"- Files: `{len(rows)}`\n"
        f"- Bytes: `{payload['archived_bytes']}`\n"
        f"- SHA-256 verification: `{verified}`\n"
        "\nOnly regenerable build-cache files remain in the source worktree.\n",
        encoding="utf-8",
    )
    if not verified or not payload["source_evidence_remaining"]["regenerable_build_cache_only"]:
        raise RuntimeError(f"archive verification failed: {destination}")


def _archive_one(root: Path, archive_parent: Path, spec: ArchiveSpec, timestamp: str) -> Path:
    source = root.parent / spec.worktree_name
    branch, commit = _clean_worktree(source, spec.branch)
    if os.stat(source).st_dev != os.stat(archive_parent).st_dev:
        raise RuntimeError(f"cross-device archive is disallowed for {source}")
    destination = archive_parent / f"{spec.label}_{timestamp}_{commit[:12]}"
    evidence = source / ".local"
    archived_local = destination / ".local"
    if destination.exists():
        # A metadata-write failure after same-device moves must be recoverable
        # without moving or deleting evidence a second time.
        if not archived_local.is_dir() or any(
            not (archived_local / child).exists() for child in spec.evidence_children
        ):
            raise RuntimeError(f"incomplete existing archive destination: {destination}")
        if any((evidence / child).exists() for child in spec.evidence_children):
            raise RuntimeError(f"ambiguous existing archive destination: {destination}")
    else:
        missing = [child for child in spec.evidence_children if not (evidence / child).exists()]
        if missing:
            raise RuntimeError(f"required archive sources missing for {source}: {missing}")
        destination.mkdir(parents=True)
        archived_local.mkdir()
        for child in spec.evidence_children:
            shutil.move(str(evidence / child), str(archived_local / child))
    rows = _inventory(archived_local)
    remaining = _inventory(evidence)
    _write_archive_metadata(
        destination,
        spec=spec,
        source=source,
        branch=branch,
        commit=commit,
        rows=rows,
        remaining=remaining,
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive_parent = root.parent / "TopoRetarget-Repro-branch-archives"
    archive_parent.mkdir(parents=True, exist_ok=True)
    destinations = [_archive_one(root, archive_parent, spec, args.timestamp) for spec in SPECS]
    print(json.dumps({"status": "pass", "archives": [str(path) for path in destinations]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
