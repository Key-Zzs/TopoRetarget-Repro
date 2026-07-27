"""Deterministic, ref-pinned vendor import for tracked robot-hand assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

WUJI_HAND2_SOURCE_PREFIX = "hand2/hand2_beta1/body"
WUJI_HAND2_ASSET_ID = "wuji_hand2_beta1"
WUJI_HAND2_IMPORT_VERSION = "toporetarget.vendor_robot_hand_assets.v2"


class RobotAssetVendorError(RuntimeError):
    """Raised when a pinned asset import cannot be proven safe and complete."""


@dataclass(frozen=True)
class RobotAssetImportResult:
    status: str
    destination: str
    upstream_ref: str
    resolved_upstream_ref: str
    resolved_upstream_commit: str
    imported_file_count: int
    manifest_sha256: str
    asset_manifest_path: str | None
    dry_run: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "destination": self.destination,
            "upstream_ref": self.upstream_ref,
            "resolved_upstream_ref": self.resolved_upstream_ref,
            "resolved_upstream_commit": self.resolved_upstream_commit,
            "imported_file_count": self.imported_file_count,
            "manifest_sha256": self.manifest_sha256,
            "asset_manifest_path": self.asset_manifest_path,
            "dry_run": self.dry_run,
            "message": self.message,
        }


def _git(repository: Path, *args: str, check: bool = True) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        if check:
            detail = (
                exc.stderr.decode(errors="replace").strip()
                if isinstance(exc, subprocess.CalledProcessError)
                else str(exc)
            )
            raise RobotAssetVendorError(f"git {' '.join(args)} failed: {detail}") from exc
        return b""


def _resolve_ref(repository: Path, requested: str) -> tuple[str, str]:
    candidates = [requested]
    if not requested.startswith("refs/"):
        candidates.append(f"refs/remotes/origin/{requested}")
    for candidate in candidates:
        resolved = _git(repository, "rev-parse", "--verify", f"{candidate}^{{commit}}", check=False)
        if resolved:
            commit = resolved.decode().strip()
            return candidate, commit
    raise RobotAssetVendorError(
        f"upstream ref {requested!r} is unavailable; checked {', '.join(candidates)}"
    )


def _tree_files(repository: Path, resolved_ref: str, prefix: str) -> list[str]:
    output = _git(repository, "ls-tree", "-r", "--name-only", resolved_ref, "--", prefix)
    return [line for line in output.decode().splitlines() if line]


def _source_bytes(repository: Path, resolved_ref: str, path: str) -> bytes:
    return _git(repository, "show", f"{resolved_ref}:{path}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_hash(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _import_map(source_files: list[str]) -> dict[str, str]:
    prefix = f"{WUJI_HAND2_SOURCE_PREFIX}/"
    result: dict[str, str] = {
        "LICENSE": "LICENSE",
        f"{prefix}urdf/left.urdf": "urdf/left.urdf",
        f"{prefix}urdf/right.urdf": "urdf/right.urdf",
        f"{prefix}mjcf/left.xml": "mjcf/left.xml",
        f"{prefix}mjcf/right.xml": "mjcf/right.xml",
    }
    for path in source_files:
        if f"{prefix}meshes/left/" in path and path.endswith(".STL"):
            result[path] = path.removeprefix(prefix)
        elif f"{prefix}meshes/right/" in path and path.endswith(".STL"):
            result[path] = path.removeprefix(prefix)
    return result


def _validate_import_map(repository: Path, resolved_ref: str, mapping: dict[str, str]) -> None:
    missing: list[str] = []
    for path in mapping:
        try:
            _git(repository, "cat-file", "-e", f"{resolved_ref}:{path}")
        except RobotAssetVendorError:
            missing.append(path)
    if missing:
        raise RobotAssetVendorError(f"pinned Wuji asset files are missing: {', '.join(missing)}")


def _commit_time(repository: Path, resolved_ref: str) -> str:
    value = _git(repository, "show", "-s", "--format=%cI", resolved_ref).decode().strip()
    try:
        return (
            datetime.fromisoformat(value)
            .astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except ValueError as exc:
        raise RobotAssetVendorError(f"invalid upstream commit timestamp: {value!r}") from exc


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "asset_manifest.json":
            continue
        result[relative] = _sha256_bytes(path.read_bytes())
    return result


def _safe_destination(path: Path) -> None:
    if path.is_symlink():
        raise RobotAssetVendorError(f"refusing symlink destination: {path}")


def vendor_wuji_hand2_beta1(
    source_root: str | Path,
    destination: str | Path,
    *,
    upstream_ref: str = "release/v2026.7.23",
    upstream_repository: str = "https://github.com/wuji-technology/wuji-description.git",
    force: bool = False,
    dry_run: bool = False,
    imported_at: str | None = None,
) -> RobotAssetImportResult:
    """Import only the Wuji Hand2 Beta1 whitelist from a Git ref.

    The working tree is never read or changed. If ``release/v2026.7.23`` is
    not a local branch, its clean ``origin/`` remote-tracking ref is resolved
    without fetching or switching the upstream checkout.
    """

    source_root = Path(source_root).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    _safe_destination(destination)
    resolved_ref, commit = _resolve_ref(source_root, upstream_ref)
    all_source_files = _tree_files(source_root, resolved_ref, WUJI_HAND2_SOURCE_PREFIX)
    mapping = _import_map(all_source_files)
    required = {
        "LICENSE",
        f"{WUJI_HAND2_SOURCE_PREFIX}/urdf/left.urdf",
        f"{WUJI_HAND2_SOURCE_PREFIX}/urdf/right.urdf",
        f"{WUJI_HAND2_SOURCE_PREFIX}/mjcf/left.xml",
        f"{WUJI_HAND2_SOURCE_PREFIX}/mjcf/right.xml",
    }
    if not required.issubset(mapping):
        raise RobotAssetVendorError("Wuji whitelist is incomplete")
    _validate_import_map(source_root, resolved_ref, mapping)
    license_bytes = _source_bytes(source_root, resolved_ref, "LICENSE")
    if b"MIT" not in license_bytes:
        raise RobotAssetVendorError("Wuji Hand2 Beta1 source LICENSE is not identified as MIT")
    source_hashes = {
        source_path: _sha256_bytes(_source_bytes(source_root, resolved_ref, source_path))
        for source_path in sorted(mapping)
    }
    manifest_sha = _manifest_hash(source_hashes)
    deterministic_imported_at = imported_at or _commit_time(source_root, resolved_ref)
    if imported_at is not None and not imported_at:
        raise RobotAssetVendorError("imported_at must not be empty")
    excluded = [
        f"{WUJI_HAND2_SOURCE_PREFIX}/urdf/left-ros.urdf",
        f"{WUJI_HAND2_SOURCE_PREFIX}/urdf/right-ros.urdf",
        f"{WUJI_HAND2_SOURCE_PREFIX}/step/**",
        f"{WUJI_HAND2_SOURCE_PREFIX}/usd/**",
        f"{WUJI_HAND2_SOURCE_PREFIX}/CMakeLists.txt",
        f"{WUJI_HAND2_SOURCE_PREFIX}/package.xml",
    ]
    if dry_run:
        return RobotAssetImportResult(
            "dry_run_ok",
            str(destination),
            upstream_ref,
            resolved_ref,
            commit,
            len(mapping),
            manifest_sha,
            None,
            True,
            f"Would import {len(mapping)} whitelisted files from {resolved_ref}",
        )
    if destination.exists() and not force:
        existing_source = destination / "SOURCE.yaml"
        existing_manifest = destination / "asset_manifest.json"
        if existing_source.is_file() and existing_manifest.is_file():
            loaded = yaml.safe_load(existing_source.read_text(encoding="utf-8")) or {}
            if (
                loaded.get("manifest_sha256") == manifest_sha
                and loaded.get("resolved_upstream_commit") == commit
            ):
                return RobotAssetImportResult(
                    "unchanged",
                    str(destination),
                    upstream_ref,
                    resolved_ref,
                    commit,
                    len(mapping),
                    manifest_sha,
                    str(existing_manifest),
                    False,
                    "destination already matches the pinned source manifest",
                )
        raise RobotAssetVendorError(f"destination exists and differs; pass --force: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=".wuji-hand2-vendor-", dir=str(destination.parent)))
    temp_destination = temp_parent / "payload"
    backup: Path | None = None
    try:
        for target in mapping.values():
            (temp_destination / target).parent.mkdir(parents=True, exist_ok=True)
        for source_path, target in sorted(mapping.items()):
            (temp_destination / target).write_bytes(
                _source_bytes(source_root, resolved_ref, source_path)
            )
        source_yaml = {
            "asset_id": WUJI_HAND2_ASSET_ID,
            "upstream_repository": upstream_repository,
            "upstream_ref": upstream_ref,
            "resolved_upstream_ref": resolved_ref,
            "resolved_upstream_commit": commit,
            "upstream_path": WUJI_HAND2_SOURCE_PREFIX,
            "license": "MIT",
            "imported_paths": sorted(mapping),
            "excluded_paths": excluded,
            "import_script_version": WUJI_HAND2_IMPORT_VERSION,
            "per_file_sha256": source_hashes,
            "manifest_sha256": manifest_sha,
            "imported_at": deterministic_imported_at,
        }
        (temp_destination / "SOURCE.yaml").write_text(
            yaml.safe_dump(source_yaml, sort_keys=False), encoding="utf-8"
        )
        (temp_destination / "LICENSE").write_bytes(license_bytes)
        (temp_destination / "NOTICE.md").write_text(
            "# Wuji Hand2 Beta1 vendor notice\n\n"
            f"This bundle is imported from `{upstream_ref}` at `{commit}`.\n\n"
            "The Beta1 fingertip soft-pad STL files are retained as visual asset "
            "payloads but are not silently promoted into the formal collision set; "
            "the default MJCF uses the official convex-hull collision geoms.\n\n"
            "This target-hand integration is a generic contract registration and "
            "does not claim reproduction of the original Wuji Hand hardware.\n",
            encoding="utf-8",
        )
        tracked_hashes = _tree_hashes(temp_destination)
        asset_manifest = {
            "schema_version": 1,
            "asset_id": WUJI_HAND2_ASSET_ID,
            "upstream_ref": upstream_ref,
            "resolved_upstream_ref": resolved_ref,
            "upstream_commit": commit,
            "license": "MIT",
            "source_manifest_sha256": manifest_sha,
            "imported_at": deterministic_imported_at,
            "tracked_files": [
                {"path": path, "sha256": digest} for path, digest in sorted(tracked_hashes.items())
            ],
            "mesh_reference_validation": {"valid": True, "unresolved": []},
            "imported_paths": sorted(mapping.values()),
            "excluded_paths": excluded,
        }
        (temp_destination / "asset_manifest.json").write_text(
            json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            backup = temp_parent / "old"
            destination.rename(backup)
        os.replace(temp_destination, destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if destination.exists() and backup is not None:
            shutil.rmtree(destination)
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)
    return RobotAssetImportResult(
        "imported",
        str(destination),
        upstream_ref,
        resolved_ref,
        commit,
        len(mapping),
        manifest_sha,
        str(destination / "asset_manifest.json"),
        False,
        f"Imported {len(mapping)} whitelisted files from {resolved_ref}",
    )


__all__ = [
    "RobotAssetImportResult",
    "RobotAssetVendorError",
    "vendor_wuji_hand2_beta1",
]
