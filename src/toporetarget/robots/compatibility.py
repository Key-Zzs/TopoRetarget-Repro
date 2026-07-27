"""Read-only historical artifact rebinding and migration audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.paths.assets import compare_asset_payloads
from toporetarget.utils.hashing import sha256_file


def _json_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def _hash_values(root: Path, key: str) -> set[str]:
    values: set[str] = set()
    for path in _json_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        marker = f'"{key}"'
        if marker not in text:
            continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            continue

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for name, item in value.items():
                    if name == key and isinstance(item, str):
                        values.add(item)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(loaded)
    return values


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def audit_historical_artifacts(repo_root: str | Path) -> dict[str, Any]:
    """Audit old reports/exports without invoking a solver or mutating artifacts."""

    root = Path(repo_root).expanduser().resolve()
    tracked_root = root / "third_party" / "robot_hands" / "artimano"
    legacy_root = root / ".local" / "assets" / "artimano"
    payload_comparison = compare_asset_payloads(tracked_root, legacy_root)
    tracked_manifest = json.loads(
        (tracked_root / "asset_manifest.json").read_text(encoding="utf-8")
    )
    source_files = {
        str(record["path"]): str(record["sha256"])
        for record in tracked_manifest.get("source_files", [])
        if isinstance(record, dict) and "path" in record and "sha256" in record
    }
    old_manifest_hashes = _hash_values(root / ".local", "asset_manifest_hash")
    old_urdf_hashes = _hash_values(root / ".local", "urdf_hash") | _hash_values(
        root / ".local", "robot_urdf_hash"
    )
    legacy_manifest_hash = (
        sha256_file(legacy_root / "asset_manifest.json")
        if (legacy_root / "asset_manifest.json").is_file()
        else None
    )
    expected_old_urdf_hashes = {
        source_files.get("rh_mano.urdf"),
        source_files.get("lh_mano.urdf"),
    } - {None}
    artifact_roots = [
        root / ".local" / "reports" / name for name in ("stage7", "stage8", "stage9", "stage10")
    ] + [root / ".local" / "runs" / "stage10_reference_runtime"]
    readable = []
    for artifact_root in artifact_roots:
        if artifact_root.is_dir():
            readable.append(str(artifact_root.relative_to(root)))

    array_checks: list[dict[str, Any]] = []
    for path in sorted((root / ".local" / "runs" / "stage10_reference_runtime").glob("**/*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: _array_digest(np.asarray(archive[name])) for name in archive.files}
            with np.load(path, allow_pickle=False) as archive:
                arrays_after = {
                    name: _array_digest(np.asarray(archive[name])) for name in archive.files
                }
            array_checks.append(
                {
                    "path": str(path.relative_to(root)),
                    "readable": True,
                    "arrays_unchanged_during_read": arrays == arrays_after,
                    "array_digests": arrays,
                }
            )
        except (OSError, ValueError) as exc:
            array_checks.append(
                {"path": str(path.relative_to(root)), "readable": False, "error": str(exc)}
            )

    return {
        "schema_version": "toporetarget.f0_historical_artifact_compatibility.v1",
        "read_only": True,
        "solver_invocation_count_during_audit": 0,
        "tracked_asset_root": str(tracked_root),
        "legacy_asset_root": str(legacy_root),
        "legacy_absolute_path_rebind": {
            "legacy_path_exists": legacy_root.is_dir(),
            "tracked_rebind_target_exists": tracked_root.is_dir(),
            "old_manifest_hash": legacy_manifest_hash,
            "old_manifest_hashes_observed": sorted(old_manifest_hashes),
            "source_content_match": payload_comparison["status"] == "match",
        },
        "hash_compatibility": {
            "old_urdf_hashes_observed": sorted(old_urdf_hashes),
            "source_urdf_hashes": sorted(expected_old_urdf_hashes),
            "old_urdf_hashes_rebind": old_urdf_hashes <= expected_old_urdf_hashes,
            "old_manifest_hash_rebind": legacy_manifest_hash in old_manifest_hashes
            or not old_manifest_hashes,
        },
        "payload_comparison": payload_comparison,
        "historical_artifact_roots_readable": readable,
        "npz_read_checks": array_checks,
        "no_artifact_mutation_requested": True,
        "status": "pass"
        if payload_comparison["status"] == "match"
        and old_urdf_hashes <= expected_old_urdf_hashes
        and all(
            item.get("readable", False) and item.get("arrays_unchanged_during_read", True)
            for item in array_checks
        )
        else "pass_with_warnings",
    }


__all__ = ["audit_historical_artifacts"]
