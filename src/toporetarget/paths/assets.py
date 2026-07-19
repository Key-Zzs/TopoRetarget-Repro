from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from toporetarget.utils.hashing import sha256_file, sha256_tree


class AssetImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssetCheckResult:
    status: str
    destination: str
    missing_files: list[str]
    changed_files: list[str]
    missing_mesh_references: list[str]
    manifest_present: bool
    mesh_references_valid: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "destination": self.destination,
            "missing_files": self.missing_files,
            "changed_files": self.changed_files,
            "missing_mesh_references": self.missing_mesh_references,
            "manifest_present": self.manifest_present,
            "mesh_references_valid": self.mesh_references_valid,
            "message": self.message,
        }


@dataclass(frozen=True)
class ImportResult:
    status: str
    destination: str
    upstream_commit: str | None
    imported_file_count: int
    missing_mesh_references: list[str]
    manifest_path: str | None
    dry_run: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "destination": self.destination,
            "upstream_commit": self.upstream_commit,
            "imported_file_count": self.imported_file_count,
            "missing_mesh_references": self.missing_mesh_references,
            "manifest_path": self.manifest_path,
            "dry_run": self.dry_run,
            "message": self.message,
        }


def _git_commit(repository: Path) -> str:
    if not (repository / ".git").exists():
        raise AssetImportError(f"ManipTrans root is not a Git repository: {repository}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AssetImportError(f"Could not read ManipTrans commit: {repository}") from exc
    return completed.stdout.strip()


def _mesh_paths(asset_root: Path, urdf_path: Path) -> list[dict[str, str]]:
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError as exc:
        raise AssetImportError(f"Invalid URDF XML: {urdf_path}") from exc
    references: list[dict[str, str]] = []
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        reference = filename
        if filename.startswith("package://"):
            filename = filename.removeprefix("package://").split("/", 1)[-1]
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidates = [urdf_path.parent / candidate, asset_root / candidate]
            resolved = next((item for item in candidates if item.is_file()), candidates[0])
        else:
            resolved = candidate
        references.append(
            {
                "urdf": urdf_path.name,
                "filename": reference,
                "resolved_relative_path": (
                    resolved.relative_to(asset_root).as_posix()
                    if resolved.is_relative_to(asset_root)
                    else str(resolved)
                ),
                "exists": str(resolved.is_file()).lower(),
                "kind": "collision" if "collision" in reference.lower() else "visual",
            }
        )
    return references


def _validate_mesh_references(asset_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    all_references: list[dict[str, str]] = []
    missing: list[str] = []
    for name in ("rh_mano.urdf", "lh_mano.urdf"):
        urdf = asset_root / name
        if not urdf.is_file():
            missing.append(name)
            continue
        references = _mesh_paths(asset_root, urdf)
        all_references.extend(references)
        missing.extend(
            f"{reference['urdf']}:{reference['filename']}"
            for reference in references
            if reference["exists"] != "true"
        )
    return all_references, sorted(missing)


def import_artimano(
    source_root: Path,
    destination: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> ImportResult:
    source_root = source_root.expanduser().resolve()
    destination = destination.expanduser().resolve()
    source_asset = source_root / "maniptrans_envs" / "assets" / "mano_urdf"
    if destination.is_symlink():
        raise AssetImportError(f"Refusing to replace symlink destination: {destination}")
    if destination.exists() and not force and not dry_run:
        raise AssetImportError(f"Destination exists; pass --force to replace: {destination}")
    commit = _git_commit(source_root)
    license_path = next(
        (
            candidate
            for candidate in (source_root / "LICENSE", source_root / "maniptrans_envs" / "LICENSE")
            if candidate.is_file()
        ),
        None,
    )
    if license_path is None:
        raise AssetImportError(f"No ManipTrans LICENSE found below {source_root}")
    if not source_asset.is_dir():
        raise AssetImportError(f"Arti-MANO asset directory is missing: {source_asset}")
    _, missing = _validate_mesh_references(source_asset)
    if missing:
        raise AssetImportError(f"Source URDF mesh references are missing: {', '.join(missing)}")
    source_files = sha256_tree(source_asset)
    if dry_run:
        return ImportResult(
            "dry_run_ok",
            str(destination),
            commit,
            len(source_files),
            [],
            None,
            True,
            f"Would import {len(source_files)} files from {source_asset}",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=".artimano-import-", dir=str(destination.parent)))
    temp_destination = temp_parent / "payload"
    backup: Path | None = None
    try:
        shutil.copytree(source_asset, temp_destination, symlinks=False)
        references, _ = _validate_mesh_references(temp_destination)
        manifest = {
            "schema_version": 1,
            "upstream_repository_local_path": str(source_root),
            "upstream_commit": commit,
            "upstream_asset_path": str(source_asset),
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source_license_path": str(license_path),
            "source_license_sha256": sha256_file(license_path),
            "imported_files": [
                {"path": relative, "sha256": digest}
                for relative, digest in sorted(source_files.items())
            ],
            "mesh_reference_validation": {
                "valid": not bool(missing),
                "references": references,
                "missing": missing,
            },
            "modified": False,
        }
        (temp_destination / "asset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    return ImportResult(
        "imported",
        str(destination),
        commit,
        len(source_files),
        [],
        str(destination / "asset_manifest.json"),
        False,
        f"Imported {len(source_files)} files from ManipTrans",
    )


def check_artimano_assets(destination: Path) -> AssetCheckResult:
    destination = destination.expanduser().resolve()
    manifest_path = destination / "asset_manifest.json"
    required = [destination / "rh_mano.urdf", destination / "lh_mano.urdf"]
    if not destination.is_dir():
        return AssetCheckResult(
            "missing",
            str(destination),
            [str(path) for path in required],
            [],
            [],
            False,
            False,
            "Asset directory is missing",
        )
    if not manifest_path.is_file():
        return AssetCheckResult(
            "invalid",
            str(destination),
            [str(path) for path in required if not path.is_file()],
            [],
            [],
            False,
            False,
            "asset_manifest.json is missing",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return AssetCheckResult(
            "invalid", str(destination), [], [], [], False, False, f"Invalid manifest: {exc}"
        )
    missing_files: list[str] = [
        str(path.relative_to(destination)) for path in required if not path.is_file()
    ]
    changed_files: list[str] = []
    for record in manifest.get("imported_files", []):
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            changed_files.append(str(relative))
            continue
        path = destination / relative
        if not path.is_file():
            missing_files.append(relative)
        elif sha256_file(path) != expected:
            changed_files.append(relative)
    try:
        _, missing_meshes = _validate_mesh_references(destination)
    except AssetImportError as exc:
        return AssetCheckResult(
            "invalid",
            str(destination),
            sorted(set(missing_files)),
            sorted(set(changed_files)),
            [str(exc)],
            True,
            False,
            "Asset validation failed: invalid URDF",
        )
    source_license = Path(manifest.get("source_license_path", ""))
    license_changed = source_license.is_file() and sha256_file(source_license) != manifest.get(
        "source_license_sha256"
    )
    if license_changed:
        changed_files.append("source_license")
    valid = (
        not missing_files
        and not changed_files
        and not missing_meshes
        and manifest.get("modified") is False
    )
    return AssetCheckResult(
        "ok" if valid else "invalid",
        str(destination),
        sorted(set(missing_files)),
        sorted(set(changed_files)),
        missing_meshes,
        True,
        not missing_meshes,
        "Arti-MANO manifest, hashes, URDFs, and mesh references are consistent"
        if valid
        else "Asset validation failed",
    )
