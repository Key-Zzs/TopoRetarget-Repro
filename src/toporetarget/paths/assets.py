from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from toporetarget.utils.hashing import sha256_file, sha256_tree


class AssetImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssetResolution:
    """Resolved asset location and provenance, without exposing machine paths in specs."""

    asset_id: str
    source: str
    root: Path
    explicit: bool
    legacy_fallback_used: bool
    warnings: tuple[str, ...] = ()

    @property
    def manifest_path(self) -> Path:
        return self.root / "asset_manifest.json"

    @property
    def asset_manifest_hash(self) -> str | None:
        return sha256_file(self.manifest_path) if self.manifest_path.is_file() else None

    def as_dict(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        if self.manifest_path.is_file():
            try:
                loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
            except (OSError, json.JSONDecodeError):
                manifest = {}
        return {
            "asset_id": self.asset_id,
            "resolved_asset_source": self.source,
            "resolved_asset_root": str(self.root),
            "explicit_override": self.explicit,
            "legacy_fallback_used": self.legacy_fallback_used,
            "asset_manifest_hash": self.asset_manifest_hash,
            "source_commit": manifest.get("upstream_commit"),
            "license": manifest.get("license"),
            "warnings": list(self.warnings),
        }


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
    for side in ("rh", "lh"):
        candidates = (asset_root / "urdf" / f"{side}_mano.urdf", asset_root / f"{side}_mano.urdf")
        urdf = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not urdf.is_file():
            missing.append(urdf.relative_to(asset_root).as_posix())
            continue
        references = _mesh_paths(asset_root, urdf)
        all_references.extend(references)
        missing.extend(
            f"{reference['urdf']}:{reference['filename']}"
            for reference in references
            if reference["exists"] != "true"
        )
    return all_references, sorted(missing)


def resolve_artimano_asset(
    repo_root: str | Path,
    *,
    asset_root: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> AssetResolution:
    """Resolve tracked, explicit override, or legacy Arti-MANO assets in that order."""

    root = Path(repo_root).expanduser().resolve()
    env = os.environ if environ is None else environ
    warning_messages: list[str] = []
    if asset_root is not None:
        return AssetResolution(
            "artimano", "override", Path(asset_root).expanduser().resolve(), True, False
        )
    configured = env.get("TOPORETARGET_ARTIMANO_ASSET_ROOT")
    if configured:
        return AssetResolution(
            "artimano", "override", Path(configured).expanduser().resolve(), True, False
        )
    legacy_env = env.get("ARTIMANO_ASSET_ROOT")
    if legacy_env:
        message = "ARTIMANO_ASSET_ROOT is deprecated; use TOPORETARGET_ARTIMANO_ASSET_ROOT"
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        warning_messages.append(message)
        return AssetResolution(
            "artimano",
            "override",
            Path(legacy_env).expanduser().resolve(),
            True,
            False,
            tuple(warning_messages),
        )
    tracked = root / "third_party" / "robot_hands" / "artimano"
    if tracked.is_dir():
        return AssetResolution("artimano", "tracked", tracked, False, False)
    legacy = root / ".local" / "assets" / "artimano"
    if legacy.is_dir():
        message = "legacy .local/assets/artimano fallback is deprecated; vendor tracked assets"
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        warning_messages.append(message)
        return AssetResolution("artimano", "legacy", legacy, False, True, tuple(warning_messages))
    return AssetResolution("artimano", "tracked", tracked, False, False)


def _source_license(source_root: Path, source_asset: Path) -> Path:
    candidates = [source_root / "LICENSE", source_root / "maniptrans_envs" / "LICENSE"]
    license_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if license_path is None:
        raise AssetImportError(f"No ManipTrans LICENSE found below {source_root}")
    separate = sorted(
        candidate
        for candidate in source_asset.rglob("*")
        if candidate.is_file()
        and candidate.name.lower() in {"license", "license.txt", "copying", "notice", "notice.md"}
    )
    if separate:
        raise AssetImportError(
            "ARTIMANO_LICENSE_DECISION_REQUIRED: separate asset license candidates: "
            + ", ".join(str(candidate) for candidate in separate)
        )
    return license_path


def _source_manifest_hash(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _rebase_urdf_mesh_references(text: str, side: str) -> str:
    return text.replace(
        f'filename="{side}_urdf_meshes/', f'filename="../meshes/{side}_urdf_meshes/'
    ).replace(
        f'filename="{side}_urdf_meshes_visonly/',
        f'filename="../meshes/{side}_urdf_meshes_visonly/',
    )


def vendor_artimano(
    source_root: Path,
    destination: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    imported_at: str | None = None,
) -> ImportResult:
    """Create the structured tracked Arti-MANO snapshot without copying ManipTrans code."""

    source_root = source_root.expanduser().resolve()
    destination = destination.expanduser().resolve()
    source_asset = source_root / "maniptrans_envs" / "assets" / "mano_urdf"
    if destination.is_symlink():
        raise AssetImportError(f"Refusing to replace symlink destination: {destination}")
    if destination.exists() and not force and not dry_run:
        raise AssetImportError(f"Destination exists; pass --force to replace: {destination}")
    commit = _git_commit(source_root)
    license_path = _source_license(source_root, source_asset)
    if not source_asset.is_dir():
        raise AssetImportError(f"Arti-MANO asset directory is missing: {source_asset}")
    _, missing = _validate_mesh_references(source_asset)
    if missing:
        raise AssetImportError(f"Source URDF mesh references are missing: {', '.join(missing)}")
    source_files = sha256_tree(source_asset)
    source_hash = _source_manifest_hash(source_files)
    if dry_run:
        return ImportResult(
            "dry_run_ok",
            str(destination),
            commit,
            len(source_files),
            [],
            None,
            True,
            f"Would vendor {len(source_files)} files into structured tracked layout; "
            f"source_manifest_sha256={source_hash}",
        )

    imported_at = imported_at or datetime.now(timezone.utc).isoformat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=".artimano-vendor-", dir=str(destination.parent)))
    temp_destination = temp_parent / "payload"
    backup: Path | None = None
    try:
        (temp_destination / "urdf").mkdir(parents=True)
        (temp_destination / "meshes").mkdir(parents=True)
        for relative in sorted(source_files):
            source_path = source_asset / relative
            if relative in {"rh_mano.urdf", "lh_mano.urdf"}:
                side = relative[:2]
                target = temp_destination / "urdf" / relative
                target.write_text(
                    _rebase_urdf_mesh_references(source_path.read_text(encoding="utf-8"), side),
                    encoding="utf-8",
                )
            else:
                target = temp_destination / "meshes" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
        shutil.copyfile(license_path, temp_destination / "LICENSE")
        source_yaml = {
            "asset_id": "artimano",
            "upstream_repository": "https://github.com/ManipTrans/ManipTrans.git",
            "upstream_commit": commit,
            "upstream_path": "maniptrans_envs/assets/mano_urdf",
            "license": "GPL-3.0 (upstream root LICENSE; no separate asset license found)",
            "imported_at": imported_at,
            "included_paths": sorted(source_files),
            "excluded_paths": ["ManipTrans Python source and runtime environments"],
            "per_file_sha256": source_files,
            "manifest_sha256": source_hash,
            "import_tool_version": "toporetarget.vendor_robot_hand_assets.v1",
            "layout": (
                "URDF references are rebased to ../meshes without changing geometry or kinematics"
            ),
        }
        (temp_destination / "SOURCE.yaml").write_text(
            yaml.safe_dump(source_yaml, sort_keys=False), encoding="utf-8"
        )
        (temp_destination / "NOTICE.md").write_text(
            "# Arti-MANO vendor notice\n\n"
            "This is a path-rebased snapshot of `maniptrans_envs/assets/mano_urdf` from "
            f"ManipTrans commit `{commit}`. URDF mesh references point into `meshes/`; "
            "mesh and kinematic payload bytes are preserved. The upstream GPL license is "
            "included in `LICENSE`. No ManipTrans Python source is vendored.\n",
            encoding="utf-8",
        )
        tracked_files = sha256_tree(temp_destination)
        manifest = {
            "schema_version": 2,
            "asset_id": "artimano",
            "upstream_repository": "https://github.com/ManipTrans/ManipTrans.git",
            "upstream_commit": commit,
            "upstream_path": "maniptrans_envs/assets/mano_urdf",
            "license": "GPL-3.0 (upstream root LICENSE; no separate asset license found)",
            "source_license_sha256": sha256_file(license_path),
            "source_manifest_sha256": source_hash,
            "imported_at": imported_at,
            "tracked_files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(tracked_files.items())
                if path != "asset_manifest.json"
            ],
            "source_files": [
                {"path": path, "sha256": digest} for path, digest in sorted(source_files.items())
            ],
            "modified": "path_rebased",
            "mesh_reference_validation": {"valid": True, "missing": []},
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
        f"Vendored {len(source_files)} source files into tracked layout; "
        f"source_manifest_sha256={source_hash}",
    )


def compare_asset_payloads(left: Path, right: Path) -> dict[str, Any]:
    """Compare source payload bytes while ignoring the structured-layout path prefix."""

    def normalized(root: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for path, digest in sha256_tree(root).items():
            if path in {"asset_manifest.json", "SOURCE.yaml", "NOTICE.md", "LICENSE"}:
                continue
            parts = Path(path).parts
            source_path = root / path
            if parts and parts[0] in {"urdf", "meshes"}:
                path = Path(*parts[1:]).as_posix()
            if Path(path).name in {"rh_mano.urdf", "lh_mano.urdf"}:
                side = Path(path).name[:2]
                text = source_path.read_text(encoding="utf-8")
                text = text.replace(f'filename="../meshes/{side}_', f'filename="{side}_')
                import hashlib

                digest = hashlib.sha256(text.encode()).hexdigest()
            result[path] = digest
        return result

    left_files, right_files = (
        normalized(left.expanduser().resolve()),
        normalized(right.expanduser().resolve()),
    )
    missing = sorted(set(left_files) - set(right_files))
    extra = sorted(set(right_files) - set(left_files))
    changed = sorted(
        path for path in set(left_files) & set(right_files) if left_files[path] != right_files[path]
    )
    return {
        "status": "match" if not missing and not extra and not changed else "different",
        "left": str(left.expanduser().resolve()),
        "right": str(right.expanduser().resolve()),
        "left_file_count": len(left_files),
        "right_file_count": len(right_files),
        "missing_from_right": missing,
        "extra_in_right": extra,
        "changed": changed,
    }


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
    required = [
        next(
            (
                candidate
                for candidate in (
                    destination / "urdf" / f"{side}_mano.urdf",
                    destination / f"{side}_mano.urdf",
                )
                if candidate.is_file()
            ),
            destination / "urdf" / f"{side}_mano.urdf",
        )
        for side in ("rh", "lh")
    ]
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
    license_changed = False
    if source_license.is_file():
        license_changed = sha256_file(source_license) != manifest.get("source_license_sha256")
    elif (destination / "LICENSE").is_file() and manifest.get("source_license_sha256"):
        license_changed = sha256_file(destination / "LICENSE") != manifest.get(
            "source_license_sha256"
        )
    if license_changed:
        changed_files.append("LICENSE")
    modified = manifest.get("modified", False)
    valid = (
        not missing_files
        and not changed_files
        and not missing_meshes
        and modified in {False, "path_rebased"}
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
