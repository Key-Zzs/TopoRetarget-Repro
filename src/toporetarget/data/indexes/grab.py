"""Bounded, filename-first indexing for the external GRAB dataset.

Indexing never imports MANO/SMPL-X, opens a sequence NPZ, or computes any
frame-dependent geometry.  The index is deliberately disposable and belongs
under ``.local/`` rather than Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from toporetarget.utils.hashing import sha256_file


class GrabIndexError(RuntimeError):
    """Raised when GRAB root discovery or index data is ambiguous/invalid."""


_SUBJECT_RE = re.compile(r"^s\d+$", re.IGNORECASE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _config_value(key: str) -> Path | None:
    path = _repo_root() / ".local" / "config.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ImportError, ValueError):
        return None
    result = value.get(key) if isinstance(value, dict) else None
    return Path(result).expanduser() if isinstance(result, str) and result else None


def _looks_like_root(path: Path) -> bool:
    grab = path / "grab"
    if not grab.is_dir() or not (path / "tools" / "object_meshes").is_dir():
        return False
    try:
        subjects = sorted(
            item for item in grab.iterdir() if item.is_dir() and not item.is_symlink()
        )
        return any(
            _SUBJECT_RE.match(subject.name)
            and any(item.is_file() and item.suffix.lower() == ".npz" for item in subject.iterdir())
            for subject in subjects
        )
    except OSError:
        return False


def _looks_like_layout(path: Path) -> bool:
    return (path / "grab").is_dir() and (path / "tools" / "object_meshes").is_dir()


def _report_roots(report_path: Path) -> list[Path]:
    if not report_path.is_file():
        return []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    candidates: list[Path] = []
    for item in payload.get("datasets", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict) or item.get("canonical_dataset_name") != "grab":
            continue
        for value in item.get("candidate_directories", []):
            if isinstance(value, str):
                candidate = Path(value)
                if _looks_like_root(candidate):
                    candidates.append(candidate)
        for value in (item.get("data_root"), item.get("alias_root")):
            if isinstance(value, str):
                candidate = Path(value)
                if _looks_like_root(candidate):
                    candidates.append(candidate)
                nested = [child for child in candidate.iterdir()] if candidate.is_dir() else []
                candidates.extend(child for child in nested if _looks_like_root(child))
    return sorted(set(candidate.resolve() for candidate in candidates))


def resolve_grab_dataset_root(
    explicit_root: str | Path | None = None,
    *,
    sequence_path: str | Path | None = None,
    discovery_report: str | Path | None = None,
) -> Path:
    """Resolve a GRAB root using the Stage 0 precedence and bounded discovery."""

    candidates: list[Path] = []
    if explicit_root is not None:
        candidates = [Path(explicit_root).expanduser()]
    elif os.environ.get("GRAB_ROOT"):
        candidates = [Path(os.environ["GRAB_ROOT"]).expanduser()]
    elif _config_value("grab_root") is not None:
        candidates = [_config_value("grab_root") or Path()]
    else:
        report = (
            Path(discovery_report).expanduser()
            if discovery_report is not None
            else _repo_root() / ".local" / "reports" / "dataset_discovery.json"
        )
        candidates = _report_roots(report)
        if not candidates:
            storage = _config_value("storage_root")
            if storage is not None:
                try:
                    from toporetarget.paths.datasets import DatasetPathResolver

                    registry = _repo_root() / "configs" / "datasets" / "registry.yaml"
                    result = DatasetPathResolver(storage, registry, max_depth=4).discover()
                    grab = next(item for item in result if item.canonical_dataset_name == "grab")
                    candidates = [Path(value) for value in grab.candidate_directories]
                except (OSError, StopIteration, ValueError):
                    candidates = []
    if sequence_path is not None:
        source = Path(sequence_path).expanduser().resolve()
        for parent in (source.parent, *source.parents):
            if _looks_like_root(parent):
                candidates.append(parent)
    confirmed = sorted(
        set(candidate.resolve() for candidate in candidates if _looks_like_root(candidate))
    )
    if len(confirmed) > 1:
        raise GrabIndexError(
            "multiple GRAB roots were discovered; pass --grab-root explicitly: "
            + ", ".join(str(item) for item in confirmed)
        )
    if not confirmed:
        source_hint = f" for {sequence_path}" if sequence_path is not None else ""
        raise GrabIndexError(
            f"could not confirm a GRAB root{source_hint}; expected grab/ and tools/object_meshes/."
        )
    return confirmed[0]


def _file_entries(root: Path) -> Iterable[Path]:
    grab = root / "grab"
    try:
        subjects = sorted(
            item for item in grab.iterdir() if item.is_dir() and not item.is_symlink()
        )
    except OSError as exc:
        raise GrabIndexError(f"cannot scan GRAB directory {grab}: {exc}") from exc
    for subject in subjects:
        if not _SUBJECT_RE.match(subject.name):
            continue
        try:
            for item in sorted(subject.iterdir()):
                if item.is_file() and item.suffix.lower() == ".npz":
                    yield item
        except OSError as exc:
            raise GrabIndexError(f"cannot scan GRAB subject directory {subject}: {exc}") from exc


def _object_tokens(root: Path) -> list[str]:
    locations = [
        root / "tools" / "object_meshes",
        root / "tools" / "object_meshes" / "contact_meshes",
    ]
    names: set[str] = set()
    for location in locations:
        if not location.is_dir():
            continue
        names.update(
            path.stem
            for path in location.iterdir()
            if path.is_file() and path.suffix.lower() == ".ply"
        )
    return sorted(names, key=lambda value: (-len(value), value))


def _filename_metadata(stem: str, object_tokens: list[str]) -> tuple[str, str, str | None]:
    object_name = next(
        (token for token in object_tokens if stem == token or stem.startswith(token + "_")), ""
    )
    rest = stem[len(object_name) + 1 :] if object_name and len(stem) > len(object_name) else ""
    tokens = rest.split("_") if rest else stem.split("_")
    if not object_name:
        object_name = tokens.pop(0) if tokens else stem
    action = tokens.pop(0) if tokens else "unknown"
    repetition = tokens[0] if tokens and tokens[0].isdigit() else None
    return object_name, action, repetition


def _root_fingerprint(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()


def _entry(path: Path, root: Path, object_tokens: list[str], *, hash_files: bool) -> dict[str, Any]:
    stat = path.stat()
    object_name, action, repetition = _filename_metadata(path.stem, object_tokens)
    relative = path.relative_to(root).as_posix()
    result: dict[str, Any] = {
        "sequence_id": f"{path.parent.name}/{path.stem}",
        "subject_id": path.parent.name,
        "sequence_filename": path.name,
        "relative_path": relative,
        "object_token": object_name,
        "action_token": action,
        "repetition_token": repetition,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "source_root_fingerprint": _root_fingerprint(root),
        "metadata_quality": "filename_derived",
        "status": "active",
    }
    if hash_files:
        result["source_hash"] = sha256_file(path)
    return result


def _read_existing(index_dir: Path) -> dict[str, dict[str, Any]]:
    path = index_dir / "index.jsonl"
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            result[str(item["sequence_id"])] = item
    return result


def load_grab_index(index_dir: str | Path) -> list[dict[str, Any]]:
    """Load active entries from a disposable JSONL index without MANO imports."""

    entries = _read_existing(Path(index_dir).expanduser())
    return [entries[key] for key in sorted(entries) if entries[key].get("status") != "deleted"]


def build_grab_index(
    *,
    grab_root: str | Path | None = None,
    output: str | Path = ".local/index/grab",
    hash_files: bool = False,
    discovery_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build or incrementally refresh the GRAB JSONL index."""

    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    previous = _read_existing(destination)
    try:
        root = resolve_grab_dataset_root(grab_root, discovery_report=discovery_report)
    except GrabIndexError:
        # A previously indexed installation may temporarily contain no NPZ
        # files; retain deletion tombstones instead of losing the index.
        if (
            grab_root is None
            or not previous
            or not _looks_like_layout(Path(grab_root).expanduser())
        ):
            raise
        root = Path(grab_root).expanduser().resolve()
    current = {
        item["sequence_id"]: item
        for item in (
            _entry(path, root, _object_tokens(root), hash_files=hash_files)
            for path in _file_entries(root)
        )
    }
    for sequence_id, old in previous.items():
        if sequence_id not in current:
            current[sequence_id] = {**old, "status": "deleted"}
        elif old.get("file_size") != current[sequence_id].get("file_size") or old.get(
            "mtime_ns"
        ) != current[sequence_id].get("mtime_ns"):
            current[sequence_id]["change"] = "modified"
    entries = [current[key] for key in sorted(current)]
    lines = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in entries
    )
    index_path = destination / "index.jsonl"
    index_path.write_text(lines, encoding="utf-8")
    active = [item for item in entries if item.get("status") == "active"]
    manifest_core = {
        "adapter_version": "grab-dataset-adapter/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "grab_root": str(root),
        "file_count": len(active),
        "deleted_count": sum(item.get("status") == "deleted" for item in entries),
        "subjects": sorted({item["subject_id"] for item in active}),
        "hashing_mode": "sha256" if hash_files else "none",
        "index_file": "index.jsonl",
        "index_hash": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
    }
    manifest = {
        **manifest_core,
        "manifest_hash": hashlib.sha256(
            json.dumps(manifest_core, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "entries": active, "index": str(destination)}


__all__ = ["GrabIndexError", "build_grab_index", "load_grab_index", "resolve_grab_dataset_root"]
