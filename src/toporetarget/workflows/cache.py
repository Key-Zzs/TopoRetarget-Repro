"""Content-addressed signatures and conservative artifact reuse."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from toporetarget.utils.hashing import sha256_file, sha256_tree

from .schema import canonical_json, stable_hash


def path_hash(path: str | Path) -> str:
    target = Path(path).expanduser()
    if target.is_file():
        return sha256_file(target)
    if target.is_dir():
        digest = hashlib.sha256()
        for name, value in sha256_tree(target).items():
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()
    raise FileNotFoundError(target)


def optional_path_hash(path: str | Path | None) -> str | None:
    return None if path is None else path_hash(path)


def signature(
    node_id: str,
    *,
    implementation_version: str,
    inputs: Mapping[str, str | None],
    configs: Mapping[str, str | None],
    parameters: dict[str, Any] | None = None,
) -> str:
    return stable_hash(
        {
            "node_id": node_id,
            "implementation_version": implementation_version,
            "inputs": inputs,
            "configs": configs,
            "parameters": parameters or {},
        }
    )


def artifact_hashes(paths: dict[str, str]) -> dict[str, str]:
    return {name: path_hash(path) for name, path in sorted(paths.items()) if Path(path).exists()}


def outputs_exist(paths: dict[str, str]) -> bool:
    return bool(paths) and all(Path(path).exists() for path in paths.values())


def cache_record(
    *,
    node_id: str,
    expected_signature: str,
    output_paths: dict[str, str],
    validation_path: str | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "expected_signature": expected_signature,
        "output_paths": dict(output_paths),
        "output_hashes": artifact_hashes(output_paths),
        "validation_path": validation_path,
        "validation_status": _validation_status(validation_path) if validation_path else "pass",
    }


def _validation_status(path: str | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "invalid"
    if not isinstance(payload, dict):
        return "invalid"
    status = payload.get("status")
    if status is None and "pass" in payload:
        status = payload["pass"]
    if status is None and "all_frames_valid" in payload:
        status = payload["all_frames_valid"]
    if status is True:
        return "pass"
    if status is False:
        return "fail"
    # Some audited Stage 5-8 reports intentionally contain only positive
    # diagnostic fields and rely on the CLI exit code for failure signaling.
    # The executor writes a cache record only after that command returned 0.
    return str(status) if status is not None else "pass"


def can_reuse(record_path: str | Path, *, expected_signature: str) -> tuple[bool, str]:
    path = Path(record_path)
    if not path.is_file():
        return False, "cache record missing"
    try:
        import json

        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "cache record invalid"
    if record.get("expected_signature") != expected_signature:
        return False, "signature mismatch"
    outputs = record.get("output_paths")
    if not isinstance(outputs, dict) or not outputs_exist(
        {str(k): str(v) for k, v in outputs.items()}
    ):
        return False, "artifact missing"
    validation_path = record.get("validation_path")
    current_validation = (
        _validation_status(str(validation_path))
        if validation_path
        else record.get("validation_status")
    )
    if current_validation not in {"pass", "True", True}:
        return False, "previous validation did not pass"
    try:
        actual = artifact_hashes({str(k): str(v) for k, v in outputs.items()})
    except (OSError, FileNotFoundError):
        return False, "artifact hash unavailable"
    if actual != record.get("output_hashes"):
        return False, "artifact hash mismatch"
    return True, "signature and artifact hashes match"


def signature_payload(
    *,
    node_id: str,
    implementation_version: str,
    inputs: Mapping[str, str | None],
    configs: Mapping[str, str | None],
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "node_id": node_id,
        "implementation_version": implementation_version,
        "inputs": inputs,
        "configs": configs,
        "parameters": parameters or {},
    }
    value["signature"] = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return value


__all__ = [
    "artifact_hashes",
    "cache_record",
    "can_reuse",
    "optional_path_hash",
    "outputs_exist",
    "path_hash",
    "signature",
    "signature_payload",
]
