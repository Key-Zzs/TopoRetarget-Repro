"""Independent raw-to-physical refinement batch contracts.

This module deliberately owns only batch concerns: outcome-independent HOCap
selection, immutable method/selection contracts, per-clip durable receipts,
resume decisions, timing, and aggregation.  It does *not* implement retarget,
source-policy training, support construction, PF evaluation, PPO, or replay.
Those operations must be supplied by a production authority manifest.  This
keeps a batch runner from accidentally turning a development-only actor into a
held-out input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.adapters.datasets.hocap_primary_object import (
    HOCapPrimaryObjectError,
    primary_object_from_authority,
)
from toporetarget.utils.hashing import sha256_file

SCHEMA_VERSION = "IndependentPhysicalRefinementBatchV1"
SELECTION_SEED = 20260822
DEVELOPMENT_CLIPS = frozenset({"hocap_170105", "hocap_170650"})
REQUIRED_AUTHORITIES = (
    "retarget",
    "source_policy",
    "support",
    "frozen_evaluation",
    "physical_refinement",
    "qualification",
    "trace_export",
)
TERMINAL_STATES = frozenset(
    {
        "ACCEPTED_FROZEN",
        "ACCEPTED_AFTER_REFINEMENT",
        "PPO_BUDGET_EXHAUSTED",
        "RETARGET_FAILED",
        "SOURCE_POLICY_FAILED",
        "SUPPORT_FAILED",
        "PF_PASS_DF_FAIL",
        "TECHNICAL_FAILURE",
        "PIPELINE_INVALID",
    }
)


class BatchContractError(RuntimeError):
    """Raised when a batch claim cannot be supported by durable evidence."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a JSON receipt after syncing its temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class HOCapCandidate:
    """Static raw-input eligibility record; it contains no physical outcome."""

    clip_id: str
    sequence: str
    subject: str
    object_ids: tuple[str, ...]
    raw_path: str
    raw_frames: int
    raw_fps: float
    raw_hashes: Mapping[str, str]
    eligible: bool
    reasons: tuple[str, ...]
    primary_object_id: str | None = None

    @property
    def object_id(self) -> str:
        if self.primary_object_id is not None:
            if self.primary_object_id not in self.object_ids:
                raise BatchContractError("PRIMARY_OBJECT_NOT_IN_DECLARED_OBJECT_SET")
            return self.primary_object_id
        if len(self.object_ids) == 1:
            return self.object_ids[0]
        raise BatchContractError(
            f"PRIMARY_OBJECT_UNRESOLVED:{self.clip_id}:{list(self.object_ids)}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "sequence": self.sequence,
            "subject": self.subject,
            "object_id": (
                self.primary_object_id
                if self.primary_object_id is not None
                else self.object_ids[0]
                if len(self.object_ids) == 1
                else None
            ),
            "primary_object_id": self.primary_object_id,
            "object_ids": list(self.object_ids),
            "raw_path": self.raw_path,
            "raw_frames": self.raw_frames,
            "raw_fps": self.raw_fps,
            "raw_duration_seconds": self.raw_frames / self.raw_fps if self.raw_fps else None,
            "raw_hashes": dict(self.raw_hashes),
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


def _clip_id(sequence: str) -> str:
    return f"hocap_{sequence.rsplit('_', 1)[-1]}"


def _finite_first_and_last(values: np.ndarray) -> bool:
    if len(values) == 0:
        return False
    return bool(np.isfinite(values[0]).all() and np.isfinite(values[-1]).all())


def scan_hocap_candidates(raw_root: Path) -> list[HOCapCandidate]:
    """Audit only immutable HOCap metadata, paths, shapes, and endpoints."""

    data_root = raw_root / "data"
    rows: list[HOCapCandidate] = []
    for meta_path in sorted(data_root.glob("subject_*/*/meta.yaml")):
        sequence_dir = meta_path.parent
        sequence = str(sequence_dir.relative_to(data_root))
        payload = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        poses_m = sequence_dir / "poses_m.npy"
        poses_o = sequence_dir / "poses_o.npy"
        subject = str(payload.get("subject_id") or sequence_dir.parent.name)
        object_ids = tuple(str(item) for item in (payload.get("object_ids") or []))
        fps = float(payload.get("fps") or payload.get("frame_rate") or 30.0)
        reasons: list[str] = []
        raw_frames = 0
        hashes: dict[str, str] = {"meta.yaml": sha256_file(meta_path)}
        if not poses_m.is_file() or not poses_o.is_file():
            reasons.append("RAW_MODALITY_MISSING")
        else:
            hashes["poses_m.npy"] = sha256_file(poses_m)
            hashes["poses_o.npy"] = sha256_file(poses_o)
            mano = np.load(poses_m, mmap_mode="r")
            obj = np.load(poses_o, mmap_mode="r")
            if mano.ndim != 3 or mano.shape[-1] != 51:
                reasons.append("MANO_SHAPE_INVALID")
            if obj.ndim != 3 or obj.shape[-1] != 7:
                reasons.append("OBJECT_POSE_SHAPE_INVALID")
            if not reasons:
                object_frames = obj.shape[1] if obj.shape[1] == mano.shape[1] else obj.shape[0]
                raw_frames = min(int(mano.shape[1]), int(object_frames))
                if raw_frames <= 0:
                    reasons.append("FRAME_COUNT_INVALID")
                elif not _finite_first_and_last(mano[:, :raw_frames]):
                    reasons.append("MANO_ENDPOINT_INVALID")
                elif not _finite_first_and_last(obj):
                    reasons.append("OBJECT_ENDPOINT_INVALID")
        sides = [str(item).lower() for item in (payload.get("mano_sides") or [])]
        if "right" not in sides:
            reasons.append("RIGHT_HAND_UNAVAILABLE")
        if not object_ids:
            reasons.append("OBJECT_ID_MISSING")
        for object_id in object_ids:
            mesh = data_root / "models" / object_id / "textured_mesh.obj"
            if not mesh.is_file():
                reasons.append(f"OBJECT_ASSET_MISSING:{object_id}")
            else:
                hashes[f"mesh:{object_id}"] = sha256_file(mesh)
        if fps <= 0.0:
            reasons.append("FPS_INVALID")
        rows.append(
            HOCapCandidate(
                clip_id=_clip_id(sequence),
                sequence=sequence,
                subject=subject,
                object_ids=object_ids,
                raw_path=str(sequence_dir.resolve()),
                raw_frames=raw_frames,
                raw_fps=fps,
                raw_hashes=hashes,
                eligible=not reasons,
                reasons=tuple(reasons),
            )
        )
    if not rows:
        raise BatchContractError(f"HOCAP_RAW_ROOT_EMPTY:{data_root}")
    return rows


def selection_key(candidate: HOCapCandidate, *, seed: int) -> str:
    return stable_hash({"seed": seed, "clip_id": candidate.clip_id, "sequence": candidate.sequence})


def select_held_out_candidates(
    candidates: Iterable[HOCapCandidate], *, seed: int = SELECTION_SEED, count: int = 5
) -> list[HOCapCandidate]:
    """Select a diverse, deterministic raw-only held-out set.

    The first pass takes at most one clip per declared object set and subject.  A
    deterministic second pass fills any remaining places, so small datasets do
    not silently become ineligible merely because full diversity is impossible.
    """

    eligible = [
        item for item in candidates if item.eligible and item.clip_id not in DEVELOPMENT_CLIPS
    ]
    ordered = sorted(eligible, key=lambda item: selection_key(item, seed=seed))
    selected: list[HOCapCandidate] = []
    subjects: set[str] = set()
    object_sets: set[tuple[str, ...]] = set()
    for item in ordered:
        if item.subject in subjects or item.object_ids in object_sets:
            continue
        selected.append(item)
        subjects.add(item.subject)
        object_sets.add(item.object_ids)
        if len(selected) == count:
            return selected
    for item in ordered:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) == count:
            return selected
    raise BatchContractError(f"HELD_OUT_POOL_TOO_SMALL:{len(selected)}/{count}")


def _manifest_core(
    selected: Iterable[HOCapCandidate],
    *,
    seed: int,
    primary_object_authority_sha256: str | None = None,
) -> dict[str, Any]:
    clips = []
    for rank, candidate in enumerate(selected, start=1):
        clip = candidate.as_dict()
        clip["selection_rank"] = rank
        clip["selection_key"] = selection_key(candidate, seed=seed)
        clip["exclusion_audit"] = {"development_clip": False, "outcome_observed": False}
        clips.append(clip)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "hocap",
        "selection_seed": seed,
        "held_out_count": len(clips),
        "HELD_OUT_SET_FROZEN": "YES",
        "selection_basis": "static_raw_metadata_only",
        "primary_object_contract": "explicit_fail_closed_authority_v1",
        "primary_object_authority_sha256": primary_object_authority_sha256,
        "exclusions": sorted(DEVELOPMENT_CLIPS),
        "clips": clips,
    }


def freeze_selection(
    *,
    candidates: Iterable[HOCapCandidate],
    root: Path,
    seed: int = SELECTION_SEED,
    primary_object_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic candidate, exclusion, and immutable five-clip manifests."""

    all_candidates = list(candidates)
    selected = select_held_out_candidates(all_candidates, seed=seed)
    resolved: list[HOCapCandidate] = []
    for candidate in selected:
        primary = candidate.primary_object_id
        if primary is None and len(candidate.object_ids) == 1:
            primary = candidate.object_ids[0]
        if primary is None:
            if primary_object_authority is None:
                raise BatchContractError(f"PRIMARY_OBJECT_AUTHORITY_REQUIRED:{candidate.clip_id}")
            try:
                primary = primary_object_from_authority(
                    primary_object_authority,
                    sequence=candidate.sequence,
                    available_object_ids=candidate.object_ids,
                )
            except HOCapPrimaryObjectError as exc:
                raise BatchContractError(str(exc)) from exc
        resolved.append(replace(candidate, primary_object_id=primary))
    selected = resolved
    authority_hash = (
        str(primary_object_authority.get("authority_sha256"))
        if primary_object_authority is not None
        else None
    )
    core = _manifest_core(
        selected,
        seed=seed,
        primary_object_authority_sha256=authority_hash,
    )
    core_hash = stable_hash(core)
    manifest = {**core, "manifest_sha256": core_hash}
    selection_root = root / "selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    rows = [item.as_dict() for item in all_candidates]
    with (selection_root / "candidate_pool.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "clip_id",
            "sequence",
            "subject",
            "object_id",
            "primary_object_id",
            "object_ids",
            "raw_path",
            "raw_frames",
            "raw_fps",
            "raw_duration_seconds",
            "eligible",
            "reasons",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row[key], (list, dict)) else row[key]
                    for key in fieldnames
                }
            )
    exclusions = [
        {"clip_id": item.clip_id, "reason": "DEVELOPMENT_CLIP", "source": "repository_audit"}
        for item in all_candidates
        if item.clip_id in DEVELOPMENT_CLIPS
    ]
    with (selection_root / "excluded_development_clips.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "reason", "source"])
        writer.writeheader()
        writer.writerows(exclusions)
    atomic_write_json(selection_root / "held_out_5_manifest.json", manifest)
    atomic_write_text(
        selection_root / "held_out_5_manifest.yaml", yaml.safe_dump(manifest, sort_keys=True)
    )
    atomic_write_json(
        selection_root / "selection_receipt.json",
        {
            "schema_version": SCHEMA_VERSION,
            "manifest": str((selection_root / "held_out_5_manifest.json").resolve()),
            "manifest_sha256": core_hash,
            "eligible_candidates": sum(item.eligible for item in all_candidates),
            "selected_clip_ids": [item.clip_id for item in selected],
            "outcome_based_selection": False,
        },
    )
    return manifest


def assert_frozen_manifest(manifest: Mapping[str, Any]) -> None:
    value = dict(manifest)
    expected = value.pop("manifest_sha256", None)
    if value.get("HELD_OUT_SET_FROZEN") != "YES" or value.get("held_out_count") != 5:
        raise BatchContractError("HELD_OUT_MANIFEST_INVALID")
    if not isinstance(expected, str) or stable_hash(value) != expected:
        raise BatchContractError("HELD_OUT_MANIFEST_HASH_DRIFT")
    clips = value.get("clips")
    if not isinstance(clips, list) or len(clips) != 5:
        raise BatchContractError("HELD_OUT_MANIFEST_CLIP_COUNT_INVALID")
    ids = [str(item.get("clip_id")) for item in clips if isinstance(item, Mapping)]
    if len(set(ids)) != 5 or DEVELOPMENT_CLIPS.intersection(ids):
        raise BatchContractError("HELD_OUT_MANIFEST_DEVELOPMENT_LEAKAGE")
    if any(item.get("exclusion_audit", {}).get("outcome_observed") for item in clips):
        raise BatchContractError("HELD_OUT_MANIFEST_OUTCOME_LEAKAGE")


def frozen_method_contract() -> dict[str, Any]:
    """Return the only shared pilot method configuration; values are not per clip."""

    return {
        "schema_version": "PhysicalRefinementMultiClipV1",
        "physics": {
            "gravity": "full",
            "object_gravity": "on",
            "nominal_friction": True,
            "support": "current_production_source_first_or_inferred_planar",
            "controller": "current_production_finite_virtual_6d_wrist_actuator_v1",
        },
        "reward": {"aggregation": "grouped_multiplicative_v1"},
        "rse": {
            "enabled": True,
            "reference_distance_relaxation": True,
            "adaptive_kappa": True,
        },
        "ppo": {"max_updates": 15, "independent_lineages": True, "joint_policy": False},
        "evaluation": {
            "frame": 0,
            "deterministic": True,
            "full_trajectory": True,
            "eval10": 10,
            "confirm20": 20,
        },
        "qualification": {
            "pf": "PF_V2",
            "df": ["pose", "linear", "angular_authority_v2"],
        },
        "rsi": {"training": "uniform_runtime_reference_valid_index_domain"},
    }


def freeze_method_contract(root: Path) -> tuple[dict[str, Any], str]:
    contract = frozen_method_contract()
    digest = stable_hash(contract)
    contract_root = root / "contracts"
    atomic_write_json(contract_root / "physical_refinement_multiclip_v1.json", contract)
    atomic_write_text(contract_root / "method_contract_hash.txt", f"{digest}\n")
    return contract, digest


def validate_authority_manifest(
    payload: Mapping[str, Any], clip_ids: Iterable[str]
) -> dict[str, Any]:
    """Fail closed unless each required authority explicitly supports every clip."""

    authorities = payload.get("authorities")
    if not isinstance(authorities, Mapping):
        raise BatchContractError("AUTHORITY_MANIFEST_MISSING_AUTHORITIES")
    missing: dict[str, list[str]] = {}
    requested = set(clip_ids)
    for name in REQUIRED_AUTHORITIES:
        entry = authorities.get(name)
        if not isinstance(entry, Mapping):
            missing[name] = sorted(requested)
            continue
        supported = {str(item) for item in entry.get("supported_clips", [])}
        unavailable = sorted(requested - supported)
        if unavailable:
            missing[name] = unavailable
    return {
        "valid": not missing,
        "unsupported": missing,
        "authority_manifest_sha256": stable_hash(payload),
    }


def initial_clip_state(*, clip: Mapping[str, Any], method_contract_hash: str) -> dict[str, Any]:
    clip_id = str(clip["clip_id"])
    root = clip_id
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "state": "SELECTED",
        "method_contract_hash": method_contract_hash,
        "raw_hashes": dict(clip["raw_hashes"]),
        "lineage": {
            "actor_root": f"ppo/{root}/actor",
            "critic_root": f"ppo/{root}/critic",
            "optimizer_root": f"ppo/{root}/optimizer",
            "normalizer_root": f"ppo/{root}/normalizer",
            "rng_seed": int(stable_hash({"clip": clip_id, "purpose": "ppo_rng"})[:16], 16),
        },
        "stages": [],
    }


def append_stage_receipt(
    state: Mapping[str, Any],
    *,
    stage: str,
    status: str,
    started_utc: str,
    ended_utc: str,
    wall_seconds: float,
    input_hashes: Mapping[str, str] | None = None,
    output_hashes: Mapping[str, str] | None = None,
    cache_hit: bool = False,
    retry_count: int = 0,
    productive_run_seconds: float | None = None,
    technical_retry_seconds: float = 0.0,
    exit_code: int = 0,
) -> dict[str, Any]:
    if state.get("state") in TERMINAL_STATES:
        raise BatchContractError("TERMINAL_CLIP_CANNOT_ADVANCE")
    updated = json.loads(json.dumps(state))
    updated["state"] = status
    updated["stages"].append(
        {
            "stage": stage,
            "status": status,
            "start_utc": started_utc,
            "end_utc": ended_utc,
            "wall_seconds": wall_seconds,
            "productive_run_seconds": wall_seconds
            if productive_run_seconds is None
            else productive_run_seconds,
            "technical_retry_seconds": technical_retry_seconds,
            "input_hashes": dict(input_hashes or {}),
            "output_hashes": dict(output_hashes or {}),
            "cache_hit": cache_hit,
            "retry_count": retry_count,
            "exit_code": exit_code,
        }
    )
    return updated


def write_clip_state(root: Path, state: Mapping[str, Any]) -> Path:
    destination = root / "clips" / str(state["clip_id"]) / "final_receipt.json"
    atomic_write_json(destination, state)
    return destination


def write_capability_gap_receipts(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    method_contract_hash: str,
    authority: Mapping[str, Any],
) -> list[Path]:
    """Record an execution block without fabricating physical outcomes or PPO work."""

    assert_frozen_manifest(manifest)
    check = validate_authority_manifest(
        authority, [str(item["clip_id"]) for item in manifest["clips"]]
    )
    if check["valid"]:
        raise BatchContractError("CAPABILITY_GAP_RECEIPT_REQUIRES_A_REAL_GAP")
    written = []
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for clip in manifest["clips"]:
        state = initial_clip_state(clip=clip, method_contract_hash=method_contract_hash)
        state = append_stage_receipt(
            state,
            stage="authority_preflight",
            status="PIPELINE_INVALID",
            started_utc=now,
            ended_utc=now,
            wall_seconds=0.0,
            input_hashes=clip["raw_hashes"],
            output_hashes={},
            exit_code=2,
        )
        state["capability_gap"] = check
        state["PPO_UPDATES"] = 0
        state["physical_outcome_observed"] = False
        written.append(write_clip_state(root, state))
    return written


def assert_independent_lineages(states: Iterable[Mapping[str, Any]]) -> None:
    rows = list(states)
    fields = ("actor_root", "critic_root", "optimizer_root", "normalizer_root", "rng_seed")
    for field in fields:
        values = [item["lineage"][field] for item in rows]
        if len(set(values)) != len(values):
            raise BatchContractError(f"INDEPENDENT_LINEAGE_DRIFT:{field}")


__all__ = [
    "BatchContractError",
    "DEVELOPMENT_CLIPS",
    "HOCapCandidate",
    "REQUIRED_AUTHORITIES",
    "SCHEMA_VERSION",
    "SELECTION_SEED",
    "append_stage_receipt",
    "assert_frozen_manifest",
    "assert_independent_lineages",
    "atomic_write_json",
    "atomic_write_text",
    "freeze_method_contract",
    "freeze_selection",
    "frozen_method_contract",
    "initial_clip_state",
    "scan_hocap_candidates",
    "select_held_out_candidates",
    "stable_hash",
    "validate_authority_manifest",
    "write_capability_gap_receipts",
    "write_clip_state",
]
