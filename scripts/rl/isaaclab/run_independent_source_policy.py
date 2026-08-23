#!/usr/bin/env python3
"""Build one independent geometry-to-Strict-V4 source-policy lineage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_manifest,
    atomic_write_json,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

L0_SAMPLES = 1_024_000
STRICT_V4_SAMPLES = 1_064_960


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-object-authority", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--geometric-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--wuji-mjcf", type=Path, required=True)
    parser.add_argument("--strict-v4-contract", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_SOURCE_POLICY_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_SOURCE_POLICY_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _run_step(
    name: str,
    command: list[str],
    *,
    log_root: Path,
    expected_artifacts: tuple[Path, ...] = (),
) -> dict[str, Any]:
    receipt_path = log_root / f"{name}.receipt.json"
    if receipt_path.is_file():
        previous = _json(receipt_path)
        if (
            previous.get("status") == "PASS"
            and previous.get("command") == command
            and all(path.is_file() for path in expected_artifacts)
        ):
            return {**previous, "resumed_from_pass_receipt": True}
    log_root.mkdir(parents=True, exist_ok=True)
    started = _utc()
    tick = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = log_root / f"{name}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    missing_artifacts = [
        str(path.resolve()) for path in expected_artifacts if not path.is_file()
    ]
    passed = result.returncode == 0 and not missing_artifacts
    receipt = {
        "stage": name,
        "status": "PASS" if passed else "FAIL",
        "command": command,
        "expected_artifacts": [str(path.resolve()) for path in expected_artifacts],
        "missing_artifacts": missing_artifacts,
        "started_utc": started,
        "ended_utc": _utc(),
        "wall_seconds": time.perf_counter() - tick,
        "returncode": result.returncode,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
    }
    atomic_write_json(receipt_path, receipt)
    if not passed:
        raise RuntimeError(f"INDEPENDENT_SOURCE_POLICY_STAGE_FAILED:{name}:{log_path}")
    return receipt


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.num_envs != 1024:
        raise ValueError("INDEPENDENT_SOURCE_POLICY_REQUIRES_EULA_AND_1024_ENVS")
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_CLIP_ID_INVALID")
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    assert_frozen_manifest(manifest)
    rows = [row for row in manifest["clips"] if row.get("clip_id") == args.clip_id]
    if len(rows) != 1 or args.clip_id in {"hocap_170105", "hocap_170650"}:
        raise ValueError("INDEPENDENT_SOURCE_POLICY_CLIP_NOT_HELD_OUT")
    lineage_seed = int(rows[0]["selection_key"][:8], 16) & 0x7FFFFFFF
    authority_path = args.primary_object_authority.resolve()
    authority = _json(authority_path)
    if manifest.get("primary_object_authority_sha256") != authority.get("authority_sha256"):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_PRIMARY_AUTHORITY_MISMATCH")
    geometry_path = args.geometric_receipt.resolve()
    geometry = _json(geometry_path)
    if (
        geometry.get("status") != "PASS"
        or geometry.get("clip_id") != args.clip_id
        or geometry.get("primary_object_id") != rows[0].get("primary_object_id")
        or geometry.get("selection_manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_GEOMETRY_RECEIPT_INVALID")

    run_root = args.run_root.resolve() / args.clip_id / "source_policy"
    report_root = args.report_root.resolve() / "clips" / args.clip_id / "source_policy"
    log_root = report_root / "logs"
    reference_root = run_root / "references"
    object_root = run_root / "objects"
    usd_parent = run_root / "object_usd"
    usd_root = usd_parent / args.clip_id
    contact_root = run_root / "source_contact"
    world_reference = reference_root / f"{args.clip_id}.world_wrist.stage16.npz"
    object_mesh = object_root / f"{args.clip_id}.obj"
    reference_v1 = reference_root / f"{args.clip_id}.reference_v1.npz"
    reference_v2 = reference_root / f"{args.clip_id}.reference_kinematics_v2.npz"
    reference_report = report_root / "reference_preparation.json"
    object_usd_report = report_root / "object_usd.json"
    object_usd = usd_root / f"{args.clip_id}.usda"
    source_contact_receipt = contact_root / args.clip_id / "source_contact_authority.json"
    l0_root = run_root / "l0"
    l0_checkpoint = (
        l0_root / args.clip_id / f"stage16d_ppo26d_{args.clip_id.removeprefix('hocap_')}_l0.pt"
    )
    l0_result = l0_root / args.clip_id / "l0_training.json"
    v4_root = run_root / "strict_v4"
    v4_result = v4_root / "ppo_v4" / args.clip_id / "training_result.json"
    final_receipt = report_root / "source_policy_receipt.json"
    if final_receipt.exists():
        raise FileExistsError(f"INDEPENDENT_SOURCE_POLICY_REFUSES_OVERWRITE:{final_receipt}")

    final = Path(str(geometry["artifacts"]["final"]["path"])).resolve()
    canonical = Path(str(geometry["artifacts"]["canonical"]["path"])).resolve()
    checkpoint_manifest = final.parent / "continuous_checkpoints" / "manifest.json"
    mjcf = args.wuji_mjcf.resolve()
    strict_contract = args.strict_v4_contract.resolve()
    steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        steps.append(
            _run_step(
                "prepare_reference",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toporetarget-rl",
                    "python",
                    "scripts/rl/prepare_independent_source_reference.py",
                    "--clip-id",
                    args.clip_id,
                    "--final-trajectory",
                    str(final),
                    "--canonical",
                    str(canonical),
                    "--checkpoint-manifest",
                    str(checkpoint_manifest),
                    "--wuji-mjcf",
                    str(mjcf),
                    "--world-reference-output",
                    str(world_reference),
                    "--object-mesh-output",
                    str(object_mesh),
                    "--reference-v1-output",
                    str(reference_v1),
                    "--reference-v2-output",
                    str(reference_v2),
                    "--report",
                    str(reference_report),
                ],
                log_root=log_root,
                expected_artifacts=(
                    world_reference,
                    object_mesh,
                    reference_v1,
                    reference_v2,
                    reference_report,
                ),
            )
        )
        steps.append(
            _run_step(
                "import_object_usd",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toporetarget-isaaclab",
                    "python",
                    "scripts/rl/isaaclab/import_hocap_objects.py",
                    "--mesh",
                    str(object_mesh),
                    "--object-id",
                    args.clip_id,
                    "--output-dir",
                    str(usd_root),
                    "--report",
                    str(object_usd_report),
                    "--accept-eula",
                ],
                log_root=log_root,
                expected_artifacts=(object_usd, object_usd_report),
            )
        )
        steps.append(
            _run_step(
                "materialize_source_contact",
                [
                    sys.executable,
                    "scripts/evaluation/materialize_independent_hocap_source_contact.py",
                    "--manifest",
                    str(manifest_path),
                    "--primary-object-authority",
                    str(authority_path),
                    "--clip-id",
                    args.clip_id,
                    "--world-reference",
                    str(world_reference),
                    "--reference-v2",
                    str(reference_v2),
                    "--strict-v4-contract",
                    str(strict_contract),
                    "--output-root",
                    str(contact_root),
                ],
                log_root=log_root,
                expected_artifacts=(source_contact_receipt,),
            )
        )
        steps.append(
            _run_step(
                "train_l0",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toporetarget-isaaclab",
                    "python",
                    "scripts/rl/isaaclab/train_stage16d_ppo26d.py",
                    "--clip",
                    args.clip_id,
                    "--reference",
                    str(reference_v1),
                    "--object-usd",
                    str(object_usd),
                    "--output-root",
                    str(l0_root),
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(lineage_seed),
                    "--accept-eula",
                ],
                log_root=log_root,
                expected_artifacts=(l0_checkpoint, l0_result),
            )
        )
        steps.append(
            _run_step(
                "train_strict_v4",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toporetarget-isaaclab",
                    "python",
                    "scripts/rl/isaaclab/train_stage16d_ppo26d_object_twist.py",
                    "--clip",
                    args.clip_id,
                    "--reference",
                    str(reference_v2),
                    "--object-usd",
                    str(object_usd),
                    "--reference-root",
                    str(reference_root),
                    "--initialization-checkpoint",
                    str(l0_checkpoint),
                    "--strict-per-finger-contact-reward-v4",
                    "--strict-v4-contract",
                    str(strict_contract),
                    "--strict-v4-source-mask-root",
                    str(contact_root),
                    "--target-reward-v4-samples",
                    str(STRICT_V4_SAMPLES),
                    "--checkpoint-targets",
                    str(STRICT_V4_SAMPLES),
                    "--output-root",
                    str(v4_root),
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(lineage_seed),
                    "--accept-eula",
                ],
                log_root=log_root,
                expected_artifacts=(v4_result,),
            )
        )
        l0 = _json(l0_result)
        v4 = _json(v4_result)
        contact = _json(source_contact_receipt)
    except BaseException as error:
        atomic_write_json(
            report_root / "source_policy_failure.json",
            {
                "schema_version": "IndependentSourcePolicyFailureV1",
                "status": "FAIL",
                "clip_id": args.clip_id,
                "reason": f"{type(error).__name__}:{error}",
                "completed_stages": steps,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise
    if (
        int(l0.get("cumulative_samples", -1)) != L0_SAMPLES
        or int(l0.get("seed", -1)) != lineage_seed
        or v4.get("status") != "STRICT_V4_TRAINING_SEGMENT_COMPLETE"
        or int(v4.get("reward_v4_samples_start", -1)) != 0
        or int(v4.get("reward_v4_samples", -1)) != STRICT_V4_SAMPLES
        or int(v4.get("seed", -1)) != lineage_seed
        or contact.get("status") != "PASS"
    ):
        raise RuntimeError("INDEPENDENT_SOURCE_POLICY_FINAL_CONTRACT_INVALID")
    v4_checkpoint = Path(str(v4["checkpoint"])).resolve()
    receipt = {
        "schema_version": "IndependentSourcePolicyReceiptV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "primary_object_id": rows[0]["primary_object_id"],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "primary_object_authority_sha256": authority["authority_sha256"],
        "l0_samples": L0_SAMPLES,
        "strict_v4_samples": STRICT_V4_SAMPLES,
        "checkpoint": str(v4_checkpoint),
        "checkpoint_sha256": sha256_file(v4_checkpoint),
        "source_training_result": _artifact(v4_result),
        "lineage": {
            "actor_root": str((v4_root / "ppo_v4" / args.clip_id).resolve()),
            "critic_root": str((v4_root / "ppo_v4" / args.clip_id).resolve()),
            "optimizer_root": str((v4_root / "ppo_v4" / args.clip_id).resolve()),
            "normalizer_root": str((v4_root / "ppo_v4" / args.clip_id).resolve()),
            "rng_seed": lineage_seed,
        },
        "artifacts": {
            "world_reference": _artifact(world_reference),
            "reference_v1": _artifact(reference_v1),
            "reference_v2": _artifact(reference_v2),
            "object_mesh": _artifact(object_mesh),
            "object_usd": _artifact(object_usd),
            "source_contact": _artifact(source_contact_receipt),
            "l0_result": _artifact(l0_result),
            "strict_v4_result": _artifact(v4_result),
        },
        "stages": steps,
        "productive_run_seconds": time.perf_counter() - started,
        "technical_retry_seconds": 0.0,
        "retry_count": 0,
        "cache_hit": False,
    }
    atomic_write_json(final_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
