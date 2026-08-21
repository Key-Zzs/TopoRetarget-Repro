#!/usr/bin/env python3
"""Run one bounded Stage16 PF-V2 symmetric grouped-reward/RSE PPO lineage."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.run_stage16_dexplore_reward_rse import (
    _gradient_sanity,
    _reward_scale,
)
from scripts.rl.isaaclab.train_stage16_p3_physical_curriculum import _restore_zero_g_checkpoint
from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from toporetarget.rl.ppo.policy_preservation import state_hash
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo"
RUN_ROOT = REPO_ROOT / ".local/runs/stage16_pf_v2_causal_lift_and_symmetric_ppo"
OFFLINE_GATE = REPO_ROOT / ".local/reports/stage16_dexplore_reward_rse/offline/offline_gate.json"
AUDIT_GATE = REPORT_ROOT / "pf_v2/audit_classification.json"
EVALUATOR = REPO_ROOT / "scripts/evaluation/evaluate_stage16_dexplore_reward_rse.py"
HISTORICAL_SOURCE_ROOT = (
    REPO_ROOT / ".local/reports/stage16_frozen_source_policy_gravity_sweep/sources"
)
U10_CHECKPOINT = (
    REPO_ROOT / ".local/runs/stage16_dexplore_reward_rse/training/U10/checkpoint/checkpoint.pt"
)
U10_CHECKPOINT_SHA256 = "58f18d934679de4a9759a91cb6b0c296e2357eeff88d23dc9915f85e72cceb95"
CHECKPOINT_SCHEMA = "Stage16DGroupedMultiplicativeRSECheckpointV1"
MAX_NEW_UPDATES = 10
NUM_ENVS = 1024
SAMPLES_PER_UPDATE = NUM_ENVS * 40


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("runtime-sanity", "train"))
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("PF_V2_SYMMETRIC_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_failure(clip: str, scope: str, error: BaseException) -> None:
    path = REPORT_ROOT / "technical_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "clip": clip,
                    "scope": scope,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _require_gates() -> dict[str, object]:
    offline = json.loads(OFFLINE_GATE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_GATE.read_text(encoding="utf-8"))
    freeze_path = REPORT_ROOT / "pf_v2/contract_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("PF_V2_SYMMETRIC_PF_CONTRACT_NOT_FROZEN")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        offline.get("classification") != "MULTIPLICATIVE_RSE_OFFLINE_VALIDATED"
        or offline.get("passed") is not True
        or offline.get("ppo_training_run_authorized") is not True
    ):
        raise RuntimeError("PF_V2_SYMMETRIC_OFFLINE_GATE_NOT_AUTHORIZED")
    if audit.get("ppo_authorized") is not True:
        raise RuntimeError("PF_V2_SYMMETRIC_PF_AUDIT_NOT_AUTHORIZED")
    if freeze.get("classification") != "PF_V2_CONTRACT_FROZEN" or freeze.get("passed") is not True:
        raise RuntimeError("PF_V2_SYMMETRIC_PF_CONTRACT_FREEZE_INVALID")
    return {"offline": offline, "pf_v2_audit": audit, "pf_v2_freeze": freeze}


def _source(clip: str) -> dict[str, object]:
    if clip == "hocap_170105":
        if not U10_CHECKPOINT.is_file() or _sha256(U10_CHECKPOINT) != U10_CHECKPOINT_SHA256:
            raise RuntimeError("PF_V2_SYMMETRIC_U10_SOURCE_HASH_INVALID")
        payload = load_checkpoint(U10_CHECKPOINT, map_location="cpu")
        if (
            payload.get("schema_version") != CHECKPOINT_SCHEMA
            or payload.get("clip") != clip
            or int(payload.get("dexplore_refinement_update", -1)) != 10
            or int(payload.get("dexplore_refinement_samples", -1)) != 409600
            or payload.get("reward_mode") != "grouped_multiplicative_v1"
            or payload.get("rse_enabled") is not True
        ):
            raise RuntimeError("PF_V2_SYMMETRIC_U10_SOURCE_CONTRACT_INVALID")
        return {
            "kind": "existing_dexplore_u10",
            "clip": clip,
            "checkpoint": str(U10_CHECKPOINT.resolve()),
            "checkpoint_sha256": U10_CHECKPOINT_SHA256,
            "initial_update": 10,
            "initial_stage_samples": 409600,
            "initial_cumulative_samples": int(payload["cumulative_samples"]),
            "source_v4_checkpoint": payload["source_v4_checkpoint"],
            "source_v4_checkpoint_sha256": payload["source_v4_checkpoint_sha256"],
        }
    source_path = HISTORICAL_SOURCE_ROOT / "v4_hocap_170650.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(source.get("checkpoint", ""))).resolve()
    if (
        source.get("id") != "v4_hocap_170650"
        or source.get("clip") != clip
        or source.get("contact_mode") != "strict_per_finger_v4"
        or source.get("checkpoint_sha256")
        != "80da5a3c2c953483f9fe5a668dfe2d4b4c458ab451836ad4b179fec28d0979f3"
        or not checkpoint.is_file()
        or _sha256(checkpoint) != source["checkpoint_sha256"]
    ):
        raise RuntimeError("PF_V2_SYMMETRIC_170650_SOURCE_AUTHORITY_INVALID")
    return {
        "kind": "historical_accepted_v4",
        "clip": clip,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "initial_update": 0,
        "initial_stage_samples": 0,
        "initial_cumulative_samples": int(source["source_sample_marker"]["reward_v4_samples"]),
        "historical_qualification_receipt": source["historical_qualification_receipt"],
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": _sha256(source_path),
    }


def _make_env(clip: str) -> Any:
    from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env

    return _make_table_env(
        clip=clip,
        num_envs=NUM_ENVS,
        start_index=0,
        mode=ContactRewardMode.STRICT_PER_FINGER_V4,
        stage="C4",
        training_rsi=True,
        reward_aggregation_mode="grouped_multiplicative_v1",
        rse_enabled=True,
    )


def _runtime_contract(env: Any, *, clip: str) -> dict[str, object]:
    report = env.contract_report()
    physics = report["gravity_friction_curriculum"]
    ppo = report["ppo26d"]
    wrist = report["finite_virtual_6d_wrist_actuator"]
    hand_gravity_off = bool(env.cfg.robot.spawn.rigid_props.disable_gravity)
    if (
        physics.get("stage") != "C4"
        or physics.get("gravity_scale") != 1.0
        or physics.get("friction_scale") != 1.0
        or physics.get("support") != "finite_inferred_table_proxy_v1"
        or physics.get("table_actor_active") is not True
        or physics.get("mid_trajectory_rsi") != "uniform[0,320]"
        or ppo.get("fixed_clip") != clip
        or ppo.get("reward", {}).get("identifier") != "TopoRetargetReferenceTrackingReward26DV4"
        or ppo.get("reward_aggregation", {}).get("mode") != "grouped_multiplicative_v1"
        or ppo.get("rse", {}).get("enabled") is not True
        or ppo.get("rse", {}).get("uniform_rsi_preserved") is not True
        or wrist.get("identifier") != "finite_virtual_6d_wrist_actuator_v1"
        or wrist.get("authority_enabled") is not True
        or not hand_gravity_off
    ):
        raise RuntimeError("PF_V2_SYMMETRIC_RUNTIME_CONTRACT_DRIFT")
    static = {
        "reward_aggregation": ppo["reward_aggregation"],
        "rse": {
            key: value
            for key, value in ppo["rse"].items()
            if key not in {"fail_count", "total_count", "kappa"}
        },
        "physics": {
            key: physics[key]
            for key in (
                "stage",
                "gravity_scale",
                "friction_scale",
                "support",
                "table_actor_active",
                "mid_trajectory_rsi",
                "external_guidance",
            )
        },
        "controller": {
            "identifier": wrist["identifier"],
            "authority_enabled": wrist["authority_enabled"],
        },
    }
    return {"environment": report, "static": static, "static_sha256": _stable_hash(static)}


def _restore_source(env: Any, source: dict[str, object]) -> tuple[PPO26DTrainer, dict[str, object]]:
    trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
    checkpoint = Path(str(source["checkpoint"]))
    if source["kind"] == "historical_accepted_v4":
        initialization = _restore_zero_g_checkpoint(
            trainer,
            checkpoint=checkpoint,
            clip=str(source["clip"]),
            mode=ContactRewardMode.STRICT_PER_FINGER_V4,
        )
        initialization["rse_state"] = "fresh_initial_counts_1_1"
        return trainer, initialization
    payload = load_checkpoint(checkpoint, map_location=env.device)
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = int(payload["cumulative_samples"])
    restore_rng_state(payload["rng"])
    rse = payload["environment_contract"]["ppo26d"]["rse"]
    env.restore_rse_state(fail_count=int(rse["fail_count"]), total_count=int(rse["total_count"]))
    return trainer, {
        "kind": "EXISTING_GROUPED_RSE_FULL_PPO_STATE",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "optimizer_restored": True,
        "critic_restored": True,
        "normalizer_restored": True,
        "rng_restored": True,
        "rse_state_restored": True,
    }


def _run_evaluation(
    *, clip: str, checkpoint: Path, output: Path, update: int, stage_samples: int, episodes: int
) -> dict[str, object]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--accept-eula",
        "--clip",
        clip,
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output.resolve()),
        "--episodes",
        str(episodes),
        "--update",
        str(update),
        "--samples",
        str(stage_samples),
    ]
    environment = dict(os.environ)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, text=True, capture_output=True
    )
    _write_json(
        output.parent / f"{output.name}_driver.json",
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-12000:],
            "stderr_tail": completed.stderr[-12000:],
        },
    )
    if completed.returncode != 0 or (output / "technical_failure.json").is_file():
        raise RuntimeError(f"PF_V2_SYMMETRIC_EVALUATION_FAILED:{clip}:{output}")
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def _runtime_sanity(env: Any, trainer: PPO26DTrainer, *, clip: str) -> dict[str, object]:
    run_root = RUN_ROOT / clip / "runtime_sanity"
    report_root = REPORT_ROOT / "runtime_sanity" / clip
    if run_root.exists() or report_root.exists():
        raise FileExistsError("PF_V2_SYMMETRIC_RUNTIME_SANITY_NAMESPACE_EXISTS")
    run_root.mkdir(parents=True)
    exact_batch = run_root / "exact_batch.pt"
    metric = trainer.collect_and_update(env, update_policy=False, exact_batch_path=exact_batch)
    metric.pop("last_policy_observation")
    gradient = _gradient_sanity(trainer, exact_batch)
    reward_scale = _reward_scale(exact_batch, trainer)
    passed = bool(
        all(metric["finite"].values())
        and gradient["gradient_finite"]
        and gradient["parameters_unchanged"]
        and gradient["optimizer_unchanged"]
        and gradient["normalizer_unchanged"]
        and reward_scale["ppo_contract_comparable"]
        and metric["ppo"]["optimizer_steps"] == 0.0
    )
    gate = {
        "schema_version": "Stage16Pfv2SymmetricRuntimeSanityV1",
        "clip": clip,
        "classification": "RUNTIME_GRADIENT_SANITY_PASS"
        if passed
        else "RUNTIME_GRADIENT_SANITY_FAIL",
        "passed": passed,
        "ppo_training_authorized": passed,
        "optimizer_steps": int(metric["ppo"]["optimizer_steps"]),
        "exact_batch": {"path": str(exact_batch.resolve()), "sha256": _sha256(exact_batch)},
        "rollout": metric,
        "gradient": gradient,
        "reward_scale": reward_scale,
    }
    _write_json(report_root / "gate.json", gate)
    return gate


def _require_runtime_gate(clip: str) -> dict[str, object]:
    gate_path = REPORT_ROOT / "runtime_sanity" / clip / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True or gate.get("ppo_training_authorized") is not True:
        raise RuntimeError("PF_V2_SYMMETRIC_RUNTIME_GATE_NOT_AUTHORIZED")
    return gate


def _progress_row(
    *,
    update: int,
    new_update: int,
    new_samples: int,
    stage_samples: int,
    metric: dict[str, Any],
    evaluation: dict[str, object],
) -> dict[str, object]:
    counts = evaluation["counts"]
    groups = evaluation["group_means"]
    return {
        "update": update,
        "new_update": new_update,
        "new_samples": new_samples,
        "stage_samples": stage_samples,
        "PF_V1": counts["PF_V1"],
        "PF_V2": counts["PF_V2"],
        "physical_lift": counts["physical_lift"],
        "causal_lift": counts["causal_lift"],
        "support_transfer": counts["support_transfer"],
        "sustained_coupling": counts["sustained_hand_object_coupling"],
        "DF_pose": counts["DF_pose"],
        "DF_linear": counts["DF_linear"],
        "DF_angular_v2": counts["DF_angular_v2"],
        "R_obj": groups["R_obj"],
        "R_hand": groups["R_hand"],
        "R_int": groups["R_int"],
        "R_reg": groups["R_reg"],
        "R_total": groups["R_total"],
        "kappa": metric["reference"]["rse"]["kappa"],
        "RSE_termination_rate": metric["termination"]["rse_primary_failure_rate"],
        "first_hand_contact": evaluation["timing"]["first_contact_median"],
        "persistent_multi_contact": evaluation["timing"]["persistent_multi_contact_median"],
        "reference_LIFT": evaluation["timing"]["LIFT"],
        "actual_lift_onset": evaluation["timing"]["actual_lift_onset_median"],
        "pre_reference_LIFT_margin": evaluation["timing"]["pre_LIFT_margin_median"],
    }


def _selection_key(row: dict[str, object]) -> tuple[int, int, int, int]:
    return tuple(int(row[name]) for name in ("PF_V2", "DF_pose", "DF_linear", "DF_angular_v2"))


def assert_symmetric_static_contracts(
    first: dict[str, object], second: dict[str, object]
) -> dict[str, object]:
    """Fail closed unless the experimental lineages match exactly.

    The source actor/checkpoint and fixed clip intentionally differ.  Every
    reward, RSE, physics, controller, PPO, and update-budget field must not.
    """

    keys = (
        "runtime_static",
        "runtime_static_sha256",
        "ppo_hyperparameters",
        "ppo_hyperparameters_sha256",
        "reward_rse_mode",
        "max_new_updates",
        "samples_per_update",
    )
    drift = {
        key: {"first": first.get(key), "second": second.get(key)}
        for key in keys
        if first.get(key) != second.get(key)
    }
    if drift:
        raise RuntimeError(f"PF_V2_SYMMETRIC_STATIC_CONTRACT_DRIFT:{sorted(drift)}")
    return {
        "passed": True,
        "compared_fields": list(keys),
        "static_sha256": first["runtime_static_sha256"],
    }


def _train(
    env: Any, trainer: PPO26DTrainer, *, source: dict[str, object], runtime: dict[str, object]
) -> dict[str, object]:
    clip = str(source["clip"])
    run_lineage = RUN_ROOT / clip
    report_lineage = REPORT_ROOT / "training" / clip
    training_started = run_lineage / "training_started.json"
    if training_started.exists() or (report_lineage / "progression.csv").exists():
        raise FileExistsError("PF_V2_SYMMETRIC_TRAINING_NAMESPACE_EXISTS")
    run_lineage.mkdir(parents=True)
    _write_json(
        training_started,
        {
            "schema_version": "Stage16Pfv2SymmetricTrainingStartV1",
            "clip": clip,
            "resume_supported": False,
            "max_new_updates": MAX_NEW_UPDATES,
        },
    )
    source_update = int(source["initial_update"])
    source_stage_samples = int(source["initial_stage_samples"])
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for new_update in range(1, MAX_NEW_UPDATES + 1):
        update = source_update + new_update
        stage_samples = source_stage_samples + new_update * SAMPLES_PER_UPDATE
        new_samples = new_update * SAMPLES_PER_UPDATE
        label = f"U{update:02d}"
        run_update = run_lineage / "updates" / label
        report_update = report_lineage / label
        checkpoint_path = run_update / "checkpoint/checkpoint.pt"
        batch_path = run_update / "exact_batch/exact_batch.pt"
        checkpoint_path.parent.mkdir(parents=True)
        batch_path.parent.mkdir(parents=True)
        metric = trainer.collect_and_update(env, exact_batch_path=batch_path)
        metric.pop("last_policy_observation")
        if int(metric["samples"]) != SAMPLES_PER_UPDATE or not all(metric["finite"].values()):
            raise RuntimeError("PF_V2_SYMMETRIC_TRAINING_UPDATE_CONTRACT_INVALID")
        payload = trainer.checkpoint_payload(
            environment_contract=env.contract_report(),
            selected_num_envs=NUM_ENVS,
            extra_payload={
                "dexplore_refinement_update": update,
                "dexplore_refinement_samples": stage_samples,
                "pf_v2_symmetric_clip": clip,
                "pf_v2_symmetric_new_update": new_update,
                "pf_v2_symmetric_new_samples": new_samples,
                "pf_v2_symmetric_source_checkpoint": source["checkpoint"],
                "pf_v2_symmetric_source_checkpoint_sha256": source["checkpoint_sha256"],
                "reward_mode": "grouped_multiplicative_v1",
                "rse_enabled": True,
                "max_new_updates": MAX_NEW_UPDATES,
            },
        )
        payload["schema_version"] = CHECKPOINT_SCHEMA
        save_checkpoint(checkpoint_path, payload)
        evaluation = _run_evaluation(
            clip=clip,
            checkpoint=checkpoint_path,
            output=report_update / "eval10",
            update=update,
            stage_samples=stage_samples,
            episodes=10,
        )
        row = _progress_row(
            update=update,
            new_update=new_update,
            new_samples=new_samples,
            stage_samples=stage_samples,
            metric=metric,
            evaluation=evaluation,
        )
        rows.append(row)
        candidate = {
            "update": update,
            "new_update": new_update,
            "new_samples": new_samples,
            "stage_samples": stage_samples,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "selection_key": list(_selection_key(row)),
            "eval10": evaluation,
        }
        if best is None or tuple(candidate["selection_key"]) > tuple(best["selection_key"]):
            best = candidate
        receipt = {
            "schema_version": "Stage16Pfv2SymmetricPPOUpdateReceiptV1",
            "clip": clip,
            "source": source,
            "runtime_static_sha256": runtime["static_sha256"],
            "update": update,
            "new_update": new_update,
            "new_samples": new_samples,
            "stage_samples": stage_samples,
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            },
            "exact_batch": {"path": str(batch_path.resolve()), "sha256": _sha256(batch_path)},
            "state_hashes": {
                "actor": parameter_hash(trainer.model, "actor"),
                "critic": parameter_hash(trainer.model, "critic"),
                "optimizer": state_hash(payload["optimizer"]),
                "normalizer": state_hash(payload["observation_normalization"]),
                "rng": state_hash(payload["rng"]),
            },
            "training": metric,
            "eval10": evaluation,
            "positive_control_training_regression": bool(
                clip == "hocap_170650" and int(evaluation["counts"]["PF_V2"]) < 10
            ),
        }
        _write_json(report_update / "receipt.json", receipt)
        _write_csv(report_lineage / "progression.csv", rows)
        _write_json(
            run_lineage / "progress.json",
            {"completed_new_updates": new_update, "new_samples": new_samples, "best": best},
        )
        if clip == "hocap_170105" and int(evaluation["counts"]["PF_V2"]) == 10:
            confirm = _run_evaluation(
                clip=clip,
                checkpoint=checkpoint_path,
                output=report_lineage / "confirm20" / label,
                update=update,
                stage_samples=stage_samples,
                episodes=20,
            )
            receipt["confirm20"] = confirm
            _write_json(report_update / "receipt.json", receipt)
            break
    if best is None:
        raise RuntimeError("PF_V2_SYMMETRIC_NO_UPDATE_COMPLETED")
    _write_json(report_lineage / "best_checkpoint.json", best)
    result = {
        "schema_version": "Stage16Pfv2SymmetricPPOTrainingV1",
        "clip": clip,
        "experimental_refinement_lineage": True,
        "historical_170650_actor_overwritten": False,
        "max_new_updates": MAX_NEW_UPDATES,
        "actual_new_updates": len(rows),
        "actual_new_samples": int(rows[-1]["new_samples"]),
        "best_checkpoint": best,
    }
    _write_json(run_lineage / "complete.json", result)
    return result


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.num_envs != NUM_ENVS:
        raise ValueError("PF_V2_SYMMETRIC_REQUIRES_EULA_AND_1024_ENVS")
    _require_gates()
    source = _source(args.clip)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        env = _make_env(args.clip)
        env.reset(seed=20260822)
        runtime = _runtime_contract(env, clip=args.clip)
        trainer, initialization = _restore_source(env, source)
        static_contract = {
            "clip": args.clip,
            "source": source,
            "runtime_static": runtime["static"],
            "runtime_static_sha256": runtime["static_sha256"],
            "ppo_hyperparameters": asdict(trainer.training_contract),
            "ppo_hyperparameters_sha256": _stable_hash(asdict(trainer.training_contract)),
            "reward_rse_mode": "grouped_multiplicative_v1_with_rse_v1",
            "max_new_updates": MAX_NEW_UPDATES,
            "samples_per_update": SAMPLES_PER_UPDATE,
        }
        if args.mode == "runtime-sanity":
            _write_json(
                RUN_ROOT / args.clip / "runtime_sanity_contract.json",
                {"initialization": initialization, **static_contract},
            )
            result = _runtime_sanity(env, trainer, clip=args.clip)
        else:
            _write_json(
                REPORT_ROOT / "training" / args.clip / "lineage_contract.json", static_contract
            )
            _write_json(
                RUN_ROOT / args.clip / "training_contract.json",
                {"initialization": initialization, **static_contract},
            )
            _require_runtime_gate(args.clip)
            result = _train(env, trainer, source=source, runtime=runtime)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        _append_failure(args.clip, args.mode, error)
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
