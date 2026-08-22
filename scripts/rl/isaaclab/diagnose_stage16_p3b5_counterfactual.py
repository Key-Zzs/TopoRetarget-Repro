#!/usr/bin/env python3
"""Run one frozen-policy C2 A/B/C/D geometry counterfactual.

This diagnostic never trains.  It starts from an exact pre-registered
Contact-ready RSI reset and either replays the captured 26-D action sequence
or evaluates the same frozen C2 actor deterministically.  The only changed
values are construction-time gravity and both material friction coefficients.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/rl/isaaclab"))

from evaluate_physical_hoi import (  # noqa: E402
    _device_trace_to_numpy,
    _initial_trace_snapshot,
    _prepend_initial_trace,
    apply_episode_seed,
    checkpoint_hash,
    model_from_checkpoint,
)
from evaluate_stage16_p3_physical_curriculum import (  # noqa: E402
    DEFAULT_GEOMETRY_MANIFEST,
    _inter_finger_penetration,
    _reconstruct_hand,
)

from toporetarget.rl.c2_geometry_attribution import (  # noqa: E402
    PHYSICS_VARIANTS,
    GeometryGateV1,
    controller_indicators,
    first_violation,
    temporal_classification,
)
from toporetarget.rl.geometry_audit.exact_evaluator import (  # noqa: E402
    evaluate_runtime_proxy_state,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode  # noqa: E402

DEFAULT_CURRICULUM = REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
DEFAULT_SAFE_BANK_ROOT = REPO_ROOT / ".local/reports/stage16_physical_p0_p2/p1"
DEFAULT_V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
DEFAULT_V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
DEFAULT_GATES = DEFAULT_V4_ROOT / "frozen_evaluation_gates.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P3B5_JSON_OBJECT_REQUIRED:{path}")
    return payload


def receipt(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"P3B5_REQUIRED_INPUT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def gate_for_clip(path: Path, clip: str) -> GeometryGateV1:
    gates = read_json(path)["task_gates"]["clips"][clip]
    return GeometryGateV1(
        max_penetration_exclusive_m=float(gates["catastrophic_penetration_m"]),
        p95_penetration_inclusive_m=float(gates["p95_penetration_m"]),
        inter_finger_inclusive_m=float(gates["maximum_inter_finger_penetration_m"]),
    )


def mode_paths(mode: ContactRewardMode) -> tuple[Path, Path]:
    if mode is ContactRewardMode.AGGREGATE_V3:
        return (
            DEFAULT_V3_ROOT / "contact_reward_contract.json",
            REPO_ROOT / ".local/reports/stage16d_reward_v3_contact",
        )
    return DEFAULT_V4_ROOT / "strict_v4_contract.json", DEFAULT_V4_ROOT


def materialize_variant(*, variant: str, friction_scale: float) -> dict[str, dict[str, object]]:
    """Materialize source-identical USDs differing only in both friction fields."""

    output: dict[str, dict[str, object]] = {}
    root = REPO_ROOT / ".local/generated_assets/isaaclab/stage16_p3b5_counterfactual" / variant
    for clip in ("hocap_170105", "hocap_170650"):
        source = REPO_ROOT / ".local/generated_assets/isaaclab" / clip / f"{clip}.usda"
        original = source.read_text(encoding="utf-8")
        static = f"{friction_scale:.8g}"
        dynamic = f"{friction_scale:.8g}"
        if (
            original.count("float physics:staticFriction = 1") != 1
            or original.count("float physics:dynamicFriction = 1") != 1
        ):
            raise RuntimeError(f"P3B5_SOURCE_OBJECT_FRICTION_CONTRACT_DRIFT:{clip}")
        derived = original.replace(
            "float physics:staticFriction = 1", f"float physics:staticFriction = {static}"
        )
        derived = derived.replace(
            "float physics:dynamicFriction = 1", f"float physics:dynamicFriction = {dynamic}"
        )
        target = root / clip / f"{clip}.usda"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_text(encoding="utf-8") != derived:
            target.write_text(derived, encoding="utf-8")
        output[clip] = {
            "source_usd": str(source.resolve()),
            "source_sha256": sha256(source),
            "derived_usd": str(target.resolve()),
            "derived_sha256": sha256(target),
            "only_changed_fields": ["physics:staticFriction", "physics:dynamicFriction"],
            "static_friction": friction_scale,
            "dynamic_friction": friction_scale,
        }
    return output


def configure_variant(*, cfg: Any, variant: str) -> dict[str, object]:
    gravity_scale, friction_scale = PHYSICS_VARIANTS[variant]
    assets = materialize_variant(variant=variant, friction_scale=friction_scale)
    cfg.sim.gravity = (0.0, 0.0, -9.81 * gravity_scale)
    cfg.sim.physics_material.static_friction = 0.5 * friction_scale
    cfg.sim.physics_material.dynamic_friction = 0.5 * friction_scale
    cfg.sim.physics_material.restitution = 0.0
    cfg.object_170105.spawn.rigid_props.disable_gravity = False
    cfg.object_170650.spawn.rigid_props.disable_gravity = False
    cfg.object_170105.spawn.usd_path = str(assets["hocap_170105"]["derived_usd"])
    cfg.object_170650.spawn.usd_path = str(assets["hocap_170650"]["derived_usd"])
    cfg.stage16_gravity_friction_curriculum = str(DEFAULT_CURRICULUM.resolve())
    cfg.stage16_curriculum_stage = f"P3B5_{variant}"
    cfg.stage16_gravity_scale = gravity_scale
    cfg.stage16_friction_scale = friction_scale
    cfg.stage16_curriculum_material_assets = assets
    cfg.stage16_curriculum_material_roles = {
        "global_default_rigid_body": {
            "static_friction": 0.5 * friction_scale,
            "dynamic_friction": 0.5 * friction_scale,
            "restitution": 0.0,
        },
        "hocap_bound_object_material": {
            "static_friction": friction_scale,
            "dynamic_friction": friction_scale,
            "restitution": 0.0,
        },
    }
    return {
        "variant": variant,
        "gravity_scale": gravity_scale,
        "friction_scale": friction_scale,
        "gravity_world_mps2": list(cfg.sim.gravity),
        "object_material_assets": assets,
        "support": "none",
        "external_guidance": False,
        "only_changed_parameters": ["gravity", "hand_friction", "object_friction"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--accept-eula", action="store_true")
    value.add_argument("--mode", choices=("open_loop", "closed_loop"), required=True)
    value.add_argument("--variant", choices=tuple(PHYSICS_VARIANTS), required=True)
    value.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    value.add_argument(
        "--contact-mode", choices=tuple(item.value for item in ContactRewardMode), required=True
    )
    value.add_argument("--episode", type=int, required=True)
    value.add_argument("--trace", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--curriculum-contract", type=Path, default=DEFAULT_CURRICULUM)
    value.add_argument("--safe-bank-root", type=Path, default=DEFAULT_SAFE_BANK_ROOT)
    value.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    value.add_argument("--geometry-manifest", type=Path, default=DEFAULT_GEOMETRY_MANIFEST)
    value.add_argument("--frozen-gates", type=Path, default=DEFAULT_GATES)
    return value


def capture_rollout(
    *,
    env: Any,
    trainer: Any,
    saved_actions: np.ndarray,
    mode: Literal["open_loop", "closed_loop"],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Capture an exact reset snapshot plus one post-physics row per action."""

    import torch

    apply_episode_seed(seed)
    observation, _ = env.reset(seed=seed)
    initial = _initial_trace_snapshot(
        env,
        capture_exact_fingertip_object_pair_force=True,
        capture_full_hand_object_pair_telemetry=True,
    )
    env.start_trace_capture(
        capacity=len(saved_actions),
        capture_exact_fingertip_object_pair_force=True,
        capture_full_hand_object_pair_telemetry=True,
    )
    action_rows: list[np.ndarray] = []
    termination_reason = 0
    for index in range(1, len(saved_actions)):
        if mode == "open_loop":
            action = torch.as_tensor(
                saved_actions[index][None], dtype=torch.float32, device=env.device
            )
        else:
            with torch.no_grad():
                action = trainer.trainer.distribution(observation["policy"]).mean
        action_rows.append(action[0].detach().cpu().numpy().copy())
        observation, _, terminated, timed_out, extras = env.step(action)
        termination_reason = int(extras["ppo26d"]["primary_reason_code"][0].detach().cpu())
        if bool(terminated[0] | timed_out[0]):
            break
    trace = _prepend_initial_trace(
        _device_trace_to_numpy(env.finish_trace_capture()), initial, all_replicas=False
    )
    trace["action"][0] = 0.0
    if len(action_rows) != len(trace["action"]) - 1:
        raise RuntimeError("P3B5_TRACE_ACTION_LENGTH_MISMATCH")
    return trace, {
        "executed_control_steps": len(action_rows),
        "termination_reason": termination_reason,
        "saved_action_sha256": hashlib.sha256(
            saved_actions.astype(np.float32).tobytes()
        ).hexdigest(),
        "executed_action_sha256": hashlib.sha256(
            trace["action"].astype(np.float32).tobytes()
        ).hexdigest(),
        "same_action_preserved": bool(
            mode != "open_loop"
            or np.array_equal(trace["action"], saved_actions[: len(trace["action"])])
        ),
    }


def summarize(
    *, trace: dict[str, np.ndarray], clip: str, gate: GeometryGateV1, manifest: Path
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    trace["hand_collision_body_names"] = np.asarray(HAND_COLLISION_BODY_NAMES)
    trace["hand_collision_body_pose"] = _reconstruct_hand(trace)
    geometry, raw = evaluate_runtime_proxy_state(
        manifest_path=manifest,
        clip=clip,
        object_pose=np.asarray(trace["object_pose"], dtype=np.float64)[:, None],
        hand_collision_body_pose=np.asarray(trace["hand_collision_body_pose"], dtype=np.float64)[
            :, None
        ],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
    )
    frame_worst = np.asarray(raw["frame_worst_penetration_m"], dtype=np.float64)[:, 0]
    frame_pair = np.asarray(raw["frame_worst_pair_index"], dtype=np.int64)[:, 0]
    maximum_frame = int(np.argmax(frame_worst))
    first = first_violation(frame_worst, gate=gate)
    contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1)
    contact_frames = np.flatnonzero(contact)
    first_contact = None if not contact_frames.size else int(contact_frames[0])
    reset_violates = bool(frame_worst[0] > gate.p95_penetration_inclusive_m)
    classification = temporal_classification(
        first_violation_frame=first,
        maximum_frame=maximum_frame,
        frame_count=len(frame_worst),
        first_contact_frame=first_contact,
        reset_violates=reset_violates,
    )
    center = maximum_frame if first is None else first
    controller = controller_indicators(
        finger_target=np.asarray(trace["finger_target"]),
        finger_actual=np.asarray(trace["finger_q"]),
        wrist_target=np.asarray(trace["wrist_target"]),
        wrist_actual=np.asarray(trace["wrist_pose"]),
        actuator_effort=np.asarray(trace["actuator_effort"]),
        # Finger drives are consistently configured at 0.6 effort units.  The
        # first six components are a mixed N/Nm virtual-wrist vector, so they
        # intentionally have no aggregated saturation percentage here.
        effort_limit=0.6,
        contact_force_world=np.asarray(trace["contact_force_world"]),
        center_frame=center,
    )
    actual_object = np.asarray(trace["object_pose"], dtype=np.float64)
    reference_object = np.asarray(trace["object_reference"], dtype=np.float64)
    controller["reference_actual_object_translation_error_m"] = float(
        np.linalg.norm(actual_object[center, :3] - reference_object[center, :3])
    )
    pair_ids = [str(value) for value in raw["pair_ids"].tolist()]
    pair_index = int(frame_pair[maximum_frame])
    inter_finger = _inter_finger_penetration(trace["hand_collision_body_pose"])
    inter_finger_max = float(inter_finger.max(initial=0.0))
    return (
        {
            "gate_pass": gate.passes(
                maximum_m=float(geometry["max_penetration_m"]),
                p95_m=float(geometry["p95_penetration_m"]),
                inter_finger_m=inter_finger_max,
            ),
            "p95_penetration_m": float(geometry["p95_penetration_m"]),
            "active_p95_penetration_m": float(geometry["active_p95_penetration_m"]),
            "max_penetration_m": float(geometry["max_penetration_m"]),
            "inter_finger_max_penetration_m": inter_finger_max,
            "first_violation_frame": first,
            "maximum_penetration_frame": maximum_frame,
            "first_contact_frame": first_contact,
            "last_pre_violation_frame": None if first in {None, 0} else first - 1,
            "violation_duration_frames": int(
                (frame_worst > gate.p95_penetration_inclusive_m).sum()
            ),
            "reset_violates_geometry": reset_violates,
            "temporal_class": classification,
            "violating_pair_index": pair_index,
            "violating_pair": pair_ids[pair_index],
            "violating_hand_body": HAND_COLLISION_BODY_NAMES[pair_index],
            "controller": controller,
            "contact_normal_telemetry": "CONTACT_NORMAL_TELEMETRY_UNAVAILABLE",
            "relative_tangential_velocity": "UNAVAILABLE",
        },
        raw,
        trace,
    )


def main() -> int:
    args = parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    mode = ContactRewardMode.parse(args.contact_mode)
    source_trace_path = args.trace.resolve()
    with np.load(source_trace_path, allow_pickle=False) as archive:
        saved_actions = np.asarray(archive["action"], dtype=np.float32)
    episode_path = source_trace_path.parent.parent / "episodes" / f"episode_{args.episode:03d}.json"
    episode = read_json(episode_path)
    if int(episode["episode"]) != args.episode or episode["rollout"]["clip"] != args.clip:
        raise ValueError("P3B5_EPISODE_TRACE_IDENTITY_MISMATCH")
    reset_index = int(episode["reset_index"])
    seed = int(episode["seed"])
    gate = gate_for_clip(args.frozen_gates.resolve(), args.clip)
    input_receipts = {
        "source_trace": receipt(source_trace_path),
        "episode": receipt(episode_path),
        "checkpoint": receipt(args.checkpoint.resolve()),
        "curriculum_contract": receipt(args.curriculum_contract.resolve()),
        "safe_bank": receipt(
            args.safe_bank_root.resolve() / f"safe_bank_{args.clip.removeprefix('hocap_')}.npz"
        ),
        "geometry_manifest": receipt(args.geometry_manifest.resolve()),
        "frozen_gates": receipt(args.frozen_gates.resolve()),
    }
    contact_contract, _ = mode_paths(mode)
    input_receipts["contact_contract"] = receipt(contact_contract)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env: Any | None = None
    try:
        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        contact_contract, contact_mask_root = mode_paths(mode)
        cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo_cfg.configure_stage16d_ppo26d(
            cfg, num_envs=1, clip=args.clip, rsi=True, critical_dr=False
        )
        ppo_cfg.configure_stage16d_contact_reward(
            cfg,
            mode=mode,
            reference_root=args.reference_root.resolve(),
            contact_reward_contract=contact_contract,
            contact_mask_root=contact_mask_root,
        )
        ppo_cfg.configure_stage16_contact_ready_rsi_v2(
            cfg,
            safe_bank_path=args.safe_bank_root.resolve()
            / f"safe_bank_{args.clip.removeprefix('hocap_')}.npz",
        )
        frozen_variant = configure_variant(cfg=cfg, variant=args.variant)
        cfg.evaluation_reset_reference_indices = (reset_index,)
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        environment_contract = env.contract_report()
        reference_receipts = {
            clip: receipt(Path(path)) for clip, path in sorted(cfg.reference_paths.items())
        }
        trainer, checkpoint_payload = model_from_checkpoint(
            args.checkpoint.resolve(), str(env.device), expected_clip=args.clip
        )
        trace, execution = capture_rollout(
            env=env,
            trainer=trainer,
            saved_actions=saved_actions,
            mode=args.mode,
            seed=seed,
        )
        result, raw_geometry, trace = summarize(
            trace=trace, clip=args.clip, gate=gate, manifest=args.geometry_manifest.resolve()
        )
        output = args.output.resolve()
        trace_path = output / "trace.npz"
        geometry_path = output / "geometry_pairs.npz"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            **trace,
            trace_type=np.asarray("stage16_p3b5_frozen_counterfactual"),
            clip=np.asarray(args.clip),
            checkpoint_path=np.asarray(str(args.checkpoint.resolve())),
            checkpoint_sha256=np.asarray(checkpoint_hash(args.checkpoint.resolve())),
            diagnostic_mode=np.asarray(args.mode),
            physics_variant=np.asarray(args.variant),
            action_contract=np.asarray("26D_reference_residual"),
        )
        np.savez_compressed(geometry_path, **raw_geometry)
        payload = {
            "schema_version": "Stage16P3B5FrozenCounterfactualV1",
            "status": "P3B5_COUNTERFACTUAL_COMPLETE",
            "diagnostic_mode": (
                "OPEN_LOOP_SAME_ACTION_COUNTERFACTUAL"
                if args.mode == "open_loop"
                else "FROZEN_POLICY_COUNTERFACTUAL"
            ),
            "clip": args.clip,
            "contact_mode": mode.value,
            "episode": args.episode,
            "seed": seed,
            "reset_index": reset_index,
            "checkpoint": {
                **input_receipts["checkpoint"],
                "schema": checkpoint_payload["schema_version"],
                "policy_training_samples": int(checkpoint_payload["policy_training_samples"]),
                "normalizer_sha256": hashlib.sha256(
                    repr(checkpoint_payload["observation_normalization"]).encode()
                ).hexdigest(),
            },
            "frozen_inputs": input_receipts,
            "reference_kinematics": reference_receipts,
            "controller_sha256": hashlib.sha256(
                json.dumps(environment_contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "physics": frozen_variant,
            "invariants": {
                "training": False,
                "optimizer_step": False,
                "reward_changed": False,
                "controller_changed": False,
                "collision_proxy_changed": False,
                "support": "none",
                "external_guidance": False,
                **env.rollout_state_write_report(),
            },
            "execution": execution,
            "geometry": result,
            "trace": receipt(trace_path),
            "geometry_sidecar": receipt(geometry_path),
        }
        write_json(output / "result.json", payload)
        print(json.dumps({"status": payload["status"], "output": str(output)}))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
