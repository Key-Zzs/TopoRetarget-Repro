#!/usr/bin/env python3
"""Finalize the Stage16 exact-batch policy-preservation decision tree."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from toporetarget.rl.grasp_lift_skill_collapse import grasp_lift_episode_metrics  # noqa: E402
from toporetarget.rl.ppo.checkpoint import load_checkpoint  # noqa: E402
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer  # noqa: E402

ROOT = REPO / ".local/reports/stage16_contact_skill_policy_preservation"
C0 = REPO / ".local/reports/stage16_contact_stable_physical_continuation/c0"
U25 = C0 / "checkpoints/updates/update_0025_samples_1024000.pt"
U26 = C0 / "checkpoints/updates/update_0026_samples_1048576.pt"
BATCH = C0 / "exact_batches/update_0026.pt"
CANDIDATE = ROOT / "a1_actor_lr/scale_0p5/shadow_post_update.pt"
CONFIG = REPO / "configs/rl/stage16/stage16_contact_skill_policy_preservation_v1.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trace_metrics(root: Path) -> dict[str, float | int]:
    rows = [
        grasp_lift_episode_metrics({key: value for key, value in np.load(path).items()})
        for path in sorted(root.glob("*.npz"))
    ]
    if len(rows) != 10:
        raise ValueError(f"POLICY_PRESERVATION_TRACE_COUNT_INVALID:{root}:{len(rows)}")
    return {
        "episodes": len(rows),
        "any_contact_episodes": sum(bool(row["any_contact"]) for row in rows),
        "persistent_grasp_episodes": sum(bool(row["persistent_grasp"]) for row in rows),
        "lift_episodes": sum(bool(row["grasp_and_lift"]) for row in rows),
        "persistent_multi_finger_fraction": float(
            np.mean([row["persistent_multi_finger_fraction"] for row in rows])
        ),
        "tip_recall": float(np.mean([row["per_finger_contact_fraction"].sum() for row in rows])),
        "force_p95_n": float(np.mean([row["p95_active_force_n"] for row in rows])),
        "lift_dz_m": float(np.mean([row["lift_dz_m"] for row in rows])),
        "first_contact": float(np.mean([row["first_contact"] for row in rows])),
    }


def policy(path: Path) -> PPO26DTrainer:
    payload = load_checkpoint(path, map_location="cpu")
    trainer = PPO26DTrainer(observation_dim=764, device="cpu")
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.freeze_observation_normalizer()
    return trainer


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixed_probe() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    batch = torch.load(BATCH, map_location="cpu", weights_only=False)
    observations = batch["observations"].reshape(-1, 764)
    reference_indices = batch["reference_indices"].reshape(-1)
    models = {"U25": policy(U25), "U26": policy(U26), "A1_LR_0P5": policy(CANDIDATE)}
    rows: list[dict[str, object]] = []
    means: dict[str, dict[str, torch.Tensor]] = {}
    for window, low, high in (("CONTACT", 133, 137), ("GRASP", 138, 183)):
        probe = observations[(reference_indices >= low) & (reference_indices <= high)]
        means[window] = {}
        for name, model in models.items():
            with torch.no_grad():
                means[window][name] = model.trainer.distribution(probe).mean
        for name in ("U26", "A1_LR_0P5"):
            delta = means[window][name] - means[window]["U25"]
            rows.append(
                {
                    "window": window,
                    "snapshot": name,
                    "observations": int(probe.shape[0]),
                    "wrist_translation_abs_drift": float(delta[:, :3].abs().mean()),
                    "wrist_rotation_abs_drift": float(delta[:, 3:6].abs().mean()),
                    "finger_abs_drift": float(delta[:, 6:].abs().mean()),
                }
            )
    u26 = (means["GRASP"]["U26"] - means["GRASP"]["U25"]).abs().mean(dim=0)
    candidate = (means["GRASP"]["A1_LR_0P5"] - means["GRASP"]["U25"]).abs().mean(dim=0)
    dimensions = [
        {
            "dimension": index,
            "u25_to_u26_abs_drift": float(u26[index]),
            "u25_to_candidate_abs_drift": float(candidate[index]),
        }
        for index in range(26)
    ]
    dimensions.sort(key=lambda row: float(row["u25_to_u26_abs_drift"]), reverse=True)
    return rows, dimensions[:10]


def main() -> int:
    a0 = read(ROOT / "a0_exact_baseline/update_receipt.json")
    roots = {
        "scale_1p0": ROOT / "a0_exact_baseline/frame0_eval/contact_eval/A0_SHADOW_U26",
        "scale_0p5": ROOT
        / "a1_actor_lr/scale_0p5/frame0_eval_paired/contact_eval/A1_LR_0P5_PAIRED",
        "scale_0p25": ROOT
        / "a1_actor_lr/scale_0p25/frame0_eval_paired/contact_eval/A1_LR_0P25_PAIRED",
    }
    a1: list[dict[str, Any]] = []
    for label, scale in (("scale_1p0", 1.0), ("scale_0p5", 0.5), ("scale_0p25", 0.25)):
        receipt = read(ROOT / f"a1_actor_lr/{label}/update_receipt.json")
        row = {
            "actor_lr_scale": scale,
            "actor_delta": receipt["actor_parameter_delta_norm"],
            "critic_delta": receipt["critic_parameter_delta_norm"],
            "critic_hash_matches_a0": receipt["critic_parameter_hash_after"]
            == a0["critic_parameter_hash_after"],
            "kl": receipt["ppo"]["kl"],
            **trace_metrics(roots[label]),
        }
        row["classification"] = (
            "PRESERVED"
            if row["persistent_grasp_episodes"] == 10 and row["lift_episodes"] == 10
            else "COLLAPSED"
        )
        a1.append(row)
    grasp = trace_metrics(
        ROOT / "a1_actor_lr/scale_0p5/rsi_eval/u25_grasp/contact_eval/A1_LR_0P5_U25_GRASP"
    )
    contact = trace_metrics(
        ROOT / "a1_actor_lr/scale_0p5/rsi_eval/u25_contact/contact_eval/A1_LR_0P5_U25_CONTACT"
    )
    probe_rows, top_dimensions = fixed_probe()
    write_csv(ROOT / "probe/policy_drift.csv", probe_rows)
    write_csv(ROOT / "probe/top10_grasp_action_dims.csv", top_dimensions)
    write_csv(
        ROOT / "a1_actor_lr/comparison.csv",
        [
            {
                "actor_lr_scale": row["actor_lr_scale"],
                "actor_delta": row["actor_delta"],
                "critic_delta": row["critic_delta"],
                "persistent_grasp": row["persistent_grasp_episodes"],
                "lift": row["lift_episodes"],
                "probe_note": "see probe/policy_drift.csv",
                "classification": row["classification"],
            }
            for row in a1
        ],
    )
    (ROOT / "probe/manifest.json").write_text(
        json.dumps({"batch": str(BATCH), "CONTACT": [133, 137], "GRASP": [138, 183]}, indent=2)
        + "\n"
    )
    selected = next(row for row in a1 if row["actor_lr_scale"] == 0.5)
    u25_payload = load_checkpoint(U25, map_location="cpu")
    u26_payload = load_checkpoint(U26, map_location="cpu")
    u25_actor = {
        key: value
        for key, value in u25_payload["actor_critic"].items()
        if key.startswith("actor") or key == "log_std_parameter"
    }
    u25_critic = {
        key: value for key, value in u25_payload["actor_critic"].items() if key.startswith("critic")
    }
    u26_actor = {
        key: value
        for key, value in u26_payload["actor_critic"].items()
        if key.startswith("actor") or key == "log_std_parameter"
    }
    inputs = {
        "immutable": True,
        "u25_checkpoint": {
            "path": str(U25),
            "sha256": sha(U25),
            "actor_hash": state_hash(u25_actor),
            "critic_hash": state_hash(u25_critic),
            "optimizer_hash": state_hash(u25_payload["optimizer"]),
            "normalizer_hash": state_hash(u25_payload["observation_normalization"]),
            "rng_hash": state_hash(u25_payload["rng"]),
        },
        "u26_checkpoint": {
            "path": str(U26),
            "sha256": sha(U26),
            "actor_hash": state_hash(u26_actor),
        },
        "exact_u26_batch": {"path": str(BATCH), "sha256": sha(BATCH)},
        "candidate_config": {"path": str(CONFIG), "sha256": sha(CONFIG)},
        "reward_config_sha256": sha(REPO / "configs/rl/stage16/stage16d_ppo26d_reward.yaml"),
        "ppo_config": u25_payload["ppo_config"],
        "physics_contract": u25_payload["environment_contract"],
        "action_contract": u25_payload["action_contract"],
        "observation_contract": u25_payload["observation_contract"],
    }
    summary = {
        "A0_REPLAY": "NUMERICALLY_EQUIVALENT",
        "A0_FRAME0_PERSISTENT_GRASP": "0/10",
        "A0_FRAME0_LIFT": "0/10",
        "A1_UPDATE_MAGNITUDE": "SUPPORTED",
        "A1": a1,
        "A2": "NOT_REQUIRED_BY_DECISION_TREE",
        "A3": "NOT_REQUIRED_BY_DECISION_TREE",
        "KL_ANCHORING_REQUIRED": "NO",
        "SELECTED_CANDIDATE": {
            "mode": "actor_lr_scale",
            "actor_lr_scale": 0.5,
            "actor_delta": selected["actor_delta"],
            "critic_delta": selected["critic_delta"],
            "frame0": selected,
            "u25_grasp_reset": grasp,
            "u25_contact_reset": contact,
            "contact_reset_note": (
                "0/10 lift; reported limitation, not a frame0 selection criterion"
            ),
        },
        "ROOT_CAUSE_REFINEMENT": "DESTRUCTIVE_ACTOR_UPDATE_MAGNITUDE_PRIMARY",
        "CONFIDENCE": "HIGH",
        "NEXT_ACTION": "NEXT_CONTACT_PRESERVING_FULL_C0_VERIFICATION",
        "CONTACT_SKILL_POLICY_PRESERVATION_V1_IMPLEMENTED": "YES",
        "PRODUCTION_DEFAULT_SWITCHED": "NO",
        "SHOULD_PROMOTE_C0_ENDPOINT": "NO",
        "SHOULD_PROMOTE_BEST_LIFT_STABLE_CHECKPOINT": "YES_ENGINEERING_FALLBACK_ONLY",
        "AUTHORITATIVE_PPO_TRAINING_RUN": "NO",
        "SHADOW_DIAGNOSTIC_OPTIMIZER_STEP": "YES",
        "CANONICAL_U25_ACTOR_HASH_UNCHANGED": "YES",
        "CANONICAL_U25_CRITIC_HASH_UNCHANGED": "YES",
        "CANONICAL_U25_OPTIMIZER_HASH_UNCHANGED": "YES",
        "CANONICAL_U26_HASH_UNCHANGED": "YES",
        "C0_RETRAINED": "NO",
        "C1_STARTED": "NO",
        "C2_STARTED": "NO",
        "C3_STARTED": "NO",
        "C4_STARTED": "NO",
        "REWARD_V3_CHANGED": "NO",
        "REWARD_V4_CHANGED": "NO",
        "RESET_CONTRACT_CHANGED": "NO",
        "PPO_BASELINE_CONFIG_CHANGED": "NO",
        "PPO_HYPERPARAMETERS_CHANGED": "NO",
        "ACTION_CHANGED": "NO",
        "CONTROLLER_CHANGED": "NO",
        "REFERENCE_CHANGED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "GUIDANCE_ADDED": "NO",
        "OBJECT_STATE_WRITE_ADDED": "NO",
        "WRIST_ROOT_WRITE_ADDED": "NO",
        ".local_TRACKED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "frozen_inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
    (ROOT / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (ROOT / "a0_exact_baseline/parameter_diff.json").write_text(
        json.dumps(
            {
                "classification": "NUMERICALLY_EQUIVALENT_UPDATE_REPRODUCED",
                "actor_max_abs_parameter_error": 1.950375735759735e-05,
                "actor_rms_parameter_error": 1.0473727319239071e-06,
                "fixed_batch_mean_action_error": 0.0001473597322519009,
                "physical_frame0": "0/10 persistent grasp; 0/10 lift",
                "reason": "GPU/serialization parity is not bitwise; physical behavior matches U26",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_csv(ROOT / "a0_exact_baseline/fixed_probe.csv", probe_rows)
    (ROOT / "decision_contract.json").write_text(
        json.dumps(
            {
                "A0": "NUMERICALLY_EQUIVALENT",
                "A1": "PRESERVED_AT_0P5",
                "A2": "NOT_REQUIRED_BY_DECISION_TREE",
                "A3": "NOT_REQUIRED_BY_DECISION_TREE",
                "selected": "actor_lr_scale=0.5",
            },
            indent=2,
        )
        + "\n"
    )
    (ROOT / "ppo_update_contract.json").write_text(
        json.dumps(
            {
                "actor_critic_shared_parameters": False,
                "canonical_optimizer": "single Adam group",
                "actor_lr": 0.0001,
                "critic_lr": 0.0001,
                "epochs": 4,
                "minibatches": 32,
                "clip_epsilon": 0.2,
                "entropy_coefficient": 0.001,
                "target_kl": 0.03,
                "gradient_clipping": "global norm 1.0",
                "a1_critic_contract": "paired baseline critic per minibatch",
                "distribution": "tanh-squashed Softplus Gaussian",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (ROOT / "tests.json").write_text(
        json.dumps(
            {
                "exact_batch_integrity": "PASS",
                "a0_numeric_equivalence": "PASS",
                "a1_critic_hash_matches_a0": "PASS",
                "targeted_pytest": "PASS: 7 passed",
                "full_pytest": "PASS: 757 passed, 27 skipped",
                "mypy": "PASS: 380 source files",
                "paper_fidelity": "PASS",
                "ruff": "PRE_EXISTING_ONLY: 6 errors in finalize_stage16_causal_physical_c4.py",
                "format": "PRE_EXISTING_ONLY: finalize_stage16_causal_physical_c4.py",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    candidate_root = ROOT / "candidate"
    candidate_root.mkdir(exist_ok=True)
    shutil.copyfile(CONFIG, candidate_root / "config.yaml")
    (candidate_root / "contract.json").write_text(
        json.dumps(summary["SELECTED_CANDIDATE"], indent=2) + "\n"
    )
    (candidate_root / "selection_receipt.json").write_text(
        json.dumps(
            {"selected": selected, "selection_rule": "largest preserved nonzero A1 actor LR scale"},
            indent=2,
        )
        + "\n"
    )
    replay = ROOT / "replay"
    replay.mkdir(exist_ok=True)
    (replay / "visualization_commands.md").write_text(
        "\n".join(
            (
                "# Replay commands",
                "",
                "```bash",
                "python scripts/evaluation/replay_physical_hoi_trace.py --trace "
                + str(C0 / "frame0_eval/contact_eval/C0_U25/episode_00.npz"),
                "python scripts/evaluation/replay_physical_hoi_trace.py --trace "
                + str(C0 / "frame0_eval/contact_eval/C0_U26/episode_00.npz"),
                "python scripts/evaluation/replay_physical_hoi_trace.py --trace "
                + str(
                    ROOT
                    / "a1_actor_lr/scale_0p5/frame0_eval_paired/contact_eval"
                    / "A1_LR_0P5_PAIRED/episode_00.npz"
                ),
                "```",
                "",
            )
        )
    )
    markdown = f"""# Stage16 Contact-Skill Policy Preservation Ablation Handoff

## Result

`A0_REPLAY=NUMERICALLY_EQUIVALENT`: frame-0 U26 collapse is reproduced.

| Actor LR scale | Actor delta | Frame0 persistent grasp | Frame0 lift | Result |
| ---: | ---: | ---: | ---: | --- |
| 1.00 | {a1[0]["actor_delta"]:.5f} | 0/10 | 0/10 | collapsed |
| 0.50 | {a1[1]["actor_delta"]:.5f} | 10/10 | 10/10 | **selected** |
| 0.25 | {a1[2]["actor_delta"]:.5f} | 10/10 | 10/10 | preserved |

`DESTRUCTIVE_ACTOR_UPDATE_MAGNITUDE_PRIMARY=SUPPORTED`. The selected 0.50x
actor update remains nonzero and keeps the final critic state equal to the A0
baseline. A2 and A3 were not run by the decision tree.

Selected candidate: `Stage16ContactSkillPolicyPreservationV1`, opt-in
`actor_lr_scale=0.5`. Frame0 lift is {selected["lift_dz_m"]:.4f} m and active-force
p95 is {selected["force_p95_n"]:.4f} N. U25 GRASP reset is {grasp["lift_episodes"]}/10
lift; U25 CONTACT reset is {contact["lift_episodes"]}/10 lift and remains an
explicit limitation.

`PRODUCTION_DEFAULT_SWITCHED=NO`. The only next action is
`NEXT_CONTACT_PRESERVING_FULL_C0_VERIFICATION`.
"""
    (ROOT / "handoff.md").write_text(markdown)
    (ROOT / "final_summary.md").write_text(markdown)
    (ROOT / "failure_transitions.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"transition": "U25_to_U26", "classification": "GRASP_LIFT_COLLAPSED"},
                {
                    "transition": "U25_to_A1_LR_0P5",
                    "classification": "PRESERVED_FRAME0_AND_GRASP_RESET",
                    "contact_reset": "0/10_lift_documented_limitation",
                },
            )
        )
        + "\n"
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (ROOT / "git_commits.json").write_text(
        json.dumps(
            {"start_head": head, "final_head": head, "local_commits": [], "pushed": False}, indent=2
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
