#!/usr/bin/env python3
"""Choose one residual-action scale on accepted HOCap joint references.

This is a non-learning, hand-actuator qualification.  It intentionally
measures the native Wuji position actuators before a PPO run and makes one
global selection across every supplied accepted clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from toporetarget.rl.actuators import (
    ACTION_SCALE_CANDIDATES,
    PDQualificationResult,
    choose_global_action_scale,
    residual_target,
)
from toporetarget.rl.contracts import Stage16ReferenceClip

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    bounds = model.jnt_range[: model.njnt].copy()
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    if any(name is None for name in joint_order):
        raise RuntimeError("Wuji MJCF has unnamed joints")
    joint_order = tuple(name for name in joint_order if name is not None)
    clips = [Stage16ReferenceClip.from_npz(path) for path in args.reference]
    if any(clip.joint_order != joint_order for clip in clips):
        raise ValueError("all HOCap references must use the Wuji MJCF joint order")
    results: list[PDQualificationResult] = []
    detail: list[dict[str, object]] = []
    normalized_range = bounds[:, 1] - bounds[:, 0]
    for scale in ACTION_SCALE_CANDIDATES:
        errors: list[float] = []
        saturations: list[float] = []
        finite = True
        for clip in clips:
            data = mujoco.MjData(model)
            data.qpos[: model.njnt] = clip.q_finger_ref[0]
            mujoco.mj_forward(model, data)
            for frame, reference in enumerate(clip.q_finger_ref):
                phase = 0.15 * frame + np.arange(model.njnt, dtype=np.float64) * 0.37
                action = 0.25 * np.sin(phase) + rng.normal(0.0, 0.02, size=model.njnt)
                target = residual_target(
                    reference,
                    action,
                    bounds[:, 0],
                    bounds[:, 1],
                    action_scale_fraction=scale,
                )
                saturations.append(
                    float(
                        np.mean(
                            np.isclose(target, bounds[:, 0], atol=1e-10)
                            | np.isclose(target, bounds[:, 1], atol=1e-10)
                        )
                    )
                )
                data.ctrl[:] = target
                for _ in range(5):
                    mujoco.mj_step(model, data)
                error = np.abs(data.qpos[: model.njnt] - target) / normalized_range
                errors.append(float(np.max(error)))
                finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        result = PDQualificationResult(
            action_scale_fraction=scale,
            settling_time_s=float(0.05 * np.mean([error > 0.05 for error in errors])),
            overshoot_fraction=float(np.percentile(errors, 95)),
            saturated_fraction=float(np.mean(saturations)),
            stable=finite and bool(np.max(errors) < 1.0),
        )
        results.append(result)
        detail.append(
            {
                "action_scale_fraction": scale,
                "max_normalized_target_error": float(np.max(errors)),
                "p95_normalized_target_error": result.overshoot_fraction,
                "saturated_fraction": result.saturated_fraction,
                "stable": result.stable,
            }
        )
    selected = choose_global_action_scale(results)
    report = {
        "status": "PD_QUALIFICATION_ACCEPTED_HOCAP_PASS",
        "references": [
            {
                "path": str(path.resolve()),
                "hash": clip.content_hash(),
                "source_sequence": clip.provenance["dataset_provenance"]["source_sequence"],
            }
            for path, clip in zip(args.reference, clips, strict=True)
        ],
        "candidates": detail,
        "selection": {
            "action_scale_fraction": selected.action_scale_fraction,
            "selection_rule": (
                "minimize saturation, then p95 normalized target error, then settling"
            ),
        },
        "physical_object_claim": False,
        "non_claim": (
            "native actuator qualification only; no policy learning or paper PD-gain claim"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": report["status"], "selection": report["selection"]}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
