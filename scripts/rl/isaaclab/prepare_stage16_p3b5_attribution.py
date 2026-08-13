#!/usr/bin/env python3
"""Freeze C2 attribution inputs and select deterministic P3-B.5 cases.

This is an offline reader of historical C0/C1/C2 evaluation artifacts.  It
creates a new report lineage only; it never touches historical qualifications,
checkpoints, safe banks, or PPO state.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.c2_geometry_attribution import GeometryGateV1, decision_contract


PILOT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_p4_full_gravity/physical_pilot"
SAFE_BANK_ROOT = REPO_ROOT / ".local/reports/stage16_physical_p0_p2/p1"
CURRICULUM = REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
GATES = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4/frozen_evaluation_gates.json"
MANIFEST = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json"
)
OUTPUT = REPO_ROOT / ".local/reports/stage16_p3b5_geometry_attribution"

MODES = {"v3": "aggregate_v3", "v4": "strict_per_finger_v4"}
CLIPS = ("hocap_170105", "hocap_170650")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"P3B5_REQUIRED_INPUT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P3B5_JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def safe_bank_classes(clip: str) -> dict[int, dict[str, str]]:
    path = SAFE_BANK_ROOT / f"safe_bank_{clip.removeprefix('hocap_')}.npz"
    with np.load(path, allow_pickle=False) as archive:
        indices = np.asarray(archive["runtime_index"], dtype=np.int64)
        semantic = np.asarray(archive["semantic_class"])
        bank = np.asarray(archive["safe_bank"])
    return {
        int(index): {"semantic_class": str(item), "safe_bank": str(label)}
        for index, item, label in zip(indices, semantic, bank, strict=True)
    }


def gate_for_clip(gates: dict[str, Any], clip: str) -> GeometryGateV1:
    item = gates["task_gates"]["clips"][clip]
    return GeometryGateV1(
        max_penetration_exclusive_m=float(item["catastrophic_penetration_m"]),
        p95_penetration_inclusive_m=float(item["p95_penetration_m"]),
        inter_finger_inclusive_m=float(item["maximum_inter_finger_penetration_m"]),
    )


def episode_inventory(
    *,
    mode: str,
    clip: str,
    qualification: dict[str, Any],
    safe_classes: dict[int, dict[str, str]],
    p95_limit_m: float,
) -> list[dict[str, object]]:
    root = PILOT_ROOT / mode / clip / "c2/dev"
    result: list[dict[str, object]] = []
    for summary in qualification["episodes"]:
        episode = int(summary["episode"])
        detail_path = root / "episodes" / f"episode_{episode:03d}.json"
        detail = read_json(detail_path)
        geometry_path = root / "geometry" / f"episode_{episode:03d}_pairs.npz"
        with np.load(geometry_path, allow_pickle=False) as raw:
            worst = np.asarray(raw["frame_worst_penetration_m"], dtype=np.float64)[:, 0]
            worst_pairs = np.asarray(raw["frame_worst_pair_index"], dtype=np.int64)[:, 0]
            pair_ids = [str(item) for item in raw["pair_ids"].tolist()]
        violating = np.flatnonzero(worst > p95_limit_m)
        maximum_frame = int(np.argmax(worst))
        pair_index = int(worst_pairs[maximum_frame])
        hand_object = detail["penetration"]["hand_object"]
        reset = int(summary["reset_index"])
        rsi = safe_classes.get(reset, {"semantic_class": "NOT_IN_SAFE_BANK", "safe_bank": "NONE"})
        result.append(
            {
                "mode": mode,
                "contact_mode": MODES[mode],
                "clip": clip,
                "episode": episode,
                "seed": int(summary["seed"]),
                "reset_index": reset,
                "rsi_class": rsi["semantic_class"],
                "safe_bank": rsi["safe_bank"],
                "absolute_geometry_pass": bool(summary["absolute_geometry_pass"]),
                "hand_object_p95_penetration_m": float(hand_object["p95_penetration_m"]),
                "hand_object_max_penetration_m": float(hand_object["max_penetration_m"]),
                "active_p95_penetration_m": float(hand_object["active_p95_penetration_m"]),
                "inter_finger_max_penetration_m": float(
                    detail["penetration"]["inter_finger_max_penetration_m"]
                ),
                "terminal_contact": bool(summary["terminal_contact"]),
                "terminal_stability": bool(detail["diagnostics"]["terminal_stability"]),
                "physics_success": bool(summary["physics_success"]),
                "qualified_success": bool(summary["qualified_success"]),
                "first_violating_frame": None if not violating.size else int(violating[0]),
                "maximum_penetration_frame": maximum_frame,
                "violating_collision_pair": pair_ids[pair_index],
                "violating_hand_body": pair_ids[pair_index].split("/")[2],
                "object_collision_proxy": pair_ids[pair_index].split("<->", 1)[1],
                "geometry_sidecar": receipt(geometry_path),
                "episode_detail": receipt(detail_path),
                "trace": receipt(root / "traces" / f"episode_{episode:03d}.npz"),
            }
        )
    return result


def historical_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in MODES:
        for clip in CLIPS:
            for stage in ("c0", "c1", "c2"):
                qualification = read_json(
                    PILOT_ROOT / mode / clip / stage / "dev/qualification.json"
                )
                penetration = qualification["penetration"]
                twist = qualification["twist"]
                rows.append(
                    {
                        "mode": mode,
                        "clip": clip,
                        "stage": stage.upper(),
                        "gravity_scale": float(
                            qualification["curriculum_physics"]["gravity_scale"]
                        ),
                        "friction_scale": float(
                            qualification["curriculum_physics"]["friction_scale"]
                        ),
                        "p95_penetration_m": float(penetration["hand_object_p95_penetration_m"]),
                        "max_penetration_m": float(penetration["hand_object_max_penetration_m"]),
                        "inter_finger_max_penetration_m": float(
                            penetration["interfinger_max_penetration_m"]
                        ),
                        "contact_force_n": "NOT_REPORTED_AT_STAGE_AGGREGATE",
                        "SRphysics": float(
                            qualification["evaluation_suite_v2"]["aggregate"]["physics_success"][
                                "rate"
                            ]
                        ),
                        "Delta_v_mean_mps": float(twist["Delta_v_mps"]["mean"]),
                        "Delta_omega_mean_radps": float(twist["Delta_omega_radps"]["mean"]),
                        "qualification": receipt(
                            PILOT_ROOT / mode / clip / stage / "dev/qualification.json"
                        ),
                    }
                )
    return rows


def representative_cases(inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    v3_105 = sorted(
        [
            row
            for row in inventory
            if row["clip"] == "hocap_170105"
            and row["mode"] == "v3"
            and not bool(row["absolute_geometry_pass"])
        ],
        key=lambda row: (float(row["hand_object_p95_penetration_m"]), int(row["episode"])),
    )
    maximum = max(
        v3_105,
        key=lambda row: (float(row["hand_object_max_penetration_m"]), -int(row["episode"])),
    )
    # The lower median can tie the maximum episode.  Prefer the closest
    # non-maximum row so the prescribed median and maximum diagnostics remain
    # two independently informative reset cases.
    values = np.asarray([float(row["hand_object_p95_penetration_m"]) for row in v3_105])
    median_value = float(np.median(values))
    median = min(
        (row for row in v3_105 if int(row["episode"]) != int(maximum["episode"])),
        key=lambda row: (
            abs(float(row["hand_object_p95_penetration_m"]) - median_value),
            int(row["episode"]),
        ),
    )
    v4_reset = {
        int(row["reset_index"])
        for row in inventory
        if row["clip"] == "hocap_170105"
        and row["mode"] == "v4"
        and not bool(row["absolute_geometry_pass"])
    }
    paired = next(row for row in v3_105 if int(row["reset_index"]) in v4_reset)
    common_650 = next(
        row
        for row in inventory
        if row["clip"] == "hocap_170650" and row["mode"] == "v3" and int(row["episode"]) == 16
    )
    choices = (
        ("C2_170105_MEDIAN_FAILURE", median),
        ("C2_170105_MAX_FAILURE", maximum),
        ("C2_170105_V3V4_PAIRED_RESET", paired),
        ("PRIMARY_COMMON_MODE_FAILURE_CASE_170650", common_650),
    )
    selected: list[dict[str, object]] = []
    for case_id, row in choices:
        selected.append(
            {
                "case_id": case_id,
                "selection_reason": {
                    "C2_170105_MEDIAN_FAILURE": (
                        "V3 failing episode nearest lower deterministic median p95"
                    ),
                    "C2_170105_MAX_FAILURE": "V3 maximum hand-object proxy penetration",
                    "C2_170105_V3V4_PAIRED_RESET": "V3 failure whose reset index also fails in V4",
                    "PRIMARY_COMMON_MODE_FAILURE_CASE_170650": (
                        "V3/V4 episode 16 shared reset and identical geometry failure"
                    ),
                }[case_id],
                "diagnostic_checkpoint_mode": row["mode"],
                **row,
            }
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"P3B5_OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=False)
    gates = read_json(GATES)
    input_receipts = {
        "curriculum_contract": receipt(CURRICULUM),
        "frozen_gates": receipt(GATES),
        "runtime_geometry_manifest": receipt(MANIFEST),
        "safe_banks": {
            clip: receipt(SAFE_BANK_ROOT / f"safe_bank_{clip.removeprefix('hocap_')}.npz")
            for clip in CLIPS
        },
    }
    qualifications: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    inventory: list[dict[str, object]] = []
    for mode in MODES:
        for clip in CLIPS:
            qualification_path = PILOT_ROOT / mode / clip / "c2/dev/qualification.json"
            qualification = read_json(qualification_path)
            qualifications[mode][clip]["c2"] = qualification
            input_receipts[f"c2_qualification_{mode}_{clip}"] = receipt(qualification_path)
            input_receipts[f"c2_checkpoint_{mode}_{clip}"] = dict(qualification["checkpoint"])
            inventory.extend(
                episode_inventory(
                    mode=mode,
                    clip=clip,
                    qualification=qualification,
                    safe_classes=safe_bank_classes(clip),
                    p95_limit_m=gate_for_clip(gates, clip).p95_penetration_inclusive_m,
                )
            )
    selected = representative_cases(inventory)
    physics_contract = {
        "schema_version": "Stage16P3B5PhysicsFreezeV1",
        "C0_C1_C2_authority": input_receipts["curriculum_contract"],
        "historical_expected_stages": {
            "C0": {"gravity_scale": 0.0, "friction_scale": 2.0},
            "C1": {"gravity_scale": 0.25, "friction_scale": 1.75},
            "C2": {"gravity_scale": 0.5, "friction_scale": 1.5},
        },
        "counterfactual_only_changes": ["gravity", "hand_friction", "object_friction"],
    }
    geometry_contract = {
        "schema_version": "Stage16P3B5GeometryFreezeV1",
        "gates": {
            clip: decision_contract(gate_for_clip(gates, clip))["geometry_gate"] for clip in CLIPS
        },
        "runtime_manifest": input_receipts["runtime_geometry_manifest"],
        "threshold_mutation": False,
    }
    decision = decision_contract(gate_for_clip(gates, "hocap_170105"))
    decision["clip_specific_geometry_gates"] = geometry_contract["gates"]
    decision["decision_written_before_counterfactual_results"] = True
    historical = historical_rows()
    c2_failure_counts = {
        f"{mode}/{clip}": sum(
            not bool(row["absolute_geometry_pass"])
            for row in inventory
            if row["mode"] == mode and row["clip"] == clip
        )
        for mode in MODES
        for clip in CLIPS
    }
    write_json(output / "frozen_inputs.json", input_receipts)
    write_json(output / "physics_contract.json", physics_contract)
    write_json(output / "geometry_contract.json", geometry_contract)
    write_json(output / "checkpoint_provenance.json", input_receipts)
    write_json(
        output / "c2_failure_inventory.json",
        {
            "schema_version": "Stage16P3B5C2FailureInventoryV1",
            "episode_count": len(inventory),
            "failure_counts": c2_failure_counts,
            "episodes": inventory,
        },
    )
    write_json(output / "decision_contract.json", decision)
    write_json(
        output / "selected_cases.json",
        {"schema_version": "Stage16P3B5SelectedCasesV1", "cases": selected},
    )
    write_json(output / "historical_c0_c1_c2.json", {"rows": historical})
    write_csv(output / "tables/failure_inventory.csv", inventory)
    write_csv(output / "tables/historical_c0_c1_c2.csv", historical)
    print(
        json.dumps(
            {"status": "P3B5_INPUTS_FROZEN", "output": str(output), "selected_cases": len(selected)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
