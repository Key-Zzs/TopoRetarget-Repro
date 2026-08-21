#!/usr/bin/env python3
"""Offline hard gate for Stage16 SourceProfileTrackingV1.

The evaluator compares only each actual trajectory against that clip's own
already-materialized HumanObjectCouplingContactProfileV1 target.  It never
reruns PhysX, retimes a trace, or substitutes the positive clip's source for
the negative clip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.reference_tracking.source_profile_tracking import (  # noqa: E402
    SourceProfileTrackingTargetsV1,
    Stage16SourceProfileTrackingV1,
    pose_derived_coupling_ratios,
    source_profile_tracking_terms,
)

NEGATIVE_ROOT = (
    REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/"
    "smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4"
)
POSITIVE_ROOT = (
    REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650"
)
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
FINGERTIP_LINKS = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)
PHASES = (
    ("APPROACH", 0, 129),
    ("CONTACT", 129, 160),
    ("GRASP", 160, 184),
    ("LIFT", 184, 225),
    ("LATE_MOTION", 225, 321),
    ("CONTACT_TO_EARLY_LIFT", 129, 225),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"SOURCE_PROFILE_OFFLINE_CSV_EMPTY:{path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("SOURCE_PROFILE_OFFLINE_CSV_FIELD_DRIFT")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--negative-root", type=Path, default=NEGATIVE_ROOT)
    parser.add_argument("--positive-root", type=Path, default=POSITIVE_ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    return parser


def _trace_paths(root: Path, *, count: int, digits: int) -> list[Path]:
    paths = [root / f"episode_{index:0{digits}d}.npz" for index in range(count)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"SOURCE_PROFILE_OFFLINE_TRACES_MISSING:{root}")
    return paths


def _tip_indices(reference_path: Path) -> list[int]:
    with np.load(reference_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    names = tuple(str(value) for value in metadata["tracked_link_names"])
    try:
        return [names.index(link) for link in FINGERTIP_LINKS]
    except ValueError as error:
        raise ValueError("SOURCE_PROFILE_OFFLINE_FINGERTIP_LINK_MAPPING_INVALID") from error


def _force(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("tip_pair_force_world", "fingertip_object_pair_force_world"):
        if key in archive.files:
            value = np.asarray(archive[key], dtype=np.float32)
            if value.shape == (321, 5, 3):
                return value
    raise ValueError("SOURCE_PROFILE_OFFLINE_TIP_PAIR_FORCE_MISSING")


def _episode_terms(
    path: Path,
    *,
    clip_index: int,
    tip_indices: list[int],
    targets: SourceProfileTrackingTargetsV1,
    contact_force_scale_n: float,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("object_pose", "wrist_pose", "tracked_link_positions", "reference_index")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"SOURCE_PROFILE_OFFLINE_TRACE_FIELDS_MISSING:{path}:{missing}")
        object_pose = np.asarray(archive["object_pose"], dtype=np.float32)
        wrist_pose = np.asarray(archive["wrist_pose"], dtype=np.float32)
        links = np.asarray(archive["tracked_link_positions"], dtype=np.float32)
        reference_index = np.asarray(archive["reference_index"], dtype=np.int64)
        force = _force(archive)
    if (
        object_pose.shape != (321, 7)
        or wrist_pose.shape != (321, 7)
        or links.shape != (321, 16, 3)
        or not np.array_equal(reference_index, np.arange(321))
    ):
        raise ValueError(f"SOURCE_PROFILE_OFFLINE_TRACE_CONTRACT_INVALID:{path}")
    device = torch.device("cpu")
    clip = torch.full((321,), clip_index, dtype=torch.long, device=device)
    frame = torch.arange(321, dtype=torch.long, device=device)
    source = targets.gather(clip, frame)
    current_wrist = torch.as_tensor(wrist_pose, device=device)
    current_object = torch.as_tensor(object_pose, device=device)
    previous_wrist = torch.cat((current_wrist[:1], current_wrist[:-1]), dim=0)
    previous_object = torch.cat((current_object[:1], current_object[:-1]), dim=0)
    linear, angular = pose_derived_coupling_ratios(
        previous_wrist_pose_wxyz=previous_wrist,
        current_wrist_pose_wxyz=current_wrist,
        previous_object_pose_wxyz=previous_object,
        current_object_pose_wxyz=current_object,
        dt_s=0.05,
    )
    terms = source_profile_tracking_terms(
        **source,
        robot_tip_positions_world=torch.as_tensor(links[:, tip_indices], device=device),
        robot_tip_pair_force_world=torch.as_tensor(force, device=device),
        object_pose_wxyz=current_object,
        robot_linear_coupling_normalized=linear / targets.linear_coupling_scale,
        robot_angular_coupling_normalized=angular / targets.angular_coupling_scale,
        contact_force_scale_n=contact_force_scale_n,
    )
    return {name: value.detach().cpu().numpy() for name, value in terms.items()}


def _summary(values: np.ndarray) -> dict[str, float]:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("SOURCE_PROFILE_OFFLINE_NONFINITE_SUMMARY")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _rows_for_clip(
    *,
    clip: str,
    path_list: list[Path],
    tip_indices: list[int],
    targets: SourceProfileTrackingTargetsV1,
    contact_force_scale_n: float,
) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    values_by_phase: dict[str, list[np.ndarray]] = {name: [] for name, *_ in PHASES}
    values_by_phase["ALL"] = []
    components = (
        "l_profile",
        "l_profile_contact",
        "l_profile_geometry",
        "l_profile_linear_coupling",
        "l_profile_angular_coupling",
    )
    clip_index = 0 if clip == "hocap_170105" else 1
    trace_receipts: list[dict[str, object]] = []
    for episode, path in enumerate(path_list):
        terms = _episode_terms(
            path,
            clip_index=clip_index,
            tip_indices=tip_indices,
            targets=targets,
            contact_force_scale_n=contact_force_scale_n,
        )
        trace_receipts.append({"episode": episode, "path": str(path), "sha256": _sha256(path)})
        for phase, start, stop in (*PHASES, ("ALL", 0, 321)):
            selection = slice(start, stop)
            aggregate = np.stack([terms[name][selection].mean() for name in components])
            if not np.isfinite(aggregate).all():
                raise ValueError("SOURCE_PROFILE_OFFLINE_NONFINITE_EPISODE")
            values_by_phase[phase].append(aggregate)
            rows.append(
                {
                    "clip": clip,
                    "episode": episode,
                    "phase": phase,
                    "start_frame": start,
                    "stop_frame_exclusive": stop,
                    **{name: float(aggregate[index]) for index, name in enumerate(components)},
                    "r_profile_mean": float(terms["r_profile"][selection].mean()),
                }
            )
    return rows, values_by_phase, trace_receipts


def _pairwise_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    comparisons = negative[:, None] - positive[None, :]
    return float((comparisons > 0.0).mean() + 0.5 * (comparisons == 0.0).mean())


def _markdown(result: dict[str, Any], comparison: list[dict[str, object]]) -> str:
    window = next(row for row in comparison if row["phase"] == "CONTACT_TO_EARLY_LIFT")
    return "\n".join(
        (
            "# SourceProfileTrackingV1 Offline Objective Validation",
            "",
            f"`OFFLINE_OBJECTIVE_VALIDATION={result['OFFLINE_OBJECTIVE_VALIDATION']}`",
            f"`classification={result['classification']}`",
            "",
            "The target uses identity reference-index phase alignment only; no DTW, learned time "
            "warp, or outcome-dependent phase offset was used.",
            "",
            "## CONTACT to early-LIFT",
            "",
            "| Metric | 170105 failed median | 170650 accepted median | Direction |",
            "| --- | ---: | ---: | --- |",
            *[
                f"| {name} | {window[f'170105_{name}_median']:.6f} | "
                f"{window[f'170650_{name}_median']:.6f} | "
                f"{window[f'{name}_direction']} |"
                for name in (
                    "l_profile",
                    "l_profile_contact",
                    "l_profile_geometry",
                    "l_profile_linear_coupling",
                    "l_profile_angular_coupling",
                )
            ],
            "",
            "The validation is a pre-training hard gate.  It does not make a functional-grasp "
            "claim from the source profile or promote topology/slip channels lacking actual PhysX "
            "contact-point authority.",
            "",
        )
    )


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    targets_path = args.targets.resolve()
    contract_path = args.target_contract.resolve()
    if not targets_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("SOURCE_PROFILE_OFFLINE_TARGET_OR_CONTRACT_MISSING")
    contact_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = Stage16SourceProfileTrackingV1()
    canonical_contract = json.loads(json.dumps(contract.as_dict(), sort_keys=True))
    if contact_contract.get("contract") != canonical_contract:
        raise ValueError("SOURCE_PROFILE_OFFLINE_CONTRACT_DRIFT")
    targets = SourceProfileTrackingTargetsV1.from_npz(targets_path, device="cpu")
    # The V4 strict-contact receipt freezes the same global scalar used for the
    # auxiliary profile's bounded force activation.  This is a scale, not a
    # per-object profile weight.
    strict_receipt = (
        REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json"
    )
    strict = json.loads(strict_receipt.read_text(encoding="utf-8"))
    force_scale = float(strict["frozen_parameters"]["lambda_tip_n"])
    if force_scale <= 0.0:
        raise ValueError("SOURCE_PROFILE_OFFLINE_V4_FORCE_SCALE_INVALID")
    negative_paths = _trace_paths(args.negative_root.resolve(), count=10, digits=2)
    positive_paths = _trace_paths(args.positive_root.resolve(), count=20, digits=3)
    tip_indices = _tip_indices(
        args.reference_root.resolve() / "hocap_170105.reference_kinematics_v2.npz"
    )
    negative_rows, negative_values, negative_receipts = _rows_for_clip(
        clip="hocap_170105",
        path_list=negative_paths,
        tip_indices=tip_indices,
        targets=targets,
        contact_force_scale_n=force_scale,
    )
    positive_rows, positive_values, positive_receipts = _rows_for_clip(
        clip="hocap_170650",
        path_list=positive_paths,
        tip_indices=tip_indices,
        targets=targets,
        contact_force_scale_n=force_scale,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "positive_170650.csv", positive_rows)
    _write_csv(output / "negative_170105.csv", negative_rows)
    comparison: list[dict[str, object]] = []
    components = (
        "l_profile",
        "l_profile_contact",
        "l_profile_geometry",
        "l_profile_linear_coupling",
        "l_profile_angular_coupling",
    )
    for phase in tuple(name for name, *_ in PHASES) + ("ALL",):
        negative = np.stack(negative_values[phase])
        positive = np.stack(positive_values[phase])
        row: dict[str, object] = {
            "phase": phase,
            "episodes_170105": len(negative),
            "episodes_170650": len(positive),
        }
        for index, name in enumerate(components):
            neg = negative[:, index]
            pos = positive[:, index]
            row.update(
                {
                    f"170105_{name}_median": float(np.median(neg)),
                    f"170650_{name}_median": float(np.median(pos)),
                    f"{name}_difference_negative_minus_positive": float(
                        np.median(neg) - np.median(pos)
                    ),
                    f"{name}_direction": "NEGATIVE_HIGHER"
                    if np.median(neg) > np.median(pos)
                    else "NOT_NEGATIVE_HIGHER",
                    f"{name}_pairwise_auc": _pairwise_auc(neg, pos),
                }
            )
        comparison.append(row)
    _write_csv(output / "component_comparison.csv", comparison)
    phase_rows = [
        {
            "phase": row["phase"],
            "l_profile_negative_minus_positive": row[
                "l_profile_difference_negative_minus_positive"
            ],
            "l_profile_pairwise_auc": row["l_profile_pairwise_auc"],
        }
        for row in comparison
    ]
    _write_csv(output / "phase_comparison.csv", phase_rows)
    window = next(row for row in comparison if row["phase"] == "CONTACT_TO_EARLY_LIFT")
    finite = all(
        np.isfinite(np.stack(values)).all()
        for phase_values in (negative_values, positive_values)
        for values in phase_values.values()
    )
    positive_pathology_bound = 4.0
    positive_p95 = float(
        np.quantile(np.stack(positive_values["CONTACT_TO_EARLY_LIFT"])[:, 0], 0.95)
    )
    positive_sensible = positive_p95 <= positive_pathology_bound
    negative_higher = bool(window["l_profile_difference_negative_minus_positive"] > 0.0)
    interpretable_component_higher = any(
        float(window[f"{name}_difference_negative_minus_positive"]) > 0.0 for name in components[1:]
    )
    criteria = {
        "A_all_profile_values_finite": finite,
        "B_positive_170650_contact_to_early_lift_p95_not_pathological": positive_sensible,
        "B_bound_predeclared_profile_loss": positive_pathology_bound,
        "B_positive_p95": positive_p95,
        "C_negative_170105_median_contact_to_early_lift_loss_higher": negative_higher,
        "D_at_least_one_interpretable_component_same_direction": interpretable_component_higher,
        "E_per_object_weights_changed": False,
    }
    if not finite:
        classification = "PROFILE_OBJECTIVE_NUMERICALLY_INVALID"
    elif not positive_sensible:
        classification = "PROFILE_OBJECTIVE_POSITIVE_CONTROL_REGRESSION"
    elif not negative_higher or not interpretable_component_higher:
        classification = "PROFILE_OBJECTIVE_NOT_DISCRIMINATIVE"
    else:
        classification = "PROFILE_OBJECTIVE_VALIDATED"
    result = {
        "schema_version": "Stage16SourceProfileTrackingOfflineValidationV1",
        "classification": classification,
        "OFFLINE_OBJECTIVE_VALIDATION": "PASS"
        if classification == "PROFILE_OBJECTIVE_VALIDATED"
        else "FAIL",
        "criteria": criteria,
        "phase_alignment": contract.time_alignment,
        "dtw": "NO",
        "learned_time_warp": "NO",
        "outcome_dependent_phase_shift": "NO",
        "target": {"path": str(targets_path), "sha256": _sha256(targets_path)},
        "target_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
        "strict_v4_force_scale": {
            "path": str(strict_receipt),
            "sha256": _sha256(strict_receipt),
            "lambda_tip_n": force_scale,
        },
        "negative_170105_traces": negative_receipts,
        "positive_170650_traces": positive_receipts,
        "contact_to_early_lift": window,
        "positive_control_compared_to_own_source": True,
        "actual_topology_and_exact_slip": "DIAGNOSTIC_ONLY_NOT_PROFILE_REWARD_CHANNELS",
    }
    _write_json(output / "objective_validation.json", result)
    (output / "objective_validation.md").write_text(_markdown(result, comparison), encoding="utf-8")
    print(json.dumps({"classification": classification, "criteria": criteria}, sort_keys=True))
    return 0 if classification == "PROFILE_OBJECTIVE_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
