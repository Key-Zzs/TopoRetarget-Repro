#!/usr/bin/env python3
"""Resolve and freeze primary-object authority for a held-out HOCap manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.hocap import HOCapAdapterV1  # noqa: E402
from toporetarget.adapters.datasets.hocap_primary_object import (  # noqa: E402
    HOCapPrimaryObjectResolverProfileV1,
    HOCapPrimaryObjectResolverProfileV2,
    resolve_hocap_primary_object,
    resolve_hocap_primary_object_v2,
)
from toporetarget.data.adapters.base import FrameRange  # noqa: E402
from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    BatchContractError,
    assert_frozen_manifest,
    atomic_write_json,
    atomic_write_text,
    freeze_selection,
    scan_hocap_candidates,
    stable_hash,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mano-model-root", type=Path)
    parser.add_argument("--resolver-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--official-label-revision",
        default="2b24836d5e51ad39e56ed4db3fc0c166e755332e",
    )
    parser.add_argument(
        "--official-label-bundle-sha256",
        default="be3be5f9e5beacab45d75d996c96d3fc959a6d822e31e402214ca899342416e2",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.start_frame < 0 or (args.end_frame is not None and args.end_frame <= args.start_frame):
        raise BatchContractError("PRIMARY_OBJECT_FRAME_RANGE_INVALID")
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    assert_frozen_manifest(base)
    adapter = HOCapAdapterV1(
        data_root=args.data_root.parent,
        mano_model_root=args.mano_model_root,
    )
    profile = (
        HOCapPrimaryObjectResolverProfileV1()
        if args.resolver_version == "v1"
        else HOCapPrimaryObjectResolverProfileV2()
    )
    mappings = []
    for clip in base["clips"]:
        sequence = str(clip["sequence"])
        end_frame = (
            args.end_frame
            if args.end_frame is not None
            else 41
            if args.resolver_version == "v1"
            else int(clip["raw_frames"])
        )
        if args.resolver_version == "v2" and (
            args.start_frame != 0 or end_frame != int(clip["raw_frames"])
        ):
            raise BatchContractError("PRIMARY_OBJECT_V2_REQUIRES_FULL_RAW_SEQUENCE")
        canonical = adapter.load_sequence(
            sequence,
            frame_range=FrameRange(args.start_frame, end_frame),
            primary_object_id=None,
        )
        result = (
            resolve_hocap_primary_object(canonical, profile=profile)
            if args.resolver_version == "v1"
            else resolve_hocap_primary_object_v2(canonical, profile=profile)
        )
        if args.resolver_version == "v1":
            result["selected_frame_range"] = [args.start_frame, end_frame]
        row = {
            **result,
            "clip_id": str(clip["clip_id"]),
            "sequence": sequence,
            "available_object_ids": [str(value) for value in clip["object_ids"]],
            "raw_hashes": dict(clip["raw_hashes"]),
            "resolution_sha256": stable_hash(result),
        }
        mappings.append(row)
    unresolved = [row["clip_id"] for row in mappings if row["status"] != "RESOLVED"]
    authority_core = {
        "schema_version": (
            "HOCapPrimaryObjectAuthorityV1"
            if args.resolver_version == "v1"
            else "HOCapPrimaryObjectAuthorityV2"
        ),
        "dataset": "hocap",
        "scope": "preselected_held_out_clips_only",
        "base_selection_manifest": str(args.base_manifest.resolve()),
        "base_selection_manifest_sha256": base["manifest_sha256"],
        "resolver_profile": profile.as_dict(),
        "official_source_verification": {
            "repository": "JWRoboticsVision/HO-Cap-Dataset",
            "revision": args.official_label_revision,
            "sequence_labels_tar_sha256": args.official_label_bundle_sha256,
            "verified_modalities": ["meta.yaml", "poses_m.npy", "poses_o.npy"],
            "verification_result": "LOCAL_RAW_HASHES_MATCH_OFFICIAL_LABEL_BUNDLE",
            "official_primary_object_field_available": False,
        },
        "outcome_inputs_used": False,
        "mappings": mappings,
    }
    authority = {
        **authority_core,
        "authority_sha256": stable_hash(authority_core),
    }
    selection_root = args.output_root / "selection"
    authority_name = f"primary_object_authority.{args.resolver_version}.json"
    atomic_write_json(selection_root / authority_name, authority)
    if unresolved:
        atomic_write_json(
            selection_root / "primary_object_resolution_blocked.json",
            {
                "status": "BLOCKED",
                "reason": "PRIMARY_OBJECT_UNRESOLVED",
                "unresolved_clip_ids": unresolved,
                "authority_sha256": authority["authority_sha256"],
            },
        )
        print(json.dumps(authority, indent=2, sort_keys=True))
        return 2

    candidates = scan_hocap_candidates(args.data_root)
    corrected = freeze_selection(
        candidates=candidates,
        root=args.output_root,
        seed=int(base["selection_seed"]),
        primary_object_authority=authority,
    )
    old_ids = [str(row["clip_id"]) for row in base["clips"]]
    new_ids = [str(row["clip_id"]) for row in corrected["clips"]]
    if old_ids != new_ids:
        raise BatchContractError(f"HELD_OUT_SELECTION_DRIFT:{old_ids}:{new_ids}")
    if args.resolver_version == "v2":
        ranges = {row["clip_id"]: row["selected_frame_range"] for row in mappings}
        corrected_core = dict(corrected)
        corrected_core.pop("manifest_sha256")
        corrected_core["interaction_segment_contract"] = profile.profile_id
        for clip in corrected_core["clips"]:
            clip["selected_frame_range"] = ranges[clip["clip_id"]]
        corrected = {
            **corrected_core,
            "manifest_sha256": stable_hash(corrected_core),
        }
        atomic_write_json(selection_root / "held_out_5_manifest.json", corrected)
        atomic_write_text(
            selection_root / "held_out_5_manifest.yaml",
            yaml.safe_dump(corrected, sort_keys=True),
        )
        selection_receipt = json.loads(
            (selection_root / "selection_receipt.json").read_text(encoding="utf-8")
        )
        selection_receipt["manifest_sha256"] = corrected["manifest_sha256"]
        selection_receipt["interaction_segment_contract"] = profile.profile_id
        atomic_write_json(selection_root / "selection_receipt.json", selection_receipt)
    atomic_write_json(
        selection_root / "primary_object_resolution_receipt.json",
        {
            "status": "PASS",
            "authority": str((selection_root / authority_name).resolve()),
            "authority_sha256": authority["authority_sha256"],
            "corrected_manifest": str((selection_root / "held_out_5_manifest.json").resolve()),
            "corrected_manifest_sha256": corrected["manifest_sha256"],
            "supersedes_manifest_sha256": base["manifest_sha256"],
            "mapping": {row["clip_id"]: row["primary_object_id"] for row in mappings},
            "outcome_inputs_used": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "mapping": {row["clip_id"]: row["primary_object_id"] for row in mappings},
                "authority_sha256": authority["authority_sha256"],
                "manifest_sha256": corrected["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
