"""Support semantics remain source-backed and fail closed without a safe bank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toporetarget.physics.support_contract import (
    SourceSupportAssetV1,
    SupportClassification,
    SupportMode,
    discover_source_support_evidence,
)
from toporetarget.physics.support_feasibility import build_support_timeline, decide_support_mode


def test_support_discovery_does_not_promote_camera_metadata_to_scene_support(
    tmp_path: Path,
) -> None:
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    (sequence / "meta.yaml").write_text("realsense:\n  serial: abc\nobject_ids: [G10_2]\n")

    discovered = discover_source_support_evidence(sequence)

    assert discovered["metadata_support_hits"] == []
    assert discovered["source_scene_geometry_candidates"] == []
    assert discovered["network_download_performed"] is False


def test_generic_plane_cannot_be_relabelled_as_source_support() -> None:
    with pytest.raises(ValueError, match="PROVENANCE_NOT_EXPLICIT"):
        SourceSupportAssetV1(
            source_path="/tmp/floor.obj",
            mesh_sha256="a" * 64,
            pose_world=(0.0,) * 7,
            scale=(1.0, 1.0, 1.0),
            collision_approximation="triangle_mesh",
            friction_source="default",
        )


def test_contact_ready_only_mode_keeps_frame_zero_unauthorized() -> None:
    timeline = build_support_timeline(
        runtime_index=np.arange(5, dtype=np.int64),
        source_expected_contact=np.asarray([False, False, True, True, True]),
        gravity_label_by_state={
            0: "GRAVITY_RISK",
            1: "GRAVITY_RISK",
            2: "GRAVITY_SAFE",
            3: "GRAVITY_SAFE",
            4: "GRAVITY_SAFE",
        },
        source_support_available=False,
    )
    decision = decide_support_mode(
        support_timeline=timeline,
        safe_bank_names=["CONTACT_READY_SAFE", "PERSISTENT_SAFE"],
        hidden_support=False,
    )

    assert timeline[0]["classification"] == SupportClassification.UNSUPPORTED_REFERENCE.value
    assert decision["support_mode"] == SupportMode.CONTACT_READY_ONLY_VALIDATED.value
    assert decision["frame_zero_full_gravity_authorized"] is False
    assert decision["p3_allowed_reset_banks"] == ["CONTACT_READY_SAFE", "PERSISTENT_SAFE"]
