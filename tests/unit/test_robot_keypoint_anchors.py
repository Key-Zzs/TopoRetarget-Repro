from pathlib import Path

import numpy as np

from toporetarget.keypoints.registry import get_layout
from toporetarget.robots.anchors import AnchorDefinition, AnchorProfile
from toporetarget.robots.base import RobotHandModel
from toporetarget.robots.spec import RobotHandSpec
from toporetarget.robots.urdf.parser import parse_urdf


def synthetic_model() -> RobotHandModel:
    fixture = Path(__file__).parents[1] / "fixtures" / "synthetic_hand.urdf"
    spec = RobotHandSpec.from_mapping(
        {
            "name": "synthetic",
            "side": "right",
            "asset_id": "fixture",
            "urdf_relative_path": "synthetic_hand.urdf",
            "base_link": "palm",
            "semantic_keypoint_layout": "mediapipe21",
            "keypoint_anchor_profile": "synthetic",
            "dof_order": ["slide", "bend"],
            "neutral_q": [0, 0],
            "expected_link_count": 4,
            "expected_total_joint_count": 3,
            "expected_actuated_joint_count": 2,
            "expected_fixed_joint_count": 1,
        }
    )
    layout = get_layout("mediapipe21")
    anchors = []
    for index, name in enumerate(layout.semantic_names):
        if index == 0:
            anchors.append(AnchorDefinition(name, "link_origin", link_name="palm"))
        elif index == 1:
            anchors.append(AnchorDefinition(name, "joint_origin", joint_name="bend"))
        elif index == 2:
            anchors.append(AnchorDefinition(name, "joint_origin", joint_name="slide"))
        elif index == 4:
            anchors.append(AnchorDefinition(name, "joint_origin", joint_name="tip_fixed"))
        else:
            anchors.append(
                AnchorDefinition(
                    name, "link_local_point", link_name="link2", local_xyz=(0.0, index * 0.003, 0.0)
                )
            )
    model = RobotHandModel(spec, parse_urdf(fixture))
    model._anchor_profile = AnchorProfile(
        "synthetic", "1.0.0", "mediapipe21", tuple(anchors)
    ).validate()
    return model


def test_anchor_order_shape_and_joint_origin_semantics() -> None:
    model = synthetic_model()
    points = model.keypoints_base(np.array([0.01, 0.2], dtype=np.float64)).detach().cpu().numpy()
    assert points.shape == (21, 3)
    assert np.isfinite(points).all()
    assert np.allclose(points[1], [0.1, 0.0, 0.0])
    assert np.linalg.norm(points[4] - points[3]) > 0


def test_jacobian_shape_and_finite_difference_gate() -> None:
    from toporetarget.robots.reports import jacobian_check

    model = synthetic_model()
    result = jacobian_check(model, np.array([0.01, 0.2]), epsilon=1e-6)
    assert result["autograd_shape"] == [21, 3, 2]
    assert result["passed"]
    jacobian = model.keypoint_jacobian_qpos(np.array([0.01, 0.2]))
    assert jacobian.shape == (21, 3, 2)
    assert np.allclose(jacobian[0].detach().numpy(), 0.0)
    assert np.allclose(jacobian[1, :, 1].detach().numpy(), 0.0, atol=1e-12)
    assert float(jacobian[4, :, 1].detach().abs().max()) > 0
