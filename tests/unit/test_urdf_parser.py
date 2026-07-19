from pathlib import Path

import pytest

from toporetarget.robots.urdf.parser import UrdfParseError, parse_urdf

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_robot" / "synthetic_hand.urdf"


def test_parser_supports_tree_joint_types_geometry_and_normalizes_axis() -> None:
    model = parse_urdf(FIXTURE)
    assert model.root_link == "palm"
    assert model.link_names == ("palm", "finger1", "finger2", "slider", "synthetic_tip")
    assert len(model.joints) == 4
    assert len(model.actuated_joints) == 3
    assert len(model.fixed_joints) == 1
    assert model.joint_by_name["joint_rev_z"].axis.tolist() == [0.0, 0.0, 1.0]
    assert model.links["palm"].visuals[0].geometry_type == "box"
    assert model.links["finger1"].collisions[0].geometry_type == "sphere"
    assert model.links["finger2"].visuals[0].geometry_type == "cylinder"
    assert model.links["synthetic_tip"].collisions == ()


def test_parser_rejects_mimic_and_bad_graph(tmp_path: Path) -> None:
    mimic = tmp_path / "mimic.urdf"
    mimic.write_text(
        (
            '<robot name="bad"><link name="a"/><link name="b"/>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="b"/>'
            '<mimic joint="other"/></joint></robot>'
        ),
        encoding="utf-8",
    )
    with pytest.raises(UrdfParseError, match="mimic"):
        parse_urdf(mimic)
