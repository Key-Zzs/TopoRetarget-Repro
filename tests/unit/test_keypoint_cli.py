from typer.testing import CliRunner

from toporetarget.cli.main import app


def test_keypoint_cli_lists_layouts_profiles_and_help() -> None:
    runner = CliRunner()
    layouts = runner.invoke(app, ["keypoints", "layouts"])
    profiles = runner.invoke(app, ["keypoints", "profiles"])
    help_result = runner.invoke(app, ["keypoints", "--help"])
    assert layouts.exit_code == 0
    assert "mediapipe21" in layouts.stdout
    assert profiles.exit_code == 0
    assert "mano_v1_2_smplx_to_mediapipe21" in profiles.stdout
    assert help_result.exit_code == 0


def test_grab_visualize_accepts_reference_compatibility_alias() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["data", "visualize", "--reference", "scene"])
    output = result.output + getattr(result, "stderr", "")
    assert "No such option" not in output
    assert "canonical mode requires --canonical" in output
