from toporetarget.workflows.visualization import viewer_command


def test_manifest_viewer_command_is_artifact_only() -> None:
    manifest = {
        "repo_root": "/repo",
        "asset_root": "/assets/artimano",
        "robot": "artimano_rh",
        "selected_frame_range": [240, 300],
        "artifacts": {
            "canonical": {"path": "/run/canonical.zarr"},
            "warm_start": {"path": "/run/warm.zarr"},
            "graph": {"path": "/run/graph.zarr"},
            "final": {"path": "/run/final.zarr"},
            "collision_samples": {"path": "/run/surface.npz"},
        },
    }
    command = viewer_command(manifest, frame=29, output="/run/frame.png")
    assert command[command.index("visualize-refinement") + 1] == "--canonical"
    assert "--frame" in command and "29" in command
    assert "--asset-root" in command
    assert "refine" not in command


def test_manifest_viewer_command_preserves_local_interactive_range() -> None:
    manifest = {
        "repo_root": "/repo",
        "robot": "artimano_rh",
        "selected_frame_range": [984, 1044],
        "artifacts": {
            "canonical": {"path": "/run/canonical.zarr"},
            "warm_start": {"path": "/run/warm.zarr"},
            "graph": {"path": "/run/graph.zarr"},
            "final": {"path": "/run/final.zarr"},
            "collision_samples": {"path": "/run/surface.npz"},
        },
    }
    command = viewer_command(
        manifest,
        interactive=True,
        start_frame=2,
        end_frame=58,
        view="scene",
    )
    assert "--interactive" in command
    assert command[command.index("--start-frame") + 1] == "2"
    assert command[command.index("--end-frame") + 1] == "58"


def test_manifest_viewer_command_supports_object_frame_view() -> None:
    manifest = {
        "repo_root": "/repo",
        "robot": "artimano_rh",
        "selected_frame_range": [0, 60],
        "artifacts": {
            "canonical": {"path": "/run/canonical.zarr"},
            "warm_start": {"path": "/run/warm.zarr"},
            "graph": {"path": "/run/graph.zarr"},
            "final": {"path": "/run/final.zarr"},
            "collision_samples": {"path": "/run/surface.npz"},
        },
    }
    command = viewer_command(manifest, frame=0, view="object", output="/run/frame.png")
    assert command[command.index("--view") + 1] == "object"
