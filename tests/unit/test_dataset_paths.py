from pathlib import Path

import yaml

from toporetarget.paths.datasets import DatasetPathResolver


def make_registry(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {"datasets": {"grab": {"aliases": ["GRAB", "grab"]}, "oakink": {"aliases": ["OakInk"]}}}
        ),
        encoding="utf-8",
    )


def test_nested_candidates_and_unregistered_directory(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    make_registry(registry)
    (tmp_path / "backups" / "data" / "not_a_dataset").mkdir(parents=True)
    nested = tmp_path / "GRAB" / "data" / "release" / "subject" / "sequence"
    nested.mkdir(parents=True)
    result = DatasetPathResolver(tmp_path, registry, max_depth=4).discover()
    grab = next(item for item in result if item.canonical_dataset_name == "grab")
    assert grab.status == "found"
    assert str(nested) in grab.candidate_directories
    assert all(
        "backups" not in item.candidate_directories[0]
        for item in result
        if item.candidate_directories
    )


def test_alias_ambiguity_and_missing_data(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    make_registry(registry)
    (tmp_path / "GRAB" / "data").mkdir(parents=True)
    (tmp_path / "grab" / "data").mkdir(parents=True)
    (tmp_path / "OakInk").mkdir()
    results = DatasetPathResolver(tmp_path, registry).discover()
    grab = next(item for item in results if item.canonical_dataset_name == "grab")
    oakink = next(item for item in results if item.canonical_dataset_name == "oakink")
    assert grab.status == "ambiguity"
    assert grab.matched_aliases == ["GRAB", "grab"]
    assert oakink.status == "missing_data_directory"


def test_symlinks_are_not_followed_and_max_depth_is_bounded(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    make_registry(registry)
    data = tmp_path / "GRAB" / "data"
    data.mkdir(parents=True)
    target = tmp_path / "outside"
    (target / "deep").mkdir(parents=True)
    (data / "linked").symlink_to(target, target_is_directory=True)
    (data / "one" / "two" / "three").mkdir(parents=True)
    result = next(
        item
        for item in DatasetPathResolver(tmp_path, registry, max_depth=1).discover()
        if item.canonical_dataset_name == "grab"
    )
    assert str(data / "linked") not in result.candidate_directories
    assert str(data / "one" / "two") not in result.candidate_directories
