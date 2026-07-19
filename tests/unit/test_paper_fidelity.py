import shutil
from pathlib import Path

import yaml

from toporetarget.paper.fidelity import validate_paper_fidelity

REPO_ROOT = Path(__file__).resolve().parents[2]


def copy_audit_tree(tmp_path: Path) -> Path:
    for name in ("PAPER_FIDELITY.yaml", "ASSUMPTIONS.md"):
        source = REPO_ROOT / "docs" / name
        target = tmp_path / "docs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(REPO_ROOT / "docs" / "TopoRetarget.pdf", tmp_path / "docs" / "TopoRetarget.pdf")
    for name in ("retarget.yaml", "metrics.yaml", "rl.yaml"):
        target = tmp_path / "configs" / "paper" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "configs" / "paper" / name, target)
    return tmp_path


def load_manifest(root: Path) -> dict:
    return yaml.safe_load((root / "docs" / "PAPER_FIDELITY.yaml").read_text(encoding="utf-8"))


def save_manifest(root: Path, value: dict) -> None:
    (root / "docs" / "PAPER_FIDELITY.yaml").write_text(
        yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
    )


def test_normal_manifest_passes() -> None:
    assert validate_paper_fidelity(REPO_ROOT) == []


def test_missing_equation_and_required_field_fail(tmp_path: Path) -> None:
    root = copy_audit_tree(tmp_path)
    manifest = load_manifest(root)
    manifest["items"] = [item for item in manifest["items"] if item["id"] != "equation_12"]
    next(item for item in manifest["items"] if item["id"] == "equation_11").pop("title")
    save_manifest(root, manifest)
    errors = validate_paper_fidelity(root)
    assert any("equation_12" in error for error in errors)
    assert any("equation_11" in error for error in errors)


def test_hash_assumption_and_not_provided_fail(tmp_path: Path) -> None:
    root = copy_audit_tree(tmp_path)
    manifest = load_manifest(root)
    manifest["paper"]["pdf_sha256"] = "bad"
    next(item for item in manifest["items"] if item["id"] == "equation_1")["assumptions"].append(
        "A_NOT_REGISTERED_001"
    )
    manifest["parameter_contract"]["configs/paper/retarget.yaml"]["optimizer"] = {
        "value": "guess",
        "source": "paper",
    }
    save_manifest(root, manifest)
    errors = validate_paper_fidelity(root)
    assert any("SHA-256 mismatch" in error for error in errors)
    assert any("not registered" in error for error in errors)
    assert any("optimizer" in error for error in errors)


def test_parameter_contract_mismatch_fails(tmp_path: Path) -> None:
    root = copy_audit_tree(tmp_path)
    manifest = load_manifest(root)
    manifest["parameter_contract"]["configs/paper/retarget.yaml"]["lambda_interaction_mesh"][
        "value"
    ] = 501
    save_manifest(root, manifest)
    assert any("lambda_interaction_mesh" in error for error in validate_paper_fidelity(root))
