from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from toporetarget.utils.hashing import sha256_file

REQUIRED_IDS = (
    [f"equation_{index}" for index in range(1, 13)]
    + [f"table_{index}" for index in range(1, 7)]
    + [f"figure_{index}" for index in range(1, 6)]
    + [
        "section_3_1",
        "section_3_2",
        "section_3_3",
        "section_3_4",
        "section_4_action",
        "section_4_observation",
        "section_4_reference_initialization",
        "section_4_reward",
        "section_4_domain_randomization",
        "appendix_a_1",
        "appendix_a_2",
        "appendix_a_3",
        "appendix_a_4",
        "appendix_a_5",
        "limitation_virtual_contact",
        "dataset_contactpose",
        "dataset_hocap",
        "dataset_pen_spin",
        "baseline_omniretarget",
        "baseline_mink",
        "baseline_dexpilot",
        "baseline_geort",
        "fixed_parameter_claim",
        "zero_shot_wuji_claim",
        "object_scale_augmentation",
        "hand_embodiment_augmentation",
    ]
)
REQUIRED_ITEM_FIELDS = {
    "title",
    "kind",
    "paper_section",
    "pdf_page",
    "implementation_targets",
    "test_targets",
    "status",
}
NOT_PROVIDED_PATTERN = re.compile(r"^A_[A-Z0-9_]+$")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _assumption_ids(path: Path) -> set[str]:
    return set(re.findall(r"\bA_[A-Z0-9_]+\b", path.read_text(encoding="utf-8")))


def _flatten_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        result: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_flatten_values(child, child_prefix))
        return result
    return [(prefix, value)]


def validate_paper_fidelity(repo_root: Path) -> list[str]:
    """Return human-readable validation errors; an empty list means valid."""

    root = repo_root.resolve()
    manifest_path = root / "docs" / "PAPER_FIDELITY.yaml"
    assumptions_path = root / "docs" / "ASSUMPTIONS.md"
    errors: list[str] = []
    try:
        manifest = _load_yaml(manifest_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"cannot load manifest: {exc}"]
    paper = manifest.get("paper")
    if not isinstance(paper, dict):
        errors.append("manifest.paper must be a mapping")
        paper = {}
    items = manifest.get("items")
    if not isinstance(items, list):
        return errors + ["manifest.items must be a list"]
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            errors.append("every manifest item must be a mapping with a string id")
            continue
        item_id = item["id"]
        by_id[item_id] = item
        missing = sorted(REQUIRED_ITEM_FIELDS - set(item))
        if missing:
            errors.append(f"{item_id}: missing fields: {', '.join(missing)}")
        if item.get("status") not in {
            "specified_not_implemented",
            "implemented",
            "implemented_with_assumptions",
            "blocked_missing_information",
            "blocked_missing_asset",
            "not_applicable",
        }:
            errors.append(f"{item_id}: invalid status {item.get('status')!r}")
    for required in REQUIRED_IDS:
        if required not in by_id:
            errors.append(f"missing required manifest item: {required}")

    if not assumptions_path.is_file():
        errors.append("docs/ASSUMPTIONS.md is missing")
        registered_assumptions: set[str] = set()
    else:
        registered_assumptions = _assumption_ids(assumptions_path)
    for item_id, item in by_id.items():
        assumptions = item.get("assumptions", [])
        if isinstance(assumptions, list):
            for assumption in assumptions:
                if isinstance(assumption, str) and assumption not in registered_assumptions:
                    errors.append(f"{item_id}: assumption {assumption} is not registered")

    pdf_path_value = paper.get("pdf_path")
    pdf_path = root / str(pdf_path_value) if isinstance(pdf_path_value, str) else None
    if pdf_path is None or not pdf_path.is_file():
        errors.append(f"paper PDF is missing: {pdf_path_value}")
    else:
        actual_hash = sha256_file(pdf_path)
        if paper.get("pdf_sha256") != actual_hash:
            errors.append(
                "paper PDF SHA-256 mismatch: "
                f"expected {paper.get('pdf_sha256')}, actual {actual_hash}"
            )
        if paper.get("page_count") != 16:
            errors.append(f"paper page_count must be 16, got {paper.get('page_count')!r}")

    contract = manifest.get("parameter_contract", {})
    if not isinstance(contract, dict):
        errors.append("parameter_contract must be a mapping")
        contract = {}
    for config_rel, expected in contract.items():
        if not isinstance(config_rel, str) or not isinstance(expected, dict):
            errors.append("parameter_contract entries must be mappings")
            continue
        config_path = root / config_rel
        try:
            config = _load_yaml(config_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"parameter contract cannot load {config_rel}: {exc}")
            continue
        for dotted_path, requirement in expected.items():
            if not isinstance(requirement, dict):
                errors.append(f"{config_rel}:{dotted_path}: invalid contract entry")
                continue
            current: Any = config
            for component in str(dotted_path).split("."):
                if not isinstance(current, dict) or component not in current:
                    current = object()
                    break
                current = current[component]
            source = requirement.get("source", "paper")
            if source == "not_provided":
                if current is not None:
                    errors.append(f"{config_rel}:{dotted_path}: not_provided value must be null")
            elif current != requirement.get("value"):
                errors.append(
                    f"{config_rel}:{dotted_path}: expected {requirement.get('value')!r}, "
                    f"got {current!r}"
                )
    for config_rel in manifest.get("not_provided_configs", []):
        config_path = root / str(config_rel)
        if not config_path.is_file():
            errors.append(f"not_provided config is missing: {config_rel}")

    return errors
