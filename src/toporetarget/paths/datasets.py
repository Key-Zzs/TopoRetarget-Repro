from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetDiscovery:
    canonical_dataset_name: str
    matched_alias: str | None
    matched_aliases: list[str]
    alias_root: str | None
    data_root: str | None
    candidate_directories: list[str]
    readable: bool
    status: str
    scan_truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetPathResolver:
    """Read-only, allowlisted discovery of dataset directory layouts."""

    def __init__(
        self,
        storage_root: Path,
        registry_path: Path,
        *,
        max_depth: int = 4,
        max_entries_per_directory: int = 8,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if max_entries_per_directory < 1:
            raise ValueError("max_entries_per_directory must be positive")
        self.storage_root = storage_root.expanduser()
        self.registry_path = registry_path
        self.max_depth = max_depth
        self.max_entries_per_directory = max_entries_per_directory

    def _registry(self) -> dict[str, list[str]]:
        with self.registry_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        datasets = loaded.get("datasets", {})
        if not isinstance(datasets, dict):
            raise ValueError("Dataset registry must contain a mapping at datasets")
        result: dict[str, list[str]] = {}
        for canonical, entry in datasets.items():
            aliases = entry.get("aliases", []) if isinstance(entry, dict) else []
            if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
                raise ValueError(f"Invalid aliases for dataset {canonical}")
            result[str(canonical)] = list(aliases)
        return result

    def _directories(self, data_root: Path) -> tuple[list[Path], bool]:
        candidates: list[Path] = []
        scan_truncated = False
        frontier = [(data_root, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            candidates.append(current)
            if depth >= self.max_depth:
                continue
            try:
                with os.scandir(current) as entries:
                    children: list[Path] = []
                    for index, entry in enumerate(entries):
                        if index >= self.max_entries_per_directory:
                            scan_truncated = True
                            break
                        if entry.is_dir(follow_symlinks=False):
                            children.append(Path(entry.path))
            except OSError:
                continue
            children.sort(key=lambda path: path.name)
            for child in children:
                frontier.append((child, depth + 1))
        return candidates, scan_truncated

    def discover(self) -> list[DatasetDiscovery]:
        results: list[DatasetDiscovery] = []
        for canonical, aliases in self._registry().items():
            existing = [
                (alias, self.storage_root / alias)
                for alias in aliases
                if (self.storage_root / alias).is_dir()
                and not (self.storage_root / alias).is_symlink()
            ]
            if not existing:
                results.append(
                    DatasetDiscovery(canonical, None, [], None, None, [], False, "missing")
                )
                continue

            candidate_dirs: list[Path] = []
            data_roots: list[Path] = []
            scan_truncated = False
            for _, alias_root in existing:
                data_root = alias_root / "data"
                if data_root.is_dir() and not data_root.is_symlink():
                    data_roots.append(data_root)
                    discovered, truncated = self._directories(data_root)
                    candidate_dirs.extend(discovered)
                    scan_truncated = scan_truncated or truncated
            aliases_found = [alias for alias, _ in existing]
            if not data_roots:
                status = "ambiguity" if len(existing) > 1 else "missing_data_directory"
                results.append(
                    DatasetDiscovery(
                        canonical,
                        aliases_found[0],
                        aliases_found,
                        str(existing[0][1]),
                        None,
                        [],
                        False,
                        status,
                        scan_truncated,
                    )
                )
                continue

            unique_candidates = sorted(set(candidate_dirs))
            readable = all(path.is_dir() and path.stat().st_mode & 0o444 for path in data_roots)
            status = "ambiguity" if len(existing) > 1 else "found"
            results.append(
                DatasetDiscovery(
                    canonical,
                    aliases_found[0],
                    aliases_found,
                    str(existing[0][1]),
                    str(data_roots[0]),
                    [str(path) for path in unique_candidates],
                    bool(readable),
                    status,
                    scan_truncated,
                )
            )
        return results


def discovery_report(
    results: list[DatasetDiscovery],
    *,
    storage_root: Path,
    max_depth: int,
    max_entries_per_directory: int = 8,
) -> dict[str, Any]:
    return {
        "storage_root": str(storage_root),
        "max_depth": max_depth,
        "max_entries_per_directory": max_entries_per_directory,
        "read_only": True,
        "datasets": [item.as_dict() for item in results],
    }
