"""Validated semantic keypoint layout definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LayoutValidationError(ValueError):
    """Raised when a keypoint layout violates its topology contract."""


@dataclass(frozen=True)
class KeypointLayoutDefinition:
    name: str
    version: str
    semantic_names: tuple[str, ...]
    parents: tuple[int | None, ...]
    edges: tuple[tuple[int, int], ...]
    fingertip_indices: tuple[int, ...]
    wrist_index: int
    units: str = "m"
    coordinate_frame: str = "scene"
    description: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def point_count(self) -> int:
        return len(self.semantic_names)

    @property
    def index_by_name(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.semantic_names)}

    @property
    def children(self) -> dict[int, tuple[int, ...]]:
        result: dict[int, list[int]] = {index: [] for index in range(self.point_count)}
        for child, parent in enumerate(self.parents):
            if parent is not None:
                result[parent].append(child)
        return {key: tuple(value) for key, value in result.items()}

    def validate(self) -> KeypointLayoutDefinition:
        count = self.point_count
        if count == 0:
            raise LayoutValidationError(f"{self.name}: layout has no points")
        if len(set(self.semantic_names)) != count or any(not name for name in self.semantic_names):
            raise LayoutValidationError(f"{self.name}: semantic names must be unique and non-empty")
        if len(self.parents) != count:
            raise LayoutValidationError(f"{self.name}: parent count must equal point count")
        if self.wrist_index < 0 or self.wrist_index >= count:
            raise LayoutValidationError(f"{self.name}: wrist index is out of range")
        if self.parents[self.wrist_index] is not None:
            raise LayoutValidationError(f"{self.name}: wrist must be the graph root")
        for child, parent in enumerate(self.parents):
            if child == self.wrist_index:
                continue
            if parent is None or parent < 0 or parent >= count or parent == child:
                raise LayoutValidationError(
                    f"{self.name}: point {child} must have one valid parent"
                )
        normalized_edges = tuple((int(parent), int(child)) for parent, child in self.edges)
        if len(normalized_edges) != count - 1 or len(set(normalized_edges)) != len(
            normalized_edges
        ):
            raise LayoutValidationError(f"{self.name}: expected exactly {count - 1} unique edges")
        expected_edges = {
            (parent, child) for child, parent in enumerate(self.parents) if parent is not None
        }
        if set(normalized_edges) != expected_edges:
            raise LayoutValidationError(f"{self.name}: edges do not match parents")
        for parent, child in normalized_edges:
            if not (0 <= parent < count and 0 <= child < count):
                raise LayoutValidationError(f"{self.name}: edge index is out of range")
        for index in range(count):
            seen: set[int] = set()
            cursor: int | None = index
            while cursor is not None:
                if cursor in seen:
                    raise LayoutValidationError(f"{self.name}: parent graph contains a cycle")
                seen.add(cursor)
                cursor = self.parents[cursor]
        if len(self.fingertip_indices) not in {0, 5} or len(set(self.fingertip_indices)) != len(
            self.fingertip_indices
        ):
            raise LayoutValidationError(
                f"{self.name}: fingertip indices must be empty or five unique indices"
            )
        if any(index < 0 or index >= count for index in self.fingertip_indices):
            raise LayoutValidationError(f"{self.name}: fingertip index is out of range")
        leaves = {index for index, children in self.children.items() if not children}
        if self.fingertip_indices and not set(self.fingertip_indices).issubset(leaves):
            raise LayoutValidationError(f"{self.name}: every fingertip must be a leaf")
        if self.units != "m":
            raise LayoutValidationError(f"{self.name}: only meter units are supported")
        if self.coordinate_frame != "scene":
            raise LayoutValidationError(f"{self.name}: the canonical layout must be scene-frame")
        return self

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> KeypointLayoutDefinition:
        result = cls(
            name=str(values["name"]),
            version=str(values["version"]),
            semantic_names=tuple(str(item) for item in values["names"]),
            parents=tuple(None if item is None else int(item) for item in values["parents"]),
            edges=tuple((int(item[0]), int(item[1])) for item in values["edges"]),
            fingertip_indices=tuple(int(item) for item in values.get("fingertip_indices", [])),
            wrist_index=int(values["wrist_index"]),
            units=str(values.get("units", "m")),
            coordinate_frame=str(values.get("coordinate_frame", "scene")),
            description=str(values.get("description", "")),
            aliases=tuple(str(item) for item in values.get("aliases", [])),
        )
        return result.validate()


__all__ = ["KeypointLayoutDefinition", "LayoutValidationError"]
