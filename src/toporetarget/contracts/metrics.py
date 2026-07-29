"""MetricRegistry v1 with explicit paper/proxy/geometry/engineering semantics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .version import METRIC_REGISTRY_V1


class MetricType(str, Enum):
    PAPER_EXACT = "PAPER_EXACT"
    DATASET_PROXY = "DATASET_PROXY"
    GENERIC_GEOMETRIC = "GENERIC_GEOMETRIC"
    ENGINEERING_DIAGNOSTIC = "ENGINEERING_DIAGNOSTIC"


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    metric_type: MetricType
    display_name: str
    mathematical_definition: str
    unit: str
    direction: str
    applicable_datasets: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    missing_data_behavior: str = "N/A with reason; never zero"
    aggregation_rule: str = "per-unit then equal-weight macro mean/median"
    implementation_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("metric_id must not be empty")
        if not isinstance(self.metric_type, MetricType):
            object.__setattr__(self, "metric_type", MetricType(str(self.metric_type)))

    @property
    def semantics(self) -> str:
        """Compatibility name used by the Q1--Q3 benchmark registry."""

        return self.metric_type.value

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["metric_type"] = self.metric_type.value
        return value


class MetricRegistry:
    """Explicit metric registry that refuses duplicate IDs."""

    schema_version = METRIC_REGISTRY_V1

    def __init__(self, definitions: Iterable[MetricSpec] = ()) -> None:
        self._definitions: dict[str, MetricSpec] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: MetricSpec, *, replace: bool = False) -> None:
        if definition.metric_id in self._definitions and not replace:
            raise ValueError(f"metric already registered: {definition.metric_id}")
        self._definitions[definition.metric_id] = definition

    def get(self, metric_id: str) -> MetricSpec:
        try:
            return self._definitions[metric_id]
        except KeyError as exc:
            raise KeyError(f"unknown metric {metric_id!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def definitions(self) -> tuple[MetricSpec, ...]:
        return tuple(self._definitions[name] for name in self.names())

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        for definition in self.definitions():
            if definition.metric_type not in tuple(MetricType):
                errors.append(f"{definition.metric_id}: unsupported metric_type")
            if definition.metric_type == MetricType.DATASET_PROXY and any(
                token in definition.display_name.lower()
                for token in ("ground truth", "ground-truth", "gt")
            ):
                errors.append(f"{definition.metric_id}: proxy must not be labeled ground truth")
        return {
            "schema_version": METRIC_REGISTRY_V1,
            "valid": not errors,
            "errors": errors,
            "metric_count": len(self._definitions),
        }

    def payload(self) -> dict[str, Any]:
        report = self.validate()
        if not report["valid"]:
            raise ValueError("invalid metric registry: " + "; ".join(report["errors"]))
        return {
            "schema_version": METRIC_REGISTRY_V1,
            "metrics": [definition.as_dict() for definition in self.definitions()],
        }


def from_legacy_metric_definitions(definitions: Iterable[Any]) -> MetricRegistry:
    """Adapt the existing benchmark registry without changing its formulas."""

    result: list[MetricSpec] = []
    for definition in definitions:
        metric_type = getattr(definition, "metric_type", getattr(definition, "semantics", ""))
        if isinstance(metric_type, MetricType):
            normalized_type = metric_type
        else:
            normalized_type = MetricType(getattr(metric_type, "value", metric_type))
        result.append(
            MetricSpec(
                metric_id=definition.metric_id,
                metric_type=normalized_type,
                display_name=definition.display_name,
                mathematical_definition=definition.mathematical_definition,
                unit=definition.unit,
                direction=definition.direction,
                applicable_datasets=tuple(definition.applicable_datasets),
                required_inputs=tuple(definition.required_inputs),
                missing_data_behavior=definition.missing_data_behavior,
                aggregation_rule=definition.aggregation_rule,
                implementation_version=definition.implementation_version,
            )
        )
    return MetricRegistry(result)


def get_metric_registry() -> MetricRegistry:
    from toporetarget.metrics.registry import metric_definitions

    return from_legacy_metric_definitions(metric_definitions())


__all__ = [
    "METRIC_REGISTRY_V1",
    "MetricRegistry",
    "MetricSpec",
    "MetricType",
    "from_legacy_metric_definitions",
    "get_metric_registry",
]
