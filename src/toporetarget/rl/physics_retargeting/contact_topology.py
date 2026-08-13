"""Shared, data-derived persistent contact-topology extraction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from .contracts import ContactWindowV1, PersistentContactTopologyV1


def body_contact_group(body_name: str) -> str | None:
    name = body_name.lower()
    # Match the anatomical prefix before segment labels such as ``_middle``.
    # Substring ordering would otherwise classify ``r_ring_finger_middle`` as
    # the middle finger and turn adjacent links into a fake inter-finger pair.
    anatomical_markers = (
        ("thumb", ("thumb",)),
        ("index", ("index_finger",)),
        ("middle", ("middle_finger",)),
        ("ring", ("ring_finger",)),
        ("pinky", ("pinky",)),
    )
    for group, markers in anatomical_markers:
        if any(marker in name for marker in markers):
            return group
    if "palm" in name or "wrist" in name:
        return "palm"
    return None


def collapse_contact_records(
    records: Sequence[Mapping[str, object]], *, force_threshold_n: float = 1.0e-4
) -> dict[int, set[str]]:
    if force_threshold_n <= 0.0:
        raise ValueError("contact force threshold must be positive")
    by_step: dict[int, set[str]] = defaultdict(set)
    for row in records:
        vector = row.get("net_contact_force_world_on_object_n")
        bodies = row.get("present_hand_body_names")
        step = row.get("control_step")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not isinstance(vector, list)
            or len(vector) != 3
            or not isinstance(bodies, list)
        ):
            continue
        magnitude = math.sqrt(sum(float(value) ** 2 for value in vector))
        if not math.isfinite(magnitude) or magnitude <= force_threshold_n:
            continue
        for body in bodies:
            if isinstance(body, str) and body_contact_group(body) is not None:
                by_step[step].add(body)
    return dict(by_step)


def consecutive_runs(steps: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(int(value) for value in steps))
    if not ordered:
        return ()
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for step in ordered[1:]:
        if step != previous + 1:
            runs.append((start, previous))
            start = step
        previous = step
    runs.append((start, previous))
    return tuple(runs)


def extract_persistent_contact_topology(
    *,
    clip: str,
    contact_records: Sequence[Mapping[str, object]],
    retimed_frame_count: int,
    force_threshold_n: float = 1.0e-4,
) -> PersistentContactTopologyV1:
    if retimed_frame_count < 2:
        raise ValueError("contact topology requires at least two frames")
    by_step = collapse_contact_records(contact_records, force_threshold_n=force_threshold_n)
    group_steps: dict[str, set[int]] = defaultdict(set)
    body_counts: Counter[str] = Counter()
    for step, bodies in by_step.items():
        for body in bodies:
            group = body_contact_group(body)
            if group is not None:
                group_steps[group].add(step)
                body_counts[body] += 1
    if not group_steps:
        raise ValueError("TASK_SEMANTIC_CLASSIFICATION_AMBIGUOUS: no valid hand-object contact")
    group_counts = {group: len(steps) for group, steps in group_steps.items()}
    max_count = max(group_counts.values())
    required = tuple(sorted(group for group, count in group_counts.items() if count == max_count))
    optional = tuple(sorted(set(group_counts) - set(required)))
    all_steps = sorted(by_step)
    onset_margin = max(2, int(round(0.025 * retimed_frame_count)))
    onset = all_steps[0]
    final = all_steps[-1]
    runs = consecutive_runs(all_steps)
    longest = max(end - start + 1 for start, end in runs)
    final_hold_start = max(0, retimed_frame_count - max(3, retimed_frame_count // 16))
    final_contact = [step for step in all_steps if step >= final_hold_start]
    return PersistentContactTopologyV1(
        clip=clip,
        required_body_groups=required,
        optional_body_groups=optional,
        forbidden_unrelated_contacts=("inactive_object", "self_contact", "support", "ground"),
        minimum_persistence_control_steps=max(1, int(math.ceil(0.50 * longest))),
        source_onset_window=ContactWindowV1(
            max(0, onset - onset_margin), min(retimed_frame_count - 1, onset + onset_margin)
        ),
        final_hold_window=ContactWindowV1(
            final_hold_start if final_contact else final,
            retimed_frame_count - 1 if final_contact else final,
        ),
        contact_graph_edges=tuple((group, "object") for group in sorted(group_counts)),
        group_weights={
            group: max(0.25, count / max_count) for group, count in group_counts.items()
        },
        source_group_step_counts=dict(sorted(group_counts.items())),
        transient_filter_control_steps=max(1, int(round(0.005 * retimed_frame_count))),
        raw_point_precision="aggregate_object_force_only_point_contacts_unavailable",
    )


__all__ = [
    "body_contact_group",
    "collapse_contact_records",
    "consecutive_runs",
    "extract_persistent_contact_topology",
]
