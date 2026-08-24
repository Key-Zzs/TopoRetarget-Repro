from __future__ import annotations

from scripts.data.freeze_hocap_episode_held_out import select_episodes


def _row(index: int, *, sequence: str, subject: str, target: str) -> dict[str, object]:
    return {
        "episode_id": f"episode_{index}",
        "raw_sequence": sequence,
        "subject": subject,
        "target_object": target,
        "physicalization_v1_eligible": True,
    }


def test_selection_is_deterministic_and_prefers_metadata_diversity() -> None:
    rows = [
        _row(
            index,
            sequence=f"subject_{index}/20231025_{index:06d}",
            subject=f"s{index}",
            target=f"G{index}",
        )
        for index in range(7)
    ]

    first, _ = select_episodes(rows, count=5, seed=42, excluded_suffixes=())
    second, _ = select_episodes(list(reversed(rows)), count=5, seed=42, excluded_suffixes=())

    assert [row["episode_id"] for row in first] == [row["episode_id"] for row in second]
    assert len({row["raw_sequence"] for row in first}) == 5
    assert len({row["target_object"] for row in first}) == 5
    assert len({row["subject"] for row in first}) == 5


def test_development_and_ineligible_rows_cannot_be_selected() -> None:
    rows = [
        _row(0, sequence="subject_1/20231025_170650", subject="s0", target="G0"),
        *[
            _row(
                index,
                sequence=f"subject_{index}/20231025_{index:06d}",
                subject=f"s{index}",
                target=f"G{index}",
            )
            for index in range(1, 7)
        ],
    ]
    rows[1]["physicalization_v1_eligible"] = False

    selected, audited = select_episodes(rows, count=5, seed=42, excluded_suffixes=("170650",))

    assert all(not str(row["raw_sequence"]).endswith("170650") for row in selected)
    assert all(row["episode_id"] != "episode_1" for row in selected)
    excluded = next(row for row in audited if row["episode_id"] == "episode_0")
    assert str(excluded["development_exclusion_reason"]).startswith("DEVELOPMENT")
