from pathlib import Path

import pytest
import typer

from toporetarget.cli import data
from toporetarget.data.adapters.base import FrameRange


def test_hocap_conversion_loads_one_explicit_primary_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class Adapter:
        def load_sequence(self, sequence: str, **kwargs: object) -> object:
            calls["sequence"] = sequence
            calls.update(kwargs)
            return object()

    monkeypatch.setattr(data, "_hocap_adapter", lambda **kwargs: Adapter())
    result = data._load_raw(
        "hocap",
        "subject_6/20231025_111118",
        sequence_path=None,
        hand="right",
        grab_root=None,
        mano_model_root=Path("/mano"),
        frame_range=FrameRange(0, 41),
        data_root=Path("/raw"),
        primary_object="G06_1",
    )

    assert result is not None
    assert calls == {
        "sequence": "subject_6/20231025_111118",
        "frame_range": FrameRange(0, 41),
        "hand": "right",
        "primary_object_id": "G06_1",
    }


def test_hocap_conversion_requires_explicit_primary_object() -> None:
    with pytest.raises(typer.BadParameter, match="primary-object"):
        data._load_raw(
            "hocap",
            "subject_6/20231025_111118",
            sequence_path=None,
            hand="right",
            grab_root=None,
            mano_model_root=None,
            frame_range=None,
        )
