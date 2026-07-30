"""Raw dataset readers with lazy implementation exports."""

__all__ = ["GrabSequenceRecord", "load_grab_auxiliary", "read_grab_npz"]


def __getattr__(name: str):
    if name in __all__:
        from toporetarget.data.readers.grab import (
            GrabSequenceRecord,
            load_grab_auxiliary,
            read_grab_npz,
        )

        return {
            "GrabSequenceRecord": GrabSequenceRecord,
            "load_grab_auxiliary": load_grab_auxiliary,
            "read_grab_npz": read_grab_npz,
        }[name]
    raise AttributeError(name)
