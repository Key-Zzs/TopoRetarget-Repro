"""Lightweight, opt-in dataset indexes."""

from toporetarget.data.indexes.grab import (
    GrabIndexError,
    build_grab_index,
    load_grab_index,
    resolve_grab_dataset_root,
)

__all__ = ["GrabIndexError", "build_grab_index", "load_grab_index", "resolve_grab_dataset_root"]
