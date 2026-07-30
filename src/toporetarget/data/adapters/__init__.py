"""Dataset adapter contracts and implementations.

The GRAB adapter imports the low-level reader, while the reader imports the
``FrameRange`` contract from this package.  Keep the implementation exports
lazy so importing either low-level module does not create a package-init
cycle.
"""

from toporetarget.data.adapters.base import FrameRange, HOIDatasetAdapter

__all__ = ["FrameRange", "GrabDatasetAdapter", "GrabLoadOptions", "HOIDatasetAdapter"]


def __getattr__(name: str):
    if name in {"GrabDatasetAdapter", "GrabLoadOptions"}:
        from toporetarget.data.adapters.grab import GrabDatasetAdapter, GrabLoadOptions

        return {"GrabDatasetAdapter": GrabDatasetAdapter, "GrabLoadOptions": GrabLoadOptions}[name]
    raise AttributeError(name)
