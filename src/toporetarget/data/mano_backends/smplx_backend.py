"""Optional official SMPL-X/MANO backend for one GRAB hand clip."""

from __future__ import annotations

import inspect
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.mano_backends.base import (
    ManoBackendError,
    ManoRenderResult,
    axis_angle_to_matrix,
)


class SmplxManoBackend:
    """Use smplx without importing it until a real backend is explicitly requested."""

    def __init__(self, model_root: str | Path) -> None:
        self.model_root = Path(model_root).expanduser()
        if not self.model_root.is_dir():
            raise ManoBackendError(
                f"MANO model root does not exist: {self.model_root}; "
                "set --mano-model-root or MANO_MODEL_ROOT"
            )
        try:
            if not hasattr(inspect, "getargspec"):
                arg_spec = namedtuple("arg_spec", "args varargs keywords defaults")

                def getargspec(function: Any) -> Any:
                    full = inspect.getfullargspec(function)
                    return arg_spec(full.args, full.varargs, full.varkw, full.defaults)

                inspect.getargspec = getargspec  # type: ignore[attr-defined]
            for name, value in {
                "bool": bool,
                "int": int,
                "float": float,
                "complex": complex,
                "object": object,
                "unicode": str,
                "str": str,
            }.items():
                if name not in np.__dict__:
                    setattr(np, name, value)
            import smplx
            import torch
        except ImportError as exc:
            raise ManoBackendError(
                "GRAB reconstruction needs optional torch and smplx; install with "
                "`pip install -e '.[grab]'`"
            ) from exc
        self._smplx = smplx
        self._torch = torch

    def _model_path_for_side(self, side: str) -> Path:
        filename = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
        direct = self.model_root / filename
        nested = self.model_root / "mano" / filename
        if direct.is_file():
            return direct
        if nested.is_file():
            return nested
        if (self.model_root / "mano").is_dir():
            return self.model_root
        raise ManoBackendError(
            f"MANO {filename} was not found below {self.model_root}; expected the file directly "
            "or under a mano/ subdirectory"
        )

    def render(
        self,
        *,
        params: dict[str, np.ndarray],
        v_template: np.ndarray,
        side: str,
        frame_count: int,
    ) -> ManoRenderResult:
        if side not in {"left", "right"}:
            raise ManoBackendError("MANO side must be left or right")
        if "global_orient" not in params or "transl" not in params:
            raise ManoBackendError("GRAB hand params require global_orient and transl")
        fullpose = params.get("fullpose")
        hand_pose = params.get("hand_pose")
        use_pca = fullpose is None
        if use_pca and hand_pose is None:
            raise ManoBackendError("GRAB hand params contain neither fullpose nor hand_pose")
        if fullpose is not None and fullpose.shape[1] != 45:
            raise ManoBackendError(
                f"expected fullpose [T,45], got {fullpose.shape}; refusing silent pose truncation"
            )
        if hand_pose is not None and hand_pose.shape[0] != frame_count:
            raise ManoBackendError("hand pose frame count mismatch")
        torch = self._torch
        model_kwargs: dict[str, Any] = {
            "model_path": str(self._model_path_for_side(side)),
            "model_type": "mano",
            "is_rhand": side == "right",
            "flat_hand_mean": True,
            "v_template": np.asarray(v_template, dtype=np.float32),
            "batch_size": frame_count,
            "use_pca": use_pca,
        }
        if use_pca:
            assert hand_pose is not None
            model_kwargs["num_pca_comps"] = int(hand_pose.shape[1])
        try:
            model = self._smplx.create(**model_kwargs)
            kwargs = {
                "global_orient": torch.as_tensor(params["global_orient"], dtype=torch.float32),
                "transl": torch.as_tensor(params["transl"], dtype=torch.float32),
            }
            kwargs["hand_pose"] = torch.as_tensor(
                fullpose if fullpose is not None else hand_pose, dtype=torch.float32
            )
            with torch.no_grad():
                output = model(**kwargs)
        except Exception as exc:  # smplx error messages carry model-specific detail
            raise ManoBackendError(f"SMPL-X/MANO reconstruction failed: {exc}") from exc
        vertices = output.vertices.detach().cpu().numpy().astype(np.float64)
        joints = getattr(output, "joints", None)
        joints_array = None if joints is None else joints.detach().cpu().numpy().astype(np.float64)
        rotation = axis_angle_to_matrix(np.asarray(params["global_orient"], dtype=np.float64))
        pose = np.repeat(np.eye(4, dtype=np.float64)[None, ...], frame_count, axis=0)
        pose[:, :3, :3] = rotation
        pose[:, :3, 3] = np.asarray(params["transl"], dtype=np.float64)
        layout = (
            None
            if joints_array is None
            else (
                "mano21"
                if joints_array.shape[1] == 21
                else "mano16"
                if joints_array.shape[1] == 16
                else "mano_native"
            )
        )
        return ManoRenderResult(
            vertices,
            np.asarray(model.faces, dtype=np.int64),
            pose,
            joints_array,
            layout,
            "smplx_mano_fullpose" if fullpose is not None else "smplx_mano_pca_exact_input",
        )


__all__ = ["SmplxManoBackend"]
