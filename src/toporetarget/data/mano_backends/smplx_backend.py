"""Explicit SMPL-X/MANO reconstruction backend.

This backend intentionally has no field-name or array-width inference.  A
``ManoReconstructionRequest`` states whether its hand pose is PCA or
axis-angle before SMPL-X is ever invoked.
"""

from __future__ import annotations

import hashlib
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
from toporetarget.data.mano_backends.contracts import (
    AmbiguousManoPoseRepresentationError,
    ManoModelProvenance,
    ManoModelProvenanceError,
    ManoPoseRepresentation,
    ManoReconstructionRequest,
    ManoReconstructionResult,
)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SmplxManoBackend:
    """Use optional SMPL-X with an explicit, float64 MANO contract."""

    def __init__(self, model_root: str | Path, *, device: str = "cpu") -> None:
        self.model_root = Path(model_root).expanduser()
        if not self.model_root.is_dir() and not self.model_root.is_file():
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
                "MANO reconstruction needs optional torch and smplx; install with "
                "`pip install -e '.[grab]'`"
            ) from exc
        self._smplx = smplx
        self._torch = torch
        self.device = device
        self._layer_cache: dict[tuple[Any, ...], Any] = {}

    def _model_path_for_side(self, side: str) -> Path:
        filename = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
        if self.model_root.is_file():
            if self.model_root.name != filename:
                raise ManoBackendError(
                    f"MANO model file {self.model_root} does not match requested side {side}"
                )
            return self.model_root
        direct = self.model_root / filename
        nested = self.model_root / "mano" / filename
        if direct.is_file():
            return direct
        if nested.is_file():
            return nested
        raise ManoBackendError(
            f"MANO {filename} was not found below {self.model_root}; expected the file directly "
            "or under a mano/ subdirectory"
        )

    def model_provenance(
        self,
        *,
        side: str,
        dataset_name: str,
        source_annotation_path: str | Path,
        source_annotation_hash: str,
        model_version: str = "MANO v1.2",
    ) -> ManoModelProvenance:
        """Resolve the exact side model instead of accepting path-based side guesses."""

        model_path = self._model_path_for_side(side)
        return ManoModelProvenance(
            model_path=str(model_path),
            model_hash=_sha256_file(model_path),
            model_version=model_version,
            dataset_name=dataset_name,
            source_annotation_path=str(source_annotation_path),
            source_annotation_hash=source_annotation_hash,
        )

    @property
    def cache_keys(self) -> tuple[tuple[Any, ...], ...]:
        """Expose cache identity for contract tests without exposing layers."""

        return tuple(self._layer_cache)

    def _validated_model_path(self, request: ManoReconstructionRequest) -> Path:
        expected = self._model_path_for_side(request.side).resolve()
        declared = Path(request.model_path).expanduser().resolve()
        if declared != expected:
            raise ManoModelProvenanceError(
                "request model_path does not match the explicitly selected side model: "
                f"declared={declared}, expected={expected}"
            )
        actual_hash = _sha256_file(expected)
        if actual_hash != request.model_hash:
            raise ManoModelProvenanceError(
                "request model_hash does not match the resolved MANO model: "
                f"declared={request.model_hash}, actual={actual_hash}"
            )
        return expected

    @staticmethod
    def _template_hash(v_template: np.ndarray | None) -> str | None:
        if v_template is None:
            return None
        value = np.asarray(v_template, dtype=np.float64)
        if value.shape != (778, 3):
            raise ManoModelProvenanceError(f"v_template must have shape [778,3], got {value.shape}")
        return _sha256_array(value)

    def _layer(
        self,
        request: ManoReconstructionRequest,
        *,
        model_path: Path,
        use_pca: bool,
        v_template: np.ndarray | None,
    ) -> Any:
        template_hash = self._template_hash(v_template)
        representation = ManoPoseRepresentation(request.pose_representation)
        # The leading fields are the required cache identity.  The explicit
        # use_pca/template suffix prevents an implementation-detail collision.
        key = (
            request.side,
            representation.value,
            request.num_pca_components,
            request.flat_hand_mean,
            request.model_hash,
            np.dtype(np.float64).str,
            self.device,
            use_pca,
            template_hash,
        )
        if key in self._layer_cache:
            return self._layer_cache[key]
        model_kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "model_type": "mano",
            "is_rhand": request.side == "right",
            "flat_hand_mean": request.flat_hand_mean,
            "batch_size": request.frame_count,
            "use_pca": use_pca,
        }
        if representation is ManoPoseRepresentation.PCA:
            assert request.num_pca_components is not None
            model_kwargs["num_pca_comps"] = request.num_pca_components
        if v_template is not None:
            model_kwargs["v_template"] = np.asarray(v_template, dtype=np.float64)
        try:
            layer = self._smplx.create(**model_kwargs).to(
                device=self.device, dtype=self._torch.float64
            )
        except Exception as exc:  # smplx adds model-specific detail
            raise ManoBackendError(f"SMPL-X/MANO layer creation failed: {exc}") from exc
        self._layer_cache[key] = layer
        return layer

    @staticmethod
    def _basis_and_mean(layer: Any, components: int) -> tuple[np.ndarray, np.ndarray]:
        basis_value = getattr(layer, "np_hand_components", None)
        if basis_value is None:
            basis_value = getattr(layer, "hand_components", None)
        if basis_value is None:
            raise ManoBackendError("SMPL-X MANO layer exposes no PCA basis")
        basis = np.asarray(basis_value, dtype=np.float64)
        mean = np.asarray(layer.hand_mean.detach().cpu().numpy(), dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] < components or basis.shape[1] != 45:
            raise ManoBackendError(
                "SMPL-X MANO PCA basis shape is incompatible with declared components: "
                f"basis={basis.shape}, components={components}"
            )
        if mean.shape != (45,):
            raise ManoBackendError(f"SMPL-X MANO hand mean must be [45], got {mean.shape}")
        return basis[:components], mean

    def reconstruct(
        self,
        request: ManoReconstructionRequest,
        *,
        v_template: np.ndarray | None = None,
    ) -> ManoReconstructionResult:
        """Reconstruct one request without pose-representation inference."""

        representation = ManoPoseRepresentation(request.pose_representation)
        if not np.isfinite(request.global_orient).all() or not np.isfinite(request.hand_pose).all():
            raise ManoBackendError("MANO request pose contains non-finite values")
        if not np.isfinite(request.translation).all():
            raise ManoBackendError("MANO request translation contains non-finite values")
        model_path = self._validated_model_path(request)
        direct_pca = (
            representation is ManoPoseRepresentation.PCA
            and request.num_pca_components is not None
            and request.num_pca_components < 45
        )
        # SMPL-X changes ``use_pca`` to False at K=45.  Passing PCA45 directly
        # would therefore reinterpret it as axis-angle.  Expand with exactly
        # this model's basis/mean, then use the non-PCA layer deliberately.
        layer = self._layer(
            request, model_path=model_path, use_pca=direct_pca, v_template=v_template
        )
        hand_pose_axis_angle: np.ndarray
        pca_basis_hash: str | None = None
        hand_mean_hash: str | None = None
        if representation is ManoPoseRepresentation.PCA:
            assert request.num_pca_components is not None
            basis, hand_mean = self._basis_and_mean(layer, request.num_pca_components)
            hand_pose_axis_angle = request.hand_pose @ basis + hand_mean
            pca_basis_hash = _sha256_array(basis)
            hand_mean_hash = _sha256_array(hand_mean)
            model_hand_pose = request.hand_pose if direct_pca else hand_pose_axis_angle
            execution = "native_pca_layer" if direct_pca else "pca45_explicit_basis_expansion"
        else:
            hand_pose_axis_angle = request.hand_pose.copy()
            model_hand_pose = hand_pose_axis_angle
            execution = "native_axis_angle_layer"
        torch = self._torch
        kwargs: dict[str, Any] = {
            "global_orient": torch.as_tensor(
                request.global_orient, dtype=torch.float64, device=self.device
            ),
            "hand_pose": torch.as_tensor(model_hand_pose, dtype=torch.float64, device=self.device),
            "transl": torch.as_tensor(request.translation, dtype=torch.float64, device=self.device),
        }
        betas = request.broadcast_betas()
        if betas is not None:
            kwargs["betas"] = torch.as_tensor(betas, dtype=torch.float64, device=self.device)
        try:
            with torch.no_grad():
                output = layer(**kwargs)
        except Exception as exc:  # smplx error messages carry model-specific detail
            raise ManoBackendError(f"SMPL-X/MANO reconstruction failed: {exc}") from exc
        vertices = output.vertices.detach().cpu().numpy().astype(np.float64, copy=False)
        joints_value = getattr(output, "joints", None)
        if joints_value is None:
            raise ManoBackendError("SMPL-X/MANO output omitted posed joints")
        joints = joints_value.detach().cpu().numpy().astype(np.float64, copy=False)
        if vertices.shape != (request.frame_count, 778, 3):
            raise ManoBackendError(f"MANO vertices have unexpected shape {vertices.shape}")
        if joints.ndim != 3 or joints.shape[0] != request.frame_count or joints.shape[2] != 3:
            raise ManoBackendError(f"MANO posed joints have unexpected shape {joints.shape}")
        if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
            raise ManoBackendError("SMPL-X/MANO output contains non-finite values")
        rotation = axis_angle_to_matrix(request.global_orient)
        wrist_pose = np.broadcast_to(
            np.eye(4, dtype=np.float64), (request.frame_count, 4, 4)
        ).copy()
        wrist_pose[:, :3, :3] = rotation
        wrist_pose[:, :3, 3] = request.translation
        full_pose = np.concatenate([request.global_orient, hand_pose_axis_angle], axis=1)
        provenance = ManoModelProvenance(
            model_path=str(model_path),
            model_hash=request.model_hash,
            model_version=request.model_version,
            dataset_name=request.dataset_name,
            source_annotation_path=str(request.source_annotation_path),
            source_annotation_hash=request.source_annotation_hash,
            pca_basis_hash=pca_basis_hash,
            hand_mean_hash=hand_mean_hash,
        )
        layout = "mano16_smplx" if joints.shape[1] == 16 else f"mano{joints.shape[1]}_smplx"
        return ManoReconstructionResult(
            vertices=vertices,
            posed_joints_native=joints,
            posed_joint_layout=layout,
            full_pose_axis_angle=full_pose,
            global_orient_axis_angle=request.global_orient.copy(),
            hand_pose_axis_angle=hand_pose_axis_angle,
            faces=np.asarray(layer.faces, dtype=np.int64),
            side=request.side,
            betas=betas,
            translation=request.translation.copy(),
            wrist_pose_scene=wrist_pose,
            model_provenance=provenance,
            reconstruction_manifest={
                "contract_version": "toporetarget.mano.reconstruction.v2",
                "pose_representation": representation.value,
                "num_pca_components": request.num_pca_components,
                "flat_hand_mean": request.flat_hand_mean,
                "units": request.units,
                "dtype": "float64",
                "betas_required": request.betas_required,
                "betas_broadcast_shape": None if betas is None else list(betas.shape),
                "pca_basis_hash": pca_basis_hash,
                "hand_mean_hash": hand_mean_hash,
                "execution": execution,
            },
        )

    @staticmethod
    def _as_render_result(result: ManoReconstructionResult) -> ManoRenderResult:
        """Bridge the explicit result to the legacy GRAB renderer shape."""

        return ManoRenderResult(
            vertices_scene=result.vertices,
            faces=result.faces,
            wrist_pose_scene=result.wrist_pose_scene,
            joints_scene=result.posed_joints_native,
            keypoint_layout=result.posed_joint_layout,
            model_profile=result.reconstruction_manifest["execution"],
        )

    def render_axis_angle(
        self,
        *,
        params: dict[str, np.ndarray],
        v_template: np.ndarray,
        side: str,
        frame_count: int,
        flat_hand_mean: bool,
        dataset_name: str,
        source_annotation_path: str | Path,
        source_annotation_hash: str,
        model_version: str = "MANO v1.2",
    ) -> ManoRenderResult:
        """Explicit compatibility wrapper for existing axis-angle-only callers."""

        pose = params.get("fullpose")
        if pose is None:
            raise AmbiguousManoPoseRepresentationError(
                "legacy wrapper requires caller-declared axis-angle params['fullpose']"
            )
        betas_value = params.get("betas")
        provenance = self.model_provenance(
            side=side,
            dataset_name=dataset_name,
            source_annotation_path=source_annotation_path,
            source_annotation_hash=source_annotation_hash,
            model_version=model_version,
        )
        request = ManoReconstructionRequest(
            side=side,
            pose_representation=ManoPoseRepresentation.AXIS_ANGLE,
            global_orient=np.asarray(params["global_orient"], dtype=np.float64),
            hand_pose=np.asarray(pose, dtype=np.float64),
            num_pca_components=None,
            translation=np.asarray(params["transl"], dtype=np.float64),
            betas=None if betas_value is None else np.asarray(betas_value, dtype=np.float64),
            flat_hand_mean=flat_hand_mean,
            units="metre",
            dtype=np.float64,
            model_path=provenance.model_path,
            model_hash=provenance.model_hash,
            model_version=provenance.model_version,
            dataset_name=provenance.dataset_name,
            source_annotation_path=provenance.source_annotation_path,
            source_annotation_hash=provenance.source_annotation_hash,
            betas_required=betas_value is not None,
        )
        if request.frame_count != frame_count:
            raise ManoBackendError(
                f"axis-angle wrapper frame count mismatch: {request.frame_count} != {frame_count}"
            )
        return self._as_render_result(self.reconstruct(request, v_template=v_template))

    def render(
        self,
        *,
        params: dict[str, np.ndarray],
        v_template: np.ndarray,
        side: str,
        frame_count: int,
    ) -> ManoRenderResult:
        """Reject the old ambiguous API instead of inferring pose semantics."""

        del params, v_template, side, frame_count
        raise AmbiguousManoPoseRepresentationError(
            "SmplxManoBackend.render is ambiguous; use reconstruct(request=...) or "
            "render_axis_angle(..., flat_hand_mean=..., provenance=...)"
        )


__all__ = ["SmplxManoBackend"]
