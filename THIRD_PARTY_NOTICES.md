# Third-party notices

## ManipTrans / Arti-MANO

`third_party/robot_hands/artimano/` is a tracked vendor snapshot of
`maniptrans_envs/assets/mano_urdf` from ManipTrans commit
`a3d08cfe3c3a5868a7f057533bcaf759c5af4705`. It contains the two URDFs and mesh payloads only; no
ManipTrans Python source is copied. The snapshot preserves the upstream mesh and kinematic bytes.
Only relative URDF mesh paths are rebased into the repository's `meshes/` directory.

The upstream checkout declares GNU GPL Version 3 in its root `LICENSE`; no separate license file
was found below the Arti-MANO asset directory during the F0 audit. That exact license is retained
at `third_party/robot_hands/artimano/LICENSE`. `SOURCE.yaml`, `NOTICE.md`, and
`.local/reports/f0/license_audit.json` record provenance and the audit decision. This is a notice
of the observed upstream files, not a legal opinion about redistribution.

Users redistributing this repository or the vendor snapshot must retain the included notices and
license and independently verify obligations for their distribution context. External datasets and
MANO/SMPL-X model files remain outside the repository and retain their own licenses.

The old `.local/assets/artimano/` import is not tracked or required for normal execution. It remains
as a compatibility fallback only and emits a deprecation warning.
