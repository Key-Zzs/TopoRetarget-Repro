# F0: tracked robot assets and generic target-hand foundation

F0 migrates Arti-MANO from the default `.local/assets/artimano` import to a tracked vendor
snapshot and makes it the first instance of the generic target-hand contract.

## Completed scope

- tracked Arti-MANO snapshot with source/license/notice manifests;
- deterministic vendor dry-run and license gate;
- tracked/override/legacy resolver with explicit warnings and provenance;
- generic contract objects and registry-backed RH/LH configs;
- CLI commands `robots list`, `resolve-assets`, `inspect`, `validate`, `fk`, `anchors`,
  `jacobian-check`, `visualize`, and `compare-assets`;
- legacy API delegation and historical absolute-path rebinding evidence;
- exact RH/LH migration regression for topology, qpos, FK, anchors, Jacobians, geometry transforms,
  and mesh payload bytes;
- synchronized user/development/fidelity documentation.

## Explicitly not in F0

Wuji Hand2, SDF penetration loss, solver-profile changes, Stage 7–9 equation changes, Stage 10
artifact rewrites, `develop/pene-loss`, RL, and changes to the ManipTrans checkout are out of scope.

## Evidence

The local audit outputs are under `.local/reports/f0/`. They are cache/report artifacts and are not
tracked. The current tracked snapshot uses ManipTrans commit
`a3d08cfe3c3a5868a7f057533bcaf759c5af4705` and source manifest hash
`1d14cce93e2ee09dedbfcda842b1d8aac29443f86b57a0a15f6289bd55e0f771`.

## Next plan

`main`: add Wuji Hand2 Beta1 assets and a generic registration, then run bounded GRAB→Wuji
validation. `develop/pene-loss`: add the generic SDF penetration loss. After both lanes are
complete, integrate Arti/Wuji × baseline/SDF and validate multi-trajectory export. RL remains
postponed; ContactPose formal evaluation remains later.
