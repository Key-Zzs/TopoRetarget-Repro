# Tracked robot-hand assets

F0 makes Arti-MANO the first tracked instance of the generic target-hand contract.

```text
third_party/robot_hands/
├── README.md
└── artimano/
    ├── LICENSE
    ├── NOTICE.md
    ├── SOURCE.yaml
    ├── asset_manifest.json
    ├── urdf/
    │   ├── lh_mano.urdf
    │   └── rh_mano.urdf
    └── meshes/
        ├── lh_urdf_meshes/
        ├── lh_urdf_meshes_visonly/
        ├── rh_urdf_meshes/
        └── rh_urdf_meshes_visonly/
```

The tracked snapshot contains 98 source payload files: two URDFs and 96 mesh files. The source
manifest hash is `1d14cce93e2ee09dedbfcda842b1d8aac29443f86b57a0a15f6289bd55e0f771`, and the
upstream commit is `a3d08cfe3c3a5868a7f057533bcaf759c5af4705`.

The path rebasing changes only URDF mesh filenames from the upstream flat layout to
`../meshes/...`. It does not change URDF topology, joint axes/limits, qpos ordering, mesh bytes,
geometry origins, or simulator behavior. `.local/reports/f0/asset_comparison.json` and
`numerical_regression.json` are the local evidence for that claim.

Use the registry rather than constructing Arti-MANO directly:

```bash
toporetarget robots list
toporetarget robots resolve-assets
toporetarget robots inspect --robot artimano_rh
toporetarget robots validate --robot artimano_rh
toporetarget robots compare-assets --reference-root .local/assets/artimano
```

Wuji Hand2 is intentionally not present in F0. Its tracked assets and registry entries are a
later main-branch milestone.
