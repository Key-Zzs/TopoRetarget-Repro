# Upstream references

The local ManipTrans checkout is used only as an acquisition-side source for the Arti-MANO asset
importer. This repository does not copy its Python code or add it as a submodule.

Relevant upstream relative paths for future work:

- `maniptrans_envs/assets/mano_urdf`
- `maniptrans_envs/lib/envs/dexhands/artimano.py`
- `main/dataset/mano2dexhand.py`

F0 vendors the pinned Arti-MANO payload under `third_party/robot_hands/artimano/`. Its
`SOURCE.yaml` records the upstream commit, source license hash, imported file hashes, and the
path-rebasing transformation. The old ignored `.local/assets/artimano/asset_manifest.json` is
retained only as a historical compatibility reference.
