# Upstream references

The local ManipTrans checkout is used only as an acquisition-side source for the Arti-MANO asset
importer. This repository does not copy its Python code or add it as a submodule.

Relevant upstream relative paths for future work:

- `maniptrans_envs/assets/mano_urdf`
- `maniptrans_envs/lib/envs/dexhands/artimano.py`
- `main/dataset/mano2dexhand.py`

The current importer records the upstream commit, source license hash, and imported file hashes in
the ignored local manifest at `.local/assets/artimano/asset_manifest.json`.

