import json

from toporetarget.workflows.cache import cache_record, can_reuse
from toporetarget.workflows.schema import stable_hash


def test_synthetic_stage10_artifact_chain_is_reusable_and_inspectable(tmp_path) -> None:
    source = tmp_path / "source.npz"
    canonical = tmp_path / "canonical.zarr"
    final = tmp_path / "final.zarr"
    source.write_bytes(b"synthetic-source")
    canonical.write_bytes(b"canonical")
    final.write_bytes(b"final")
    records = []
    for node_id, artifact in (("canonicalize_grab", canonical), ("final_refinement", final)):
        record = cache_record(
            node_id=node_id,
            expected_signature=stable_hash({"node": node_id, "source": str(source)}),
            output_paths={"artifact": str(artifact)},
        )
        path = tmp_path / f"{node_id}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        records.append((path, record["expected_signature"]))
    assert all(can_reuse(path, expected_signature=signature)[0] for path, signature in records)
