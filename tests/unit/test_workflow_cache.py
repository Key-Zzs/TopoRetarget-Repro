import json

from toporetarget.workflows.cache import cache_record, can_reuse, path_hash


def test_cache_reuse_requires_matching_artifact_and_validation(tmp_path) -> None:
    artifact = tmp_path / "artifact.txt"
    report = tmp_path / "validation.json"
    artifact.write_text("stable\n", encoding="utf-8")
    report.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    record_path = tmp_path / "cache.json"
    record_path.write_text(
        json.dumps(
            cache_record(
                node_id="synthetic",
                expected_signature="sig",
                output_paths={"artifact": str(artifact)},
                validation_path=str(report),
            )
        ),
        encoding="utf-8",
    )
    assert can_reuse(record_path, expected_signature="sig")[0]
    artifact.write_text("changed\n", encoding="utf-8")
    assert can_reuse(record_path, expected_signature="sig") == (
        False,
        "artifact hash mismatch",
    )


def test_cache_record_does_not_reuse_failed_report(tmp_path) -> None:
    artifact = tmp_path / "artifact.txt"
    report = tmp_path / "validation.json"
    artifact.write_text("stable\n", encoding="utf-8")
    report.write_text(json.dumps({"pass": False}), encoding="utf-8")
    record = cache_record(
        node_id="synthetic",
        expected_signature="sig",
        output_paths={"artifact": str(artifact)},
        validation_path=str(report),
    )
    assert record["validation_status"] == "fail"
    record_path = tmp_path / "cache.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert can_reuse(record_path, expected_signature="sig")[0] is False
    assert path_hash(artifact) == record["output_hashes"]["artifact"]
