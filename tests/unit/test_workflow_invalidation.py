from toporetarget.workflows.cache import signature


def test_dependency_or_profile_change_invalidates_signature() -> None:
    base = signature(
        "node",
        implementation_version="v1",
        inputs={"source": "a"},
        configs={"profile": "one"},
        parameters={"window": [0, 60]},
    )
    assert base != signature(
        "node",
        implementation_version="v1",
        inputs={"source": "b"},
        configs={"profile": "one"},
        parameters={"window": [0, 60]},
    )
    assert base != signature(
        "node",
        implementation_version="v2",
        inputs={"source": "a"},
        configs={"profile": "one"},
        parameters={"window": [0, 60]},
    )
