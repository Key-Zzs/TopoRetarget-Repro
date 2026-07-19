# Contributing

Keep external data, model files, imported robot assets, and extraction caches under `.local/`.
Preserve paper-to-code provenance and add an assumption entry whenever a paper detail is unknown.
Run `ruff check .`, `ruff format --check .`, `mypy src`, `pytest -q`, the paper checker, and
`git diff --check` before proposing a change.

