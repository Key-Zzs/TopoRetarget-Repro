#!/usr/bin/env python3
"""Build the local, portable C++17 exact BVH extension without sudo."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from time import perf_counter

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean", action="store_true", help="remove only this local build product first"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "src/toporetarget/geometry/signed_distance/_compiled_sdf_cpu.cpp"
    output_dir = root / ".local/build/compiled_sdf_cpu_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python did not report an extension suffix")
    output = output_dir / f"_compiled_sdf_cpu{suffix}"
    metadata = output_dir / "build.json"
    if args.clean:
        for path in (output, metadata):
            if path.exists():
                path.unlink()
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("C++17 compiler unavailable: expected c++ on PATH")
    python_include = sysconfig.get_paths()["include"]
    command = [
        compiler,
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-pthread",
        str(source),
        f"-I{python_include}",
        f"-I{np.get_include()}",
        "-o",
        str(output),
    ]
    started = perf_counter()
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    elapsed = perf_counter() - started
    payload = {
        "status": "pass" if completed.returncode == 0 else "failed",
        "source": str(source),
        "output": str(output),
        "command": command,
        "portable": True,
        "native_flags": False,
        "elapsed_s": elapsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        return completed.returncode
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
