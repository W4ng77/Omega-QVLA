"""Fetch exact upstream algorithms/kernels without replacing the GR00T stack."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    for name, source in json.loads((Path(__file__).with_name("sources.json")).read_text()).items():
        directory = ROOT / "third_party" / name
        if not directory.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", source["url"], str(directory)], check=True)
            subprocess.run(["git", "-C", str(directory), "checkout", "--detach", source["commit"]], check=True)
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != source["commit"]:
            raise RuntimeError(f"Wrong dependency revision: {name} {actual}")
        for subpath, expected in source.get("header_submodules", {}).items():
            subprocess.run(
                ["git", "-C", str(directory), "submodule", "update", "--init", "--depth", "1", subpath],
                check=True,
            )
            subcommit = subprocess.check_output(
                ["git", "-C", str(directory / subpath), "rev-parse", "HEAD"], text=True
            ).strip()
            assert subcommit == expected
        if "cutlass_submodule" in source:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(directory),
                    "submodule",
                    "update",
                    "--init",
                    "--depth",
                    "1",
                    "third-party/cutlass",
                ],
                check=True,
            )
            cutlass = subprocess.check_output(
                ["git", "-C", str(directory / "third-party/cutlass"), "rev-parse", "HEAD"], text=True
            ).strip()
            assert cutlass == source["cutlass_submodule"]
        print(name, actual, flush=True)


if __name__ == "__main__":
    main()
