#!/usr/bin/env python3
"""Build an installable ZIP from the add-on's single source directory."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "mujoco_world.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted((ROOT / "mujoco_world").glob("*.py")):
            archive.write(source, f"mujoco_world/{source.name}")
        for name in ("README.md", "LICENSE"):
            archive.write(ROOT / name, f"mujoco_world/{name}")
    print(f"Built {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
