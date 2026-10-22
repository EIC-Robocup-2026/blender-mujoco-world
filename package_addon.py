#!/usr/bin/env python3
"""Build the installable add-on ZIP: dist/mujoco_world-<VERSION>.zip.

Usage:
  python3 package_addon.py [--output DIRECTORY]

The archive holds every file of mujoco_world/ (without bytecode caches or
hidden files) plus README.md and LICENSE, all under mujoco_world/, which is
the layout Blender's "Install from Disk" expects for a legacy add-on.

VERSION is read from bl_info in mujoco_world/__init__.py, so the add-on's
version has a single source. The build is reproducible: entries are sorted
and carry a fixed timestamp and fixed permissions, so the same sources give
a byte-identical ZIP. Set SOURCE_DATE_EPOCH to stamp the entries with that
time instead of 1980-01-01.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGE = "mujoco_world"
EXTRA_FILES = ("README.md", "LICENSE")
EXCLUDED_DIRECTORIES = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EARLIEST_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def addon_version():
    """bl_info["version"] of the add-on, read without importing bpy."""
    tree = ast.parse((ROOT / PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "bl_info" for t in node.targets):
            return ".".join(str(part) for part in ast.literal_eval(node.value)["version"])
    raise SystemExit(f"{PACKAGE}/__init__.py has no bl_info")


VERSION = addon_version()


def entry_time():
    """Timestamp of every entry: SOURCE_DATE_EPOCH when set, otherwise the earliest ZIP time."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if not epoch:
        return EARLIEST_ZIP_TIME
    return max(EARLIEST_ZIP_TIME, time.gmtime(int(epoch))[:6])


def package_files():
    """(archive name, source path) of every packaged file, sorted by archive name."""
    files = []
    for path in (ROOT / PACKAGE).rglob("*"):
        relative = path.relative_to(ROOT / PACKAGE)
        if not path.is_file() or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if any(part in EXCLUDED_DIRECTORIES or part.startswith(".") for part in relative.parts):
            continue
        files.append((f"{PACKAGE}/{relative.as_posix()}", path))
    for name in EXTRA_FILES:
        files.append((f"{PACKAGE}/{name}", ROOT / name))
    names = [name for name, _ in files]
    if len(names) != len(set(names)):
        raise SystemExit(f"Duplicate archive entries: {sorted(n for n in names if names.count(n) > 1)}")
    return sorted(files)


def build(output_directory):
    files = package_files()
    for name, path in files:
        if path.suffix == ".py":
            # Refuse to package a file Blender could not even import.
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"{PACKAGE}-{VERSION}.zip"
    stamp = entry_time()
    temporary = output.with_name(output.name + ".partial")
    with zipfile.ZipFile(temporary, "w") as archive:
        for name, path in files:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, so the permission bits below are used.
            info.external_attr = 0o100644 << 16  # Regular file, rw-r--r--.
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary.replace(output)
    return output, files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "dist",
                        help="directory for the ZIP (default: dist/ in the repository)")
    arguments = parser.parse_args()
    output, files = build(arguments.output.resolve())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Built {output} ({len(files)} files, {output.stat().st_size / 1024:.1f} KiB, sha256 {digest})")


if __name__ == "__main__":
    main()
