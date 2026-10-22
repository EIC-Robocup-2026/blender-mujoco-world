"""Check the add-on ZIP that package_addon.py built.

Run:
  python tests/check_package.py ZIP

The archive must hold exactly package_addon.package_files() (the add-on's
modules plus README.md and LICENSE), sorted, each entry with the build's
fixed timestamp (SOURCE_DATE_EPOCH when set), Unix regular-file permissions
rw-r--r--, deflate compression and the bytes of its source file. In a git
checkout, the packaged sources must be exactly the tracked add-on modules
plus README.md and LICENSE, so an untracked scratch module never ships.
"""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import package_addon  # noqa: E402

archive_path = Path(sys.argv[1])
expected = package_addon.package_files()
stamp = package_addon.entry_time()
with zipfile.ZipFile(archive_path) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    assert names == sorted(names), f"entries are not sorted: {names}"
    assert names == [name for name, _path in expected], (names, [name for name, _path in expected])
    for info, (name, source) in zip(infos, expected):
        assert info.date_time == stamp, (name, info.date_time, stamp)
        assert info.external_attr == 0o100644 << 16, (name, oct(info.external_attr))
        assert info.create_system == 3, (name, info.create_system)
        assert info.compress_type == zipfile.ZIP_DEFLATED, (name, info.compress_type)
        assert archive.read(info) == source.read_bytes(), f"{name} differs from {source}"

tracked = None
try:
    listing = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", package_addon.PACKAGE,
                              *package_addon.EXTRA_FILES], check=True, capture_output=True).stdout
    tracked = sorted(name for name in listing.decode("utf-8").split("\0")
                     if name and (name.endswith(".py") or name in package_addon.EXTRA_FILES))
except (OSError, subprocess.CalledProcessError):
    pass  # Not a git checkout: the archive checks above still apply.
if tracked is not None:
    sources = sorted(path.relative_to(ROOT).as_posix() for _name, path in expected)
    assert sources == tracked, {"untracked": sorted(set(sources) - set(tracked)),
                                "missing": sorted(set(tracked) - set(sources))}
print("PACKAGE_OK", json.dumps({"entries": len(names), "date_time": list(stamp), "git_checked": tracked is not None}))
