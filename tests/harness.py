"""Shared start and finish steps of the Blender tests and their validators.

A Blender test starts with ``begin`` and ends with ``complete``::

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import harness  # noqa: E402
    OUT, ARGS = harness.begin("smoke")
    ...
    harness.complete(OUT, "smoke")

``begin`` reads ``OUTPUT_DIR [ARGS...]`` after Blender's ``--``, clears the
output directory, and imports and registers the add-on from this checkout,
never an installed copy. DeprecationWarnings raised by the add-on become
errors, so an API that Blender is about to remove fails the test.
``complete`` writes ``run_complete.json`` as the last file of a successful
run.

A validator reads its output directory with ``completed_output``, which
refuses a directory without that marker, so output left behind by an earlier
run never passes as the result of a failed one. When ``MJWORLD_RUN_TOKEN`` is
set (``run_all.sh`` sets it), the marker must also carry the same token.

This module imports ``bpy`` only inside ``begin`` and ``complete``, so the
validators can import it outside Blender.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
MARKER = "run_complete.json"
SENTINEL = ".mujoco_world_test_output"
TOKEN_VARIABLE = "MJWORLD_RUN_TOKEN"


def _blender_arguments():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def prepare_output(out, test):
    """Empty ``out``, but only when it is empty or an earlier test output."""
    if out.exists():
        if not out.is_dir():
            raise SystemExit(f"{out} is not a directory")
        if any(out.iterdir()) and not (out / SENTINEL).is_file():
            raise SystemExit(f"Refusing to clear {out}: it is not empty and was not written by a test. "
                             "Pass a new or empty directory.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / SENTINEL).write_text(test + "\n", encoding="utf-8")


def begin(test, usage="OUTPUT_DIR", minimum=1, maximum=1):
    """Prepare a Blender test run; return the output directory and the remaining arguments."""
    script = Path(sys.argv[sys.argv.index("--python") + 1]).name if "--python" in sys.argv else "TEST.py"
    args = _blender_arguments()
    if not minimum <= len(args) <= maximum:
        raise SystemExit(f"usage: blender --background --factory-startup --python-exit-code 1 "
                         f"--python tests/{script} -- {usage}")
    out = Path(args[0]).resolve()
    prepare_output(out, test)
    warnings.filterwarnings("error", category=DeprecationWarning, module=r"mujoco_world")
    # Keep Blender's bytecode out of the source tree.
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import mujoco_world

    location = Path(mujoco_world.__file__).resolve().parent
    if location != ROOT / "mujoco_world":
        raise RuntimeError(f"Imported the add-on from {location}, not from {ROOT / 'mujoco_world'}; "
                           "run Blender with --factory-startup")
    mujoco_world.register()
    return out, args[1:]


def complete(out, test, **details):
    """Mark ``out`` as the complete output of a successful run of ``test``."""
    import bpy

    marker = {"test": test, "blender": bpy.app.version_string, "token": os.environ.get(TOKEN_VARIABLE, ""),
              "details": details}
    (out / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def completed_output(test, usage="OUTPUT_DIR", minimum=1, maximum=1):
    """Output directory and remaining arguments of a validator, after checking the run marker."""
    script = Path(sys.argv[0]).name
    args = sys.argv[1:]
    if not minimum <= len(args) <= maximum:
        raise SystemExit(f"usage: python tests/{script} {usage}")
    out = Path(args[0]).resolve()
    path = out / MARKER
    if not path.is_file():
        raise SystemExit(f"{out} has no {MARKER}: the Blender step '{test}' did not finish there. "
                         "Run it again before validating.")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("test") != test:
        raise SystemExit(f"{out} holds the output of '{marker.get('test')}', not '{test}'")
    token = os.environ.get(TOKEN_VARIABLE)
    if token and marker.get("token") != token:
        raise SystemExit(f"{out} was written by another run (token {marker.get('token')!r}, expected {token!r})")
    return out, args[1:], marker


def load_world_loader(walkie_root, scratch):
    """Import walkie-ros ``robot_bringup.world_loader`` from a copy in ``scratch``.

    The copy keeps every import side effect (bytecode, caches) out of the
    walkie-ros checkout, which is only read. Returns ``(module, None)``, or
    ``(None, reason)`` when the loader is missing or its ``assemble_world``
    no longer has the expected parameters, for example while it is being
    changed.
    """
    source = Path(walkie_root) / "bringup" / "robot_bringup" / "world_loader.py"
    if not source.is_file():
        return None, f"{source} does not exist"
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    copy = scratch / "walkie_world_loader.py"
    shutil.copyfile(source, copy)
    spec = importlib.util.spec_from_file_location("walkie_world_loader", copy)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure means the loader is unusable here
        return None, f"importing {source} failed: {type(exc).__name__}: {exc}"
    assemble = getattr(module, "assemble_world", None)
    if assemble is None:
        return None, f"{source} has no assemble_world"
    parameters = list(inspect.signature(assemble).parameters)
    if parameters[:3] != ["robot_path", "scene_path", "wheel_scene_path"]:
        return None, f"assemble_world{inspect.signature(assemble)} has unexpected parameters"
    return module, None
