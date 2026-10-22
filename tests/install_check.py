"""Install the packaged add-on ZIP into a sandbox and check that it works.

Run with a sandboxed user resource directory:
  BLENDER_USER_RESOURCES=SANDBOX blender --background --factory-startup \
      --python-exit-code 1 --python tests/install_check.py -- OUTPUT_DIR ZIP

The check refuses to run unless Blender's user script directory is inside
SANDBOX, so it never installs into the real Blender configuration. It
installs ZIP like "Install from Disk", enables the add-on, checks its
operators, panel and File > Export entry, exports a world with the export
operator, then disables, re-enables and disables it again.
"""
import json
import os
from pathlib import Path
import sys

import addon_utils
import bpy

sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(args) != 2:
    raise SystemExit("usage: BLENDER_USER_RESOURCES=SANDBOX blender --background --factory-startup "
                     "--python-exit-code 1 --python tests/install_check.py -- OUTPUT_DIR ZIP")
sandbox = os.environ.get("BLENDER_USER_RESOURCES")
scripts = Path(bpy.utils.user_resource("SCRIPTS")).resolve()
if not sandbox or not scripts.is_relative_to(Path(sandbox).resolve()):
    raise SystemExit(f"Refusing to install: user scripts {scripts} are not inside BLENDER_USER_RESOURCES={sandbox!r}")
out = Path(args[0]).resolve()
zip_path = Path(args[1]).resolve()
harness.prepare_output(out, "install")
sys.dont_write_bytecode = True
assert "mujoco_world" not in sys.modules, "the add-on was imported before it was installed"

OPERATORS = ("set_role", "hinge_at_cursor", "group_parts", "adopt_mjcf", "bake_lighting", "restore_sources",
             "walkie_contacts", "export_world")
result = {"blender": bpy.app.version_string, "zip": str(zip_path), "user_scripts": str(scripts)}
result["install"] = sorted(bpy.ops.preferences.addon_install(filepath=str(zip_path), overwrite=True))
installed = scripts / "addons" / "mujoco_world"
assert installed.is_dir(), f"{installed} was not created"
result["installed_files"] = sorted(path.name for path in installed.iterdir())
result["enable"] = sorted(bpy.ops.preferences.addon_enable(module="mujoco_world"))
import mujoco_world  # noqa: E402

assert Path(mujoco_world.__file__).resolve().parent == installed.resolve(), mujoco_world.__file__
assert addon_utils.check("mujoco_world")[1], "the add-on is not enabled"


def registered(name):
    try:
        getattr(bpy.ops.mjworld, name).get_rna_type()
    except KeyError:
        return False
    return True


missing = [name for name in OPERATORS if not registered(name)]
assert not missing, f"operators not registered: {missing}"
assert hasattr(bpy.types, "MJWORLD_PT_main"), "the panel is not registered"
assert hasattr(bpy.types.Scene, "mujoco_world") and hasattr(bpy.types.Object, "mujoco_world")
export_entries = [getattr(function, "__name__", "") for function in bpy.types.TOPBAR_MT_file_export._dyn_ui_initialize()]
assert "export_menu" in export_entries, export_entries

bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0, 0, 0.5))
assert bpy.ops.mjworld.set_role(role="FREE") == {"FINISHED"}
assert bpy.ops.mjworld.export_world(filepath=str(out / "scene.xml")) == {"FINISHED"}
assert (out / "scene.xml").is_file() and (out / "export_report.json").is_file()

result["disable"] = sorted(bpy.ops.preferences.addon_disable(module="mujoco_world"))
assert not hasattr(bpy.types.Object, "mujoco_world"), "properties remain after disabling"
assert not hasattr(bpy.types, "MJWORLD_PT_main"), "the panel remains after disabling"
assert not any(registered(name) for name in OPERATORS), "operators remain after disabling"
result["reenable"] = sorted(bpy.ops.preferences.addon_enable(module="mujoco_world"))
assert hasattr(bpy.types.Scene, "mujoco_world"), "properties are missing after enabling again"
result["final_disable"] = sorted(bpy.ops.preferences.addon_disable(module="mujoco_world"))
(out / "install_check.json").write_text(json.dumps(result, indent=2) + "\n")
harness.complete(out, "install", installed_files=result["installed_files"])
print("INSTALL_CHECK_OK", json.dumps(result))
