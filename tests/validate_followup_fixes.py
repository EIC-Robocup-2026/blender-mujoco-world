"""Compile the worlds written by test_followup_fixes_blender.py and check their physics.

Run after the Blender fixture:
  python tests/validate_followup_fixes.py OUTPUT_DIR
"""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

OUT, _, _ = harness.completed_output("followup")
summary = json.loads((OUT / "followup.json").read_text())
assert not summary["failures"], f"the Blender fixture failed: {sorted(summary['failures'])}"
models = {name: mujoco.MjModel.from_xml_path(path) for name, path in summary["worlds"].items()}
results = {"compiled": len(models)}


def placed(model):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return data


# The stretched capsule keeps its caps: radius 0.05, total length 1.0 m.
rod = models["capsule"].geom("Rod")
assert np.allclose(rod.size[:2], [0.05, 0.45]), rod.size
results["capsule"] = rod.size[:2].tolist()

# The untyped joint is a hinge with the edited limits; the slide joints keep theirs.
joints = models["joints"]
door, drawer, tray = joints.joint("Door_joint"), joints.joint("Drawer_slide"), joints.joint("Tray_slide")
assert door.type[0] == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(door.range, np.radians([-45, 0])), (door.type, door.range)
assert np.isclose(joints.dof_damping[door.dofadr[0]], 0.5)
assert drawer.type[0] == mujoco.mjtJoint.mjJNT_SLIDE and np.allclose(drawer.range, [0, 0.4]), drawer.range
assert tray.type[0] == mujoco.mjtJoint.mjJNT_SLIDE and np.allclose(tray.range, [0, 0.3]), tray.range
results["joints"] = {"door_range": door.range.tolist()}

# A class-typed mesh geom has a mesh again; the class-typed box was resized.
classes = models["classes"]
assert classes.geom("Shelf_shell").type[0] == mujoco.mjtGeom.mjGEOM_MESH and classes.nmesh == 1
assert np.allclose(classes.geom("Shelf_slab").size, [0.4, 0.1, 0.05])

# Geoms and bodies moved out of the childclass body keep its contact parameters.
moved = models["moved"]
for name in ("Shelf_box", "Shelf_lip", "Shelf_drawer_box"):
    item = moved.geom(name)
    assert item.condim[0] == 4 and np.allclose(item.friction, [0.9, 0.01, 0.001]), (name, item.condim, item.friction)
    assert item.bodyid[0] == 0 or moved.body(item.bodyid[0]).name == "Shelf_drawer", name
results["moved"] = {"condim": 4}

# Every mesh name is distinct, including Thai names.
names = [models["mesh_names"].mesh(i).name for i in range(models["mesh_names"].nmesh)]
assert len(set(names)) == 4, names

# Untouched offset primitives sit where the source put them.
data = placed(models["offsets_untouched"])
assert np.allclose(data.geom("Table_top").xpos, [0, 0, 0.7], atol=1e-9), data.geom("Table_top").xpos
assert np.allclose(data.geom("Cart_box").xpos, [2, 0, 0.3], atol=1e-7), data.geom("Cart_box").xpos
data = placed(models["offsets_edited"])
assert np.allclose(data.geom("Table_top").xpos, [0.25, 0, 0.72], atol=1e-6), data.geom("Table_top").xpos
assert np.allclose(data.geom("Cart_box").xpos, [2, 0.5, 0.3], atol=1e-6), data.geom("Cart_box").xpos

# The grouped body's centre of mass follows collider volume (equal shares put it at z=0.53).
volume = summary["volume_mass"]
model = models["volume"]
data = placed(model)
body = model.body(volume["body"])
assert np.isclose(body.mass[0], 3.0, rtol=1e-6), body.mass
com = data.xipos[body.id]
assert abs(com[2] - volume["expected_com_z"]) < 1e-3, (com, volume["expected_com_z"])
results["volume"] = {"com_z": float(com[2]), "expected": volume["expected_com_z"]}

# The asset directory worlds load the same textures and height fields as their sources.
for name in ("assetdir", "strippath"):
    source, export = models[name + "_source"], models[name]
    assert (export.ntex, export.nhfield) == (source.ntex, source.nhfield), (name, export.ntex, source.ntex)
    assert np.array_equal(export.tex_data, source.tex_data), name
results["asset_dirs"] = {"textures": int(models["assetdir"].ntex), "hfields": int(models["assetdir"].nhfield)}

# Opted-in outside textures and a variant's shared library load as textures.
for name in ("outside_allowed", "outside_shared_library"):
    assert models[name].ntex == 1, (name, models[name].ntex)
results["outside_assets"] = {name: int(models[name].ntex) for name in ("outside_allowed", "outside_shared_library")}

# Doors oriented by euler and xyaxes, with a new starting angle: MuJoCo reads
# the single quaternion, the source orientation turned by the angle change.
doors = models["door_orientations"]
for name, wanted in summary["door_orientations"].items():
    quat = doors.body(name).quat
    assert np.isclose(abs(np.dot(quat, wanted)), 1, atol=1e-6), (name, quat, wanted)
results["door_orientations"] = sorted(summary["door_orientations"])

# The material layer's texture loads; both height fields and the three used textures load.
assert models["layered"].ntex == 1 and models["layered"].nmat == 1, (models["layered"].ntex, models["layered"].nmat)
for mode in ("per_xml", "shared"):
    model = models[f"copy_names_{mode}"]
    assert (model.ntex, model.nhfield) == (3, 2), (mode, model.ntex, model.nhfield)
results["copy_names"] = {"textures": 3, "hfields": 2}

print("FOLLOWUP_FIXES_PHYSICS_OK", json.dumps(results))
