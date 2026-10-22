"""Compile the worlds written by test_export_fixes_blender.py and check their physics.

Run after the Blender fixture:
  python tests/validate_export_fixes.py OUTPUT_DIR [WALKIE_ROS_ROOT]

With WALKIE_ROS_ROOT, the door world is also merged with the Walkie robot through
robot_bringup.world_loader (read only) to confirm that the merged world compiles
and that the robot keeps its own masses and inertias.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("usage: validate_export_fixes.py OUTPUT_DIR [WALKIE_ROS_ROOT]")
OUT = Path(sys.argv[1]).resolve()
WALKIE = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
summary = json.loads((OUT / "export_fixes.json").read_text())
results = {}


def load(path):
    return mujoco.MjModel.from_xml_path(str(path))


def z_axis(quat):
    w, x, y, z = quat
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def same_rotation(a, b, tolerance=1e-6):
    return min(np.abs(np.asarray(a) - np.asarray(b)).max(), np.abs(np.asarray(a) + np.asarray(b)).max()) <= tolerance


def masks(model, geom):
    return int(model.geom_contype[geom]), int(model.geom_conaffinity[geom])


# 1. Door world: radian angles, unchanged masses under MuJoCo's default inertia
# settings, and hex masks normalized in grouped bodies.
scene = Path(summary["walkie_door"])
report = json.loads((scene.parent / "export_report.json").read_text())
body_of = {detail["source"]: detail["body"] for detail in report["object_details"]}
model = load(scene)
hinge = model.joint("Cabinet_door_hinge").id
assert np.allclose(model.jnt_range[hinge], np.radians([-100, 0]), atol=1e-8), model.jnt_range[hinge]
assert math.isclose(model.qpos0[model.jnt_qposadr[hinge]], math.radians(-30), abs_tol=1e-8)
assert math.isclose(model.dof_damping[model.jnt_dofadr[hinge]], 0.3, rel_tol=1e-12)
for source, mass in (("Crate", 2.5), ("Cabinet_door_panel", 6.0), ("Tote_bottom", 1.5)):
    body = model.body(body_of[source]).id
    assert math.isclose(model.body_mass[body], mass, rel_tol=1e-9), (source, model.body_mass[body])
for name, expected in ((body_of["Crate"], (32, 121)), (body_of["Tote_bottom"], (32, 121)),
                       (body_of["Cabinet_door_panel"], (32, 121)), (body_of["Wall"], (64, 56))):
    body = model.body(name).id
    colliding = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g]]
    assert colliding and all(masks(model, g) == expected for g in colliding), (name, [masks(model, g) for g in colliding])
assert masks(model, model.geom("floor").id) == (1, 32)
# The flags the exporter no longer writes must not have been needed: the same
# file with them restored compiles to identical inertials.
root = ET.parse(scene).getroot()
root.find("compiler").attrib.update(inertiafromgeom="true", inertiagrouprange="3 3")
legacy_path = scene.with_name("legacy_inertia.xml")
ET.ElementTree(root).write(legacy_path, encoding="utf-8", xml_declaration=True)
try:
    legacy = load(legacy_path)
finally:
    legacy_path.unlink()
for field in ("body_mass", "body_inertia", "body_ipos"):
    assert np.allclose(getattr(model, field), getattr(legacy, field), rtol=1e-9, atol=1e-12), field
results["walkie_door"] = {"nq": model.nq, "nbody": model.nbody, "mass": float(model.body_mass.sum())}

if WALKIE is not None:
    spec = importlib.util.spec_from_file_location("world_loader", WALKIE / "bringup/robot_bringup/world_loader.py")
    world_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(world_loader)
    robot_path = WALKIE / "description/mujoco/walkie.xml"
    merged_root = world_loader.assemble_world(robot_path, scene, WALKIE / "description/mujoco/scene.xml")
    merged_path = OUT / "walkie_door_merged.xml"
    merged_path.write_bytes(ET.tostring(merged_root, encoding="utf-8", xml_declaration=True))
    merged, robot = load(merged_path), load(robot_path)
    # The robot is listed first, so its bodies keep their indices.
    count = robot.nbody
    for field in ("body_mass", "body_inertia", "body_ipos"):
        assert np.allclose(getattr(merged, field)[:count], getattr(robot, field), rtol=1e-9, atol=1e-12), field
    assert merged.nq == robot.nq + model.nq, (merged.nq, robot.nq, model.nq)
    data = mujoco.MjData(merged)
    mujoco.mj_resetDataKeyframe(merged, data, merged.key("home").id)
    for _ in range(200):
        mujoco.mj_step(merged, data)
    assert np.isfinite(data.qpos).all()
    results["walkie_merge"] = {"robot_mass": float(robot.body_mass.sum()), "merged_nq": merged.nq}

# 2. 'Leave out' on a body group removed its parts.
model = load(OUT / "excluded_group" / "scene.xml")
assert [model.body(i).name for i in range(model.nbody)] == ["world", "Kept_box_000"]

# 6 and 7. A tilted flat box and an upper-case suffix both compile.
load(OUT / "tilted_box" / "scene.xml")
model = load(Path(summary["suffix"]))
assert model.jnt_type[model.joint("Bin_free").id] == mujoco.mjtJoint.mjJNT_FREE

# 8. Adopted fromto and orientation forms: an untouched export is the source
# world; an edited one moves and resizes only the edited geoms.
files = summary["fromto"]
source, untouched, edited = (load(files[key]) for key in ("source", "untouched", "edited"))
for name in [source.geom(i).name for i in range(source.ngeom)]:
    a, b = source.geom(name), untouched.geom(name)
    assert np.allclose(a.pos, b.pos, atol=1e-7) and np.allclose(a.size, b.size, atol=1e-7), name
    assert same_rotation(a.quat, b.quat, 1e-7), (name, a.quat, b.quat)
for name in [source.body(i).name for i in range(1, source.nbody)]:
    a, b = source.body(name), untouched.body(name)
    assert np.allclose(a.pos, b.pos, atol=1e-7) and same_rotation(a.quat, b.quat, 1e-7), name
leg, source_leg = edited.geom("Leg"), source.geom("Leg")
assert np.allclose(leg.pos, source_leg.pos + [0.1, 0, 0], atol=1e-6) and np.allclose(leg.size, source_leg.size, atol=1e-7)
assert same_rotation(z_axis(leg.quat), z_axis(source_leg.quat))
post, source_post = edited.geom("Post"), source.geom("Post")
assert np.allclose(post.pos, source_post.pos, atol=1e-6) and np.allclose(post.size[:2], [0.03, 0.45], atol=1e-7)
for name in ("Rail", "Beam", "Shelf_e_box", "Shelf_za_box"):
    a, b = source.geom(name), edited.geom(name)
    assert np.allclose(a.pos, b.pos, atol=1e-7) and np.allclose(a.size, b.size, atol=1e-7), name
# New doors keep degrees in the file and compile to the authored angles.
for joint, limits, start in ((files["flap_joint"], (-90, 0), -20), ("New_door_hinge", (-80, 0), -25)):
    index = edited.joint(joint).id
    assert np.allclose(edited.jnt_range[index], np.radians(limits), atol=1e-8), (joint, edited.jnt_range[index])
    assert math.isclose(edited.qpos0[edited.jnt_qposadr[index]], math.radians(start), abs_tol=1e-8), joint
door = edited.body("New_door")
assert same_rotation(door.quat, [math.cos(math.radians(-12.5)), 0, 0, math.sin(math.radians(-12.5))])
assert math.isclose(door.mass[0], 3.0, rel_tol=1e-9)
results["fromto"] = {"source_ngeom": source.ngeom, "edited_ngeom": edited.ngeom, "edited_nq": edited.nq}

# 9. The unnamed atlas geom got a mesh named after its Blender object.
model = load(Path(summary["unnamed_atlas"]))
assert [model.mesh(i).name for i in range(model.nmesh)] == ["Baked_atlas_authored_mesh"]

print("EXPORT_FIXES_PHYSICS_OK", json.dumps(results))
