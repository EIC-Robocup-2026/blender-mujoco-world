"""Compile the worlds written by test_adopt_blender.py and check their physics.

Run after the Blender fixture:
  python tests/validate_adopt.py OUTPUT_DIR
"""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

if len(sys.argv) != 2:
    raise SystemExit("usage: validate_adopt.py OUTPUT_DIR")
OUT = Path(sys.argv[1]).resolve()
summary = json.loads((OUT / "adopt_summary.json").read_text())
assert not summary["failures"], f"the Blender fixture failed: {sorted(summary['failures'])}"
results = {}


def load(path):
    return mujoco.MjModel.from_xml_path(str(path))


def body_physics(model):
    """Mass and principal inertia of every named body."""
    return {model.body(i).name: (float(model.body_mass[i]), model.body_inertia[i].copy()) for i in range(1, model.nbody)}


def same_physics(source, export, names):
    for name in names:
        (mass_a, inertia_a), (mass_b, inertia_b) = source[name], export[name]
        assert np.isclose(mass_a, mass_b, rtol=1e-6, atol=1e-9), (name, mass_a, mass_b)
        assert np.allclose(inertia_a, inertia_b, rtol=1e-5, atol=1e-9), (name, inertia_a, inertia_b)


def walkie_loader_masks(model):
    """Masks robot_bringup.world_loader assigns, from the compiled model.

    The floor plane is 1/32, geoms whose contype and conaffinity resolve to 0
    stay 0/0, geoms of a body that moves with a joint (its own or an
    ancestor's) are objects (32/121) and the rest furniture (64/56).
    """
    def moving(body):
        while body > 0:
            if model.body_jntnum[body]:
                return True
            body = model.body_parentid[body]
        return False

    expected = {}
    for geom in range(model.ngeom):
        if model.geom(geom).name == "floor":
            expected[geom] = (1, 32)
        elif model.geom_contype[geom] == model.geom_conaffinity[geom] == 0:
            expected[geom] = (0, 0)
        else:
            expected[geom] = (32, 121) if moving(model.geom_bodyid[geom]) else (64, 56)
    return expected


# Every exported world compiles and simulates.
compiled = {}
for path in sorted(OUT.rglob("scene.xml")):
    model = load(path)
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all(), path
    compiled[str(path.relative_to(OUT))] = {"nbody": model.nbody, "nq": model.nq}
results["compiled"] = len(compiled)

# Regenerated meshes keep the source inertia: an untouched export has the
# source world's body masses and inertias.
files = summary["mesh_inertia"]
source, export = load(files["source"]), load(files["export"])
source_bodies, export_bodies = body_physics(source), body_physics(export)
same_physics(source_bodies, export_bodies, source_bodies)
results["mesh_inertia"] = {"bin_mass": source_bodies["Bin"][0]}

# Copies of adopted bodies are real bodies with their own joints.
files = summary["duplicates"]
source, export = load(Path(files["export"]).parent / "source" / "scene.xml"), load(files["export"])
assert export.nq == source.nq + 7, (export.nq, source.nq)
assert np.isclose(export.body("Crate.001").mass[0], export.body("Crate").mass[0], rtol=1e-9)
assert export.jnt_type[export.joint("Crate.001_free").id] == mujoco.mjtJoint.mjJNT_FREE
assert export.body_jntnum[export.body("Crate").id] == 1
results["duplicates"] = {"nq": export.nq}

# Weight applies to adopted bodies made movable and to grouped adopted parts;
# every other body keeps its source physics.
files = summary["weights"]
source, export = load(files["source"]), load(files["export"])
assert np.isclose(export.body("Stool").mass[0], 2.0, rtol=1e-9), export.body("Stool").mass
assert np.isclose(export.body(files["group"]).mass[0], 1.5, rtol=1e-9), export.body(files["group"]).mass
for name in ("Stool", files["group"]):
    assert export.jnt_type[export.body_jntadr[export.body(name).id]] == mujoco.mjtJoint.mjJNT_FREE, name
same_physics(body_physics(source), body_physics(export), ("Crate", "Bin", "Door"))
results["weights"] = {"stool": 2.0, "group": 1.5}

# With the Walkie preset, every geom already has the mask world_loader assigns
# in the merged world, doors and bodies nested in moving bodies included.
model = load(summary["door_masks"]["walkie"])
for geom, mask in walkie_loader_masks(model).items():
    assert (int(model.geom_contype[geom]), int(model.geom_conaffinity[geom])) == mask, (model.geom(geom).name, mask)
results["door_masks"] = {"geoms": model.ngeom}

# Worlds sharing one asset folder both compile with their own files.
for key in ("first", "second"):
    load(summary["shared_folder_safe_names"][key])

print("ADOPT_PHYSICS_OK", json.dumps(results))
