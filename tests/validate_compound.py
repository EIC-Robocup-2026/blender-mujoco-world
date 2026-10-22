"""MuJoCo checks of the test_compound_blender.py output; no GUI needed.

Run:
  python tests/validate_compound.py OUTPUT_DIR
"""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

path, _, marker = harness.completed_output("compound")
model = mujoco.MjModel.from_xml_path(str(path / "fresh.xml"))
assert model.nq == 8 and model.nv == 7
cup = model.body("Hollow_cup")
assert abs(cup.mass[0] - 0.4) < 1e-6
assert cup.geomnum[0] == 14  # thirteen helpers + a noncolliding visual
assert model.body("Nested_door").parentid[0] == model.body("Assembly").id
assert abs(model.qpos0[-1] - math.radians(-85)) < 1e-6
assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
# A small sphere must fall through the ring to the bottom of the hollow cup.
root = ET.parse(path / "fresh.xml").getroot()
body = ET.SubElement(root.find("worldbody"), "body", name="Cavity_probe", pos="0 0 .5")
ET.SubElement(body, "freejoint", name="probe_free")
ET.SubElement(body, "geom", name="probe", type="sphere", size=".03", mass=".02", group="3")
probe_path = path / "cavity_probe.xml"
ET.ElementTree(root).write(probe_path)
probe_model = mujoco.MjModel.from_xml_path(str(probe_path))
data = mujoco.MjData(probe_model)
for _ in range(2000):
    mujoco.mj_step(probe_model, data)
relative_z = data.xpos[probe_model.body("Cavity_probe").id, 2] - data.xpos[probe_model.body("Hollow_cup").id, 2]
assert 0.065 < relative_z < 0.095, relative_z
assert all(data.warning[i].number == 0 for i in range(len(data.warning)))
grouped = mujoco.MjModel.from_xml_path(str(path / "grouped.xml"))
assert grouped.nq == 15 and abs(grouped.body("Grouped_prop").mass[0] - .5) < 1e-6
if marker["details"]["real_project"]:
    # The existing project's exports use the shared folder inside the output.
    for name in ("adopted.xml", "edited.xml"):
        compiler = ET.parse(path / name).getroot().find("compiler")
        for attribute in ("meshdir", "texturedir"):
            if compiler is not None and compiler.get(attribute):
                assert (path / compiler.get(attribute)).resolve() == path / "shared_assets", (name, compiler.attrib)
    edits = json.loads((path / "edits.json").read_text())
    original = mujoco.MjModel.from_xml_path(str(path / "adopted.xml"))
    edited = mujoco.MjModel.from_xml_path(str(path / "edited.xml"))
    assert edited.nq == original.nq + 8
    assert abs(edited.body(edits["body"]).mass[0] - edits["mass"]) < 1e-6
    assert abs(edited.body(edits["body"]).pos[0] - original.body(edits["body"]).pos[0] - .123) < 2e-6
    door = edited.joint(edits["door"] + "_hinge")
    assert np.allclose(door.range, np.radians([-100, 0]), atol=1e-6)
    assert abs(edited.qpos0[door.qposadr][0] - math.radians(-45)) < 1e-6
    assert abs(edited.dof_damping[door.dofadr][0] - .75) < 1e-6
    new_hinge = edited.joint("New_group_door_hinge")
    assert abs(edited.qpos0[new_hinge.qposadr][0] - math.radians(-30)) < 1e-6
    assert abs(edited.body("New_group_door").mass[0] - 2.5) < 1e-6
    geom = edited.geom(edits["door_geom"])
    color = edited.mat_rgba[geom.matid[0]]
    assert color[1] > color[0] * 3 and color[1] > color[2], color
print("COMPOUND_PHYSICS_OK", json.dumps({"cup_mass": cup.mass[0], "cavity_probe_relative_z": float(relative_z), "fresh_nq": model.nq}))
