"""Run after build_smoke_scene.py with Python + mujoco + numpy + Pillow."""
import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "output"
report = json.loads((OUT / "export_report.json").read_text())
expected = json.loads((OUT / "expected_vertices.json").read_text())
model = mujoco.MjModel.from_xml_path(str(OUT / "scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
assert report["objects"] == 11
assert report["textures"] == 2
assert model.nq == 36 and model.nv == 31  # Five free bodies and one hinge.
assert not any("Excluded" in x["body"] or "Hidden" in x["body"] for x in report["object_details"])

maximum_error = 0.0
for obj in report["object_details"]:
    vertices = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and name.startswith(obj["body"] + "_visual_"):
            assert model.geom_contype[geom_id] == model.geom_conaffinity[geom_id] == 0
            assert model.geom_group[geom_id] == 1
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            local = model.mesh_vert[start:start + count]
            vertices.extend(local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id])
    if obj["role"] == "COLLIDER":
        assert not vertices and obj["triangles"] == 0
        geom = model.geom(obj["body"] + "_collision")
        assert int(geom.contype[0]) != 0 and int(geom.conaffinity[0]) != 0
        assert geom.rgba[3] == 0
        assert not list((OUT / "scene_assets").glob(obj["body"] + "*"))
        continue
    vertices = np.array(vertices)
    source = np.array(expected[obj["source"]])
    dist = np.linalg.norm(source[:, None, :] - vertices[None, :, :], axis=-1)
    error = float(max(dist.min(axis=0).max(), dist.min(axis=1).max()))
    assert error < 2e-6, (obj["source"], error)
    maximum_error = max(maximum_error, error)
    if obj["role"] in {"FREE", "HINGE"}:
        body_id = model.body(obj["body"]).id
        assert abs(model.body_mass[body_id] - obj["mass_kg"]) < 1e-8

floor_id = model.geom("floor").id
assert model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE
for obj in report["object_details"]:
    if "floor" in obj["source"].lower():
        assert obj["role"] == "VISUAL" and obj["collision"] is None
assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1

texture = np.asarray(Image.open(OUT / "scene_assets" / "texture_000.png"))
assert texture.shape[:2] == (32, 32)
assert np.ptp(texture[:, :, 0]) >= 250 and np.ptp(texture[:, :, 1]) >= 250
assert abs(np.mean(texture[:, :, 2]) - 102) < 1  # Preserve generated image's .4 display buffer.
hdr = np.asarray(Image.open(OUT / "scene_assets" / "texture_001.png"))
reference = np.asarray(Image.open(OUT / "hdr_agx_reference.png"))
assert np.array_equal(hdr, reference)
assert 0 < hdr.min() < 200 and 200 < hdr.max() < 255
assert len(np.unique(hdr[:, :, 0])) == 4  # Values >1 were tone-mapped, not all clipped white.
buffer_check = json.loads((OUT / "source_buffer_check.json").read_text())
assert buffer_check["hdr_max"] == 16 and buffer_check["hdr_filepath"] == ""

door = next(x for x in report["object_details"] if x["role"] == "HINGE")
joint_id = model.joint(door["body"] + "_hinge").id
dof = model.jnt_dofadr[joint_id]
qadr = model.jnt_qposadr[joint_id]
for _ in range(2500):
    data.qfrc_applied[dof] = -4
    mujoco.mj_step(model, data)
opened = float(data.qpos[qadr])
assert opened < math.radians(-100) and opened > math.radians(-112), opened
assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
for _ in range(2500):
    data.qfrc_applied[dof] = 4
    mujoco.mj_step(model, data)
closed = float(data.qpos[qadr])
assert abs(closed) < math.radians(2), closed
for _ in range(2500):
    data.qfrc_applied[dof] = 0
    mujoco.mj_step(model, data)
assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()

# Prove relative paths survive moving the world away from the source tree.
with tempfile.TemporaryDirectory(prefix="mjworld-portable-") as temp:
    destination = Path(temp) / "moved"
    shutil.copytree(OUT, destination)
    moved = mujoco.MjModel.from_xml_path(str(destination / "scene.xml"))
    assert moved.ngeom == model.ngeom
selected = mujoco.MjModel.from_xml_path(str(OUT / "selected" / "scene.xml"))
assert selected.nq == 7 and selected.nv == 6
walkie = mujoco.MjModel.from_xml_path(str(OUT / "walkie_masks" / "scene.xml"))
assert walkie.geom("floor").contype == 1 and walkie.geom("floor").conaffinity == 32
walkie_data = mujoco.MjData(walkie)
for obj in report["object_details"]:
    if obj["role"] == "VISUAL":
        continue
    geom = walkie.geom(obj["body"] + "_collision")
    assert (int(geom.contype[0]), int(geom.conaffinity[0])) == ((32, 121) if obj["role"] in {"FREE", "HINGE"} else (64, 56))


def walkie_loader_masks(world):
    """Masks robot_bringup.world_loader assigns (a copy of its classification rules).

    The floor plane is 1/32, geoms with contype and conaffinity 0 stay 0/0,
    geoms that move with a joint are objects (32/121) and the rest furniture (64/56).
    """
    expected_masks = {}
    def walk(element, moving):
        if element.tag == "body":
            moving = moving or any(child.tag in {"joint", "freejoint"} for child in element)
        if element.tag == "geom":
            if element.get("name") == "floor":
                mask = (1, 32)
            elif element.get("contype", "1") == "0" and element.get("conaffinity", "1") == "0":
                mask = (0, 0)
            else:
                mask = (32, 121) if moving else (64, 56)
            expected_masks[element.get("name")] = mask
        for child in element:
            walk(child, moving)
    walk(world.find("worldbody"), False)
    return expected_masks


# With the Walkie preset, the exported masks are the ones world_loader gives the
# merged world, so the standalone scene behaves like the merged one.
walkie_root = ET.parse(OUT / "walkie_masks" / "scene.xml").getroot()
assert not walkie_root.findall("default"), "the vendored rules ignore default classes"
for name, mask in walkie_loader_masks(walkie_root).items():
    geom = walkie.geom(name)
    assert (int(geom.contype[0]), int(geom.conaffinity[0])) == mask, (name, mask)
for _ in range(2500):
    mujoco.mj_step(walkie, walkie_data)
for obj in report["object_details"]:
    if obj["role"] == "FREE":
        assert walkie_data.xpos[walkie.body(obj["body"]).id, 2] > -0.02

result = {"status": "pass", "blender": report["blender_version"], "mujoco": mujoco.__version__,
          "objects": report["objects"], "geoms": model.ngeom, "nq": model.nq, "nv": model.nv,
          "triangles": report["triangles"], "maximum_geometry_error_m": maximum_error,
          "door_open_degrees": math.degrees(opened), "door_closed_degrees": math.degrees(closed),
          "physics_seconds": 15, "portable_relative_paths": True, "selection_only": True,
          "unsaved_image_pixels": "pass", "visual_floor_has_no_collision": True,
          "hdr_bake_agx_display_conversion": "pass", "hdr_source_buffer_unchanged": True,
          "walkie_masks_free_objects_supported_by_plane": True,
          "collision_helper_has_contacts_without_visible_mesh_or_texture": True,
          "nonuniform_mirrored_parent_shear_transforms": "pass"}
(OUT / "validation_report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
