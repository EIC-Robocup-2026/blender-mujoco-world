"""Compile the worlds written by build_smoke_scene.py and check them against the scene.

Run after the Blender step:
  python tests/validate_export.py OUTPUT_DIR [WALKIE_ROS_ROOT]

With WALKIE_ROS_ROOT, the smoke worlds are also merged with the Walkie robot
through a copy of robot_bringup.world_loader, the consumer of these exports:
the merged world must compile and simulate, the robot and the world keep
their masses and inertias, and the loader's collision classes equal the
masks of the Walkie preset. The walkie-ros checkout is only read.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

OUT, ARGS, _ = harness.completed_output("smoke", "OUTPUT_DIR [WALKIE_ROS_ROOT]", 1, 2)
WALKIE = Path(ARGS[0]).resolve() if ARGS else None
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


def asset_hashes(directory):
    """Content of every asset file the export wrote, by name."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((directory / "scene_assets").iterdir()) if path.is_file()}


# The export operator writes the same world as the export function.
operator_dir = OUT / "operator"
assert (operator_dir / "scene.xml").read_text() == (OUT / "scene.xml").read_text()
operator_assets, direct_assets = asset_hashes(operator_dir), asset_hashes(OUT)
operator_assets.pop(".mujoco_world_manifest.json", None)
direct_assets.pop(".mujoco_world_manifest.json", None)
assert operator_assets == direct_assets, sorted(set(operator_assets.items()) ^ set(direct_assets.items()))
# Without the floor plane nothing is named floor; a raised plane keeps its height.
no_plane = mujoco.MjModel.from_xml_path(str(OUT / "no_plane" / "scene.xml"))
assert sum(no_plane.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 0
assert "floor" not in [no_plane.geom(i).name for i in range(no_plane.ngeom)]
assert no_plane.nq == model.nq
raised = mujoco.MjModel.from_xml_path(str(OUT / "raised_plane" / "scene.xml"))
assert sum(raised.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
assert np.isclose(raised.geom("floor").pos[2], 0.05, atol=1e-9), raised.geom("floor").pos
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


def geom_masks(world):
    return {world.geom(i).name: (int(world.geom_contype[i]), int(world.geom_conaffinity[i]))
            for i in range(world.ngeom)}


def body_physics(world, names):
    return {name: (float(world.body(name).mass[0]), world.body(name).inertia.copy()) for name in names}


def merge_with_walkie(scene_name, loader):
    """Merge one smoke world with the Walkie robot and check the merged model."""
    robot_path = WALKIE / "description" / "mujoco" / "walkie.xml"
    wheel_scene = WALKIE / "description" / "mujoco" / "scene.xml"
    scene_path = OUT / scene_name / "scene.xml" if scene_name else OUT / "scene.xml"
    # Spawn clear of every smoke object.
    merged_root = loader.assemble_world(robot_path, scene_path, wheel_scene, spawn_pose=(-1.6, -1.0, 0.0))
    merged_path = OUT / "walkie_loader" / f"{scene_name or 'default'}_merged.xml"
    merged_path.write_bytes(ET.tostring(merged_root, encoding="utf-8", xml_declaration=True))
    merged, robot, world = (mujoco.MjModel.from_xml_path(str(path)) for path in (merged_path, robot_path, scene_path))
    assert merged.nq == robot.nq + world.nq, (merged.nq, robot.nq, world.nq)
    assert sum(merged.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
    assert merged.npair == world.npair + 4, merged.npair  # The four wheel-floor pairs.
    # The robot is listed first, so its bodies keep their indices and physics.
    # The inertial offset of a massless body (base_footprint) follows the
    # spawn pose, so it is compared for bodies with mass only.
    massive = robot.body_mass > 0
    for field, rows in (("body_mass", slice(None)), ("body_inertia", slice(None)), ("body_ipos", massive)):
        assert np.allclose(getattr(merged, field)[:robot.nbody][rows], getattr(robot, field)[rows],
                           rtol=1e-9, atol=1e-12), field
    # The world keeps its masses and inertias under the robot's compiler settings.
    names = [world.body(i).name for i in range(1, world.nbody)]
    for name, (mass, inertia) in body_physics(world, names).items():
        merged_mass, merged_inertia = body_physics(merged, [name])[name]
        assert np.isclose(mass, merged_mass, rtol=1e-9, atol=1e-12), (name, mass, merged_mass)
        assert np.allclose(inertia, merged_inertia, rtol=1e-7, atol=1e-12), (name, inertia, merged_inertia)
    data = mujoco.MjData(merged)
    mujoco.mj_resetDataKeyframe(merged, data, merged.key("home").id)
    for _ in range(1000):
        mujoco.mj_step(merged, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert all(data.warning[i].number == 0 for i in range(len(data.warning))), \
        [int(data.warning[i].number) for i in range(len(data.warning))]
    base = data.xpos[merged.body("base_footprint").id]
    assert np.allclose(base[:2], [-1.6, -1.0], atol=0.05) and base[2] > -0.02, base
    for obj in report["object_details"]:
        if obj["role"] == "FREE" and obj["body"] in names:
            assert data.xpos[merged.body(obj["body"]).id, 2] > -0.02, obj["body"]
    world_masks = geom_masks(world)
    return {name: mask for name, mask in geom_masks(merged).items() if name in world_masks}


consumer = {"walkie_ros": str(WALKIE) if WALKIE else None}
loader, skipped = (None, "no WALKIE_ROS_ROOT given") if WALKIE is None else \
    harness.load_world_loader(WALKIE, OUT / "walkie_loader")
if loader is None:
    print("WALKIE_CONSUMER_PARITY_SKIPPED", skipped)
    consumer["skipped"] = skipped
else:
    preset = geom_masks(walkie)
    for scene_name in ("walkie_masks", ""):
        merged_masks = merge_with_walkie(scene_name, loader)
        # The loader's collision classes are exactly the masks of the Walkie
        # preset, so the standalone preset world collides like the merged one.
        assert merged_masks == preset, sorted(set(merged_masks.items()) ^ set(preset.items()))
        consumer[scene_name or "default"] = "pass"

result = {"status": "pass", "blender": report["blender_version"], "mujoco": mujoco.__version__,
          "objects": report["objects"], "geoms": model.ngeom, "nq": model.nq, "nv": model.nv,
          "triangles": report["triangles"], "maximum_geometry_error_m": maximum_error,
          "door_open_degrees": math.degrees(opened), "door_closed_degrees": math.degrees(closed),
          "physics_seconds": 15, "portable_relative_paths": True, "selection_only": True,
          "unsaved_image_pixels": "pass", "visual_floor_has_no_collision": True,
          "hdr_bake_agx_display_conversion": "pass", "hdr_source_buffer_unchanged": True,
          "walkie_masks_free_objects_supported_by_plane": True,
          "collision_helper_has_contacts_without_visible_mesh_or_texture": True,
          "nonuniform_mirrored_parent_shear_transforms": "pass", "operator_export_identical": True,
          "plane_options": "pass", "walkie_consumer_parity": consumer}
(OUT / "validation_report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
