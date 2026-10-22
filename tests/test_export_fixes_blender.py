"""Blender fixture for fresh-export correctness and adopted fromto primitives.

Run:
  blender --background --factory-startup --python-exit-code 1 \
      --python tests/test_export_fixes_blender.py -- OUTPUT_DIR
Then validate the written worlds with MuJoCo:
  python tests/validate_export_fixes.py OUTPUT_DIR [WALKIE_ROS_ROOT]

Every scene is generated here; nothing outside OUTPUT_DIR is written.
"""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco_world  # noqa: E402
from mujoco_world.exporter import ExportError, export_world  # noqa: E402
from mujoco_world.project import adopt_mjcf  # noqa: E402

assert Path(mujoco_world.__file__).resolve().is_relative_to(ROOT), mujoco_world.__file__
mujoco_world.register()
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not args:
    raise SystemExit("Pass an output directory after --")
OUT = Path(args[0]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
summary = {}


def fresh_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.scale_length = 1.0
    return scene


def box(name, pos, half, role="STATIC", parent=None, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos, rotation=rotation)
    obj = bpy.context.object
    obj.name, obj.scale = name, half
    obj.mujoco_world.role = role
    if parent:
        bpy.context.view_layer.update()
        world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = world
    return obj


def empty(name, pos, role="STATIC", parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = pos
    obj.mujoco_world.body = True
    obj.mujoco_world.role = role
    obj.parent = parent
    return obj


def expect_error(fragment, target):
    """The export must fail with ``fragment`` before the XML is written."""
    bpy.context.view_layer.update()
    try:
        export_world(bpy.context, str(target))
    except ExportError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        assert not target.exists(), target
        return str(exc)
    raise AssertionError(f"export should have failed with '{fragment}'")


def xml(path):
    return ET.parse(path).getroot()


# 1. Walkie-compatible fresh world: radians, MuJoCo default inertia settings,
# hex masks in grouped bodies, and float properties written as entered.
scene = fresh_scene()
settings = scene.mujoco_world
settings.plane_contype, settings.plane_conaffinity = "0b1", "0x20"
settings.object_contype, settings.object_conaffinity = "0x40", "0x38"
settings.hinge_contype, settings.hinge_conaffinity = "0x20", "0x79"
settings.movable_contype, settings.movable_conaffinity = "0x20", "0x79"
box("Decorative_floor", (0, 0, -0.01), (3, 3, 0.01), "VISUAL")
box("Wall", (0, 2.5, 0.5), (2, 0.05, 0.5))
crate = box("Crate", (1, 0, 0.3), (0.2, 0.2, 0.2), "FREE")
crate.mujoco_world.mass = 2.5
door = empty("Cabinet_door", (-1, 0, 0), "HINGE")
door.mujoco_world.mass = 6
door.mujoco_world.hinge_min, door.mujoco_world.hinge_max = -100, 0
door.mujoco_world.hinge_start = -30
door.mujoco_world.damping = 0.3
box("Cabinet_door_panel", (-0.7, 0, 0.8), (0.3, 0.02, 0.75), "STATIC", door)
tote = empty("Tote", (0.5, -1, 0.05), "FREE")
tote.mujoco_world.mass = 1.5
box("Tote_bottom", (0.5, -1, 0.03), (0.2, 0.15, 0.02), "COLLIDER", tote)
box("Tote_side", (0.5, -1.13, 0.12), (0.2, 0.02, 0.08), "COLLIDER", tote)
box("Tote_visual", (0.5, -1, 0.1), (0.2, 0.15, 0.1), "VISUAL", tote)
bpy.context.view_layer.update()
walkie_path = OUT / "walkie_door" / "scene.xml"
export_world(bpy.context, str(walkie_path))
root = xml(walkie_path)
compiler = root.find("compiler")
assert compiler.get("angle") == "radian", compiler.attrib
assert not {"inertiafromgeom", "inertiagrouprange"} & set(compiler.attrib), compiler.attrib
hinge = root.find(".//joint[@name='Cabinet_door_hinge']")
low, high = (float(v) for v in hinge.get("range").split())
assert abs(low - math.radians(-100)) < 1e-8 and high == 0, hinge.attrib
assert abs(float(hinge.get("ref")) - math.radians(-30)) < 1e-8, hinge.attrib
assert hinge.get("damping") == "0.3", hinge.attrib
floor = root.find(".//geom[@name='floor']")
assert (floor.get("contype"), floor.get("conaffinity")) == ("1", "32"), floor.attrib
tote_body = root.find(".//body[@name='Tote']")
colliders = [g for g in tote_body.findall("geom") if g.get("contype") != "0"]
assert len(colliders) == 2 and all((g.get("contype"), g.get("conaffinity")) == ("32", "121") for g in colliders)
crate_collision = next(g for g in root.iter("geom") if g.get("name", "").startswith("Crate_") and g.get("name").endswith("_collision"))
assert crate_collision.get("friction") == "0.8 0.005 0.0001", crate_collision.attrib
assert crate_collision.get("mass") == "2.5", crate_collision.attrib
cabinet_panel = next(g for g in root.find(".//body[@name='Cabinet_door']").findall("geom") if g.get("contype") != "0")
# Door parts take the door masks, which Walkie's world_loader expects for anything on a joint.
assert (cabinet_panel.get("contype"), cabinet_panel.get("conaffinity")) == ("32", "121"), cabinet_panel.attrib
wall = next(g for g in root.iter("geom") if g.get("name", "").startswith("Wall_") and g.get("name").endswith("_collision"))
assert (wall.get("contype"), wall.get("conaffinity")) == ("64", "56"), wall.attrib
summary["walkie_door"] = str(walkie_path)

# 2. 'Leave out' on a body group leaves out every part it owns.
fresh_scene()
excluded = empty("Excluded_group", (0, 0, 0), "EXCLUDE")
box("Excluded_part", (0, 0, 0.5), (0.1, 0.1, 0.1), "STATIC", excluded)
box("Kept_box", (1, 0, 0.5), (0.1, 0.1, 0.1))
bpy.context.view_layer.update()
excluded_path = OUT / "excluded_group" / "scene.xml"
report = export_world(bpy.context, str(excluded_path))
assert [d["source"] for d in report["object_details"]] == ["Kept_box"], report["object_details"]
assert report.get("compound_bodies", 0) == 0, report
text = excluded_path.read_text()
assert "Excluded" not in text, text

# 3. Body group names that MuJoCo would reject are reported before writing.
fresh_scene()
box("Crate", (0, 0, 0.5), (0.1, 0.1, 0.1))  # Its generated body label is Crate_000.
clash = empty("Crate_000", (1, 0, 0))
box("Shelf_part", (1, 0, 0.5), (0.1, 0.1, 0.1), "STATIC", clash)
summary["name_clash"] = expect_error("Crate_000", OUT / "name_clash" / "scene.xml")
clash.name = "world"
summary["world_name"] = expect_error("reserves", OUT / "world_name" / "scene.xml")

# 4. A Door or Movable part inside a group would silently lose its joint.
fresh_scene()
cabinet = empty("Cabinet", (0, 0, 0))
leaf = box("Leaf", (0.3, 0, 0.5), (0.3, 0.02, 0.5), "HINGE", cabinet)
summary["child_hinge"] = expect_error("would lose its joint", OUT / "child_hinge" / "scene.xml")
leaf.mujoco_world.role = "FREE"
expect_error("would lose its joint", OUT / "child_free" / "scene.xml")

# 5. Group door limits follow the same rules as a single door.
leaf.mujoco_world.role = "STATIC"
cabinet.mujoco_world.role = "HINGE"
cabinet.mujoco_world.hinge_min = cabinet.mujoco_world.hinge_max = 10
summary["equal_limits"] = expect_error("contain 0", OUT / "equal_limits" / "scene.xml")
cabinet.mujoco_world.hinge_min, cabinet.mujoco_world.hinge_max = -10, 10
cabinet.mujoco_world.hinge_start = 20
summary["start_outside"] = expect_error("starting angle", OUT / "start_outside" / "scene.xml")

# 6. A flat mesh tilted in edit mode passes the local box test, but MuJoCo cannot
# build its convex hull.
fresh_scene()
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0.5))
ramp = bpy.context.object
ramp.name = "Ramp"
for vertex in ramp.data.vertices:
    vertex.co.z = vertex.co.x
ramp.mujoco_world.role = "STATIC"
ramp.mujoco_world.collision = "CONVEX"
summary["tilted_convex"] = expect_error("zero thickness", OUT / "tilted_convex" / "scene.xml")
ramp.mujoco_world.collision = "BOX"
report = export_world(bpy.context, str(OUT / "tilted_box" / "scene.xml"))
assert any("tilted" in warning for warning in report["warnings"]), report["warnings"]

# 7. An upper-case suffix is kept through the grouped export.
fresh_scene()
bin_body = empty("Bin", (0, 0, 0.2), "FREE")
box("Bin_collider", (0, 0, 0.2), (0.1, 0.1, 0.1), "COLLIDER", bin_body)
bpy.context.view_layer.update()
upper = OUT / "suffix" / "scene.XML"
report = export_world(bpy.context, str(upper))
assert report["file"] == "scene.XML" and report["compound_bodies"] == 1, report
assert xml(upper).find(".//body[@name='Bin']") is not None
summary["suffix"] = str(upper)

# 8. Adopted fromto primitives and alternative orientations stay verbatim until
# edited; edits write an explicit full size. MuJoCo 3.15 compiled body_quat
# values for the source orientations (degree angles, eulerseq zyx) are below.
SOURCE = """<mujoco model="fromto_source">
  <compiler angle="degree" eulerseq="zyx" autolimits="true"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <geom name="Leg" type="capsule" fromto="0.4 0.2 0 0.4 0.2 0.68" size="0.02"/>
    <geom name="Rail" type="capsule" fromto="0 0 0.5 0.4 0 0.5" size="0.05"/>
    <geom name="Post" type="cylinder" fromto="-0.5 0 0 -0.5 0 0.6" size="0.03"/>
    <geom name="Beam" type="box" fromto="0 0 1 0 0.3 1" size="0.05"/>
    <body name="Shelf_e" pos="1 0 0" euler="10 20 30"><geom name="Shelf_e_box" type="box" size="0.1 0.1 0.1"/></body>
    <body name="Shelf_aa" pos="0 1 0" axisangle="0 1 1 30"><geom name="Shelf_aa_box" type="box" size="0.1 0.1 0.1"/></body>
    <body name="Shelf_xy" pos="0 0 1.5" xyaxes="0 1 0 -1 0.2 0"><geom name="Shelf_xy_box" type="box" size="0.1 0.1 0.1"/></body>
    <body name="Shelf_za" pos="1 1 0" zaxis="1 1 0"><geom name="Shelf_za_box" type="box" size="0.1 0.1 0.1" euler="0 0 45"/></body>
  </worldbody>
</mujoco>
"""
MUJOCO_BODY_QUATS = {"Shelf_e": (0.951548525, 0.239298338, 0.189307857, 0.038134576),
                     "Shelf_aa": (0.965925826, 0.0, 0.183012702, 0.183012702),
                     "Shelf_xy": (0.707106781, 0.0, 0.0, 0.707106781),
                     "Shelf_za": (0.707106781, -0.5, 0.5, 0.0)}
fromto_dir = OUT / "fromto"
fromto_dir.mkdir(parents=True, exist_ok=True)
source_path = fromto_dir / "source.xml"
source_path.write_text(SOURCE)
fresh_scene()
bpy.ops.mesh.primitive_plane_add(size=4)
bpy.context.object.name = "floor"


def cylinder(name, radius, depth, pos, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=depth, location=pos, rotation=rotation)
    bpy.context.object.name = name
    return bpy.context.object


# Blender modellers build +Z from 'from' to 'to'; MuJoCo's frame points the
# other way, which a symmetric capsule or cylinder does not distinguish.
cylinder("Leg", 0.02, 0.68, (0.4, 0.2, 0.34))
rail = cylinder("Rail", 0.05, 0.5, (0.2, 0, 0.5), (0, math.pi / 2, 0))
rail["mujoco_type"] = "capsule"  # Legacy primitive metadata uses the capsule's full length.
post = cylinder("Post", 0.03, 0.6, (-0.5, 0, 0.3))
box("Beam", (0, 0.15, 1), (0.05, 0.05, 0.15), "STATIC", rotation=(math.pi / 2, 0, 0))
for name, position in (("Shelf_e", (1, 0, 0)), ("Shelf_aa", (0, 1, 0)), ("Shelf_xy", (0, 0, 1.5)), ("Shelf_za", (1, 1, 0))):
    shelf = empty(name, position)
    shelf.rotation_mode = "QUATERNION"
    shelf.rotation_quaternion = MUJOCO_BODY_QUATS[name]
    part = box(name + "_box", (0, 0, 0), (0.1, 0.1, 0.1))
    part.parent = shelf
    if name == "Shelf_za":
        part.rotation_euler = (math.radians(45), 0, 0)  # euler="0 0 45" in zyx order is about x.
bpy.context.view_layer.update()
adopt_mjcf(bpy.context, str(source_path))
untouched_path = fromto_dir / "untouched.xml"
export_world(bpy.context, str(untouched_path))
source_root, untouched_root = xml(source_path), xml(untouched_path)
for tag in ("geom", "body"):
    for element in source_root.iter(tag):
        exported = untouched_root.find(f".//{tag}[@name='{element.get('name')}']")
        for field in ("fromto", "size", "euler", "axisangle", "xyaxes", "zaxis"):
            assert exported.get(field) == element.get(field), (element.get("name"), field, exported.attrib)
        assert exported.get("quat") is None or element.get("quat") is not None, exported.attrib

# Move one fromto capsule and stretch one fromto cylinder.
bpy.context.scene.objects["Leg"].location.x += 0.1
post.scale.z = 1.5
# Exercise the radian part export inside a degree world: a loose new door.
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(-1, -1, 0.5))
flap = bpy.context.object
flap.name = "New_flap"
flap.mujoco_world.role = "HINGE"
flap.mujoco_world.hinge_min, flap.mujoco_world.hinge_max = -90, 0
flap.mujoco_world.hinge_start = -20
# A new door body group: limits in degrees, body rotated by the start angle.
new_door = empty("New_door", (-1, 1, 0), "HINGE")
new_door.mujoco_world.mass = 3
new_door.mujoco_world.hinge_min, new_door.mujoco_world.hinge_max = -80, 0
new_door.mujoco_world.hinge_start = -25
box("New_door_panel", (-0.7, 1, 0.5), (0.3, 0.02, 0.4), "STATIC", new_door)
bpy.context.view_layer.update()
edited_path = fromto_dir / "edited.xml"
export_world(bpy.context, str(edited_path))
edited = xml(edited_path)
leg = edited.find(".//geom[@name='Leg']")
assert leg.get("fromto") is None and leg.get("size") == "0.02 0.34", leg.attrib
assert abs(float(leg.get("pos").split()[0]) - 0.5) < 1e-6, leg.attrib
post_geom = edited.find(".//geom[@name='Post']")
assert post_geom.get("fromto") is None and post_geom.get("size") == "0.03 0.45", post_geom.attrib
assert edited.find(".//geom[@name='Beam']").get("fromto") == "0 0 1 0 0.3 1"
assert edited.find(".//body[@name='Shelf_e']").get("euler") == "10 20 30"
flap_joint = next(j for j in edited.iter("joint") if j.get("name", "").startswith("New_flap_"))
assert flap_joint.get("range") == "-90 0" and flap_joint.get("ref") == "-20", flap_joint.attrib
door_joint = edited.find(".//joint[@name='New_door_hinge']")
assert door_joint.get("range") == "-80 0" and door_joint.get("ref") == "-25", door_joint.attrib
# The same beam modelled along local X: a rewritten pose would turn it by 90 degrees.
beam = bpy.context.scene.objects["Beam"]
beam.rotation_euler, beam.scale = (0, 0, math.pi / 2), (0.15, 0.05, 0.05)
summary["fromto_axis"] = expect_error("not modelled along local Z", fromto_dir / "misaligned.xml")
summary["fromto"] = {"source": str(source_path), "untouched": str(untouched_path), "edited": str(edited_path),
                     "flap_joint": flap_joint.get("name")}

# 9. An unnamed source geom matched to the lone baked atlas still gets a mesh.
atlas_dir = OUT / "unnamed_atlas"
atlas_dir.mkdir(parents=True, exist_ok=True)
(atlas_dir / "atlas.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
(atlas_dir / "source.xml").write_text("""<mujoco model="atlas_source">
  <compiler angle="radian"/>
  <asset><mesh name="atlas_mesh" file="atlas.obj"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <geom type="mesh" mesh="atlas_mesh" contype="0" conaffinity="0" group="1"/>
  </worldbody>
</mujoco>
""")
fresh_scene()
bpy.ops.mesh.primitive_plane_add(size=4)
bpy.context.object.name = "floor"
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
atlas = bpy.context.object
atlas.name = "Baked atlas"
atlas["mjworld_baked_atlas"] = True
bpy.context.view_layer.update()
adopt_mjcf(bpy.context, str(atlas_dir / "source.xml"))
atlas_path = atlas_dir / "exported" / "scene.xml"
export_world(bpy.context, str(atlas_path))
meshes = [m.get("name") for m in xml(atlas_path).iter("mesh")]
assert meshes == ["Baked_atlas_authored_mesh"], meshes
summary["unnamed_atlas"] = str(atlas_path)

(OUT / "export_fixes.json").write_text(json.dumps(summary, indent=2) + "\n")
print("EXPORT_FIXES_FIXTURE_OK", json.dumps(summary))
