"""Body hierarchies and lossless physics adoption for existing MJCF worlds.

No room layouts live here. Adoption attaches physics to same-named Blender
objects; Blender remains the editable geometry and placement source.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Quaternion, Vector


def is_body(obj):
    return obj.type == "EMPTY" and (obj.mujoco_world.body or obj.get("simulation_body", False))


def body_owner(obj):
    parent = obj.parent
    while parent:
        if is_body(parent):
            return parent
        parent = parent.parent
    return None


def _included(obj):
    while obj:
        if obj.mujoco_world.role == "EXCLUDE":
            return False
        obj = obj.parent
    return True


def _json(obj, key, fallback=None):
    value = obj.get(key)
    return json.loads(value) if value else fallback


def _rigid(matrix, unit=1.0):
    return Matrix.LocRotScale(matrix.translation * unit, matrix.to_quaternion().normalized(), Vector((1, 1, 1)))


def _scale(matrix):
    return Vector(tuple(matrix.col[i].to_3d().length for i in range(3)))


def _pose(matrix):
    from .exporter import numbers
    return {"pos": numbers(matrix.translation), "quat": numbers(matrix.to_quaternion().normalized())}


_AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
# Primitive types that accept fromto; their length then comes from the segment.
_ELONGATED = {"capsule", "cylinder", "box", "ellipsoid"}


def _floats(attrs, field, count=None):
    from .exporter import ExportError
    try:
        values = [float(v) for v in attrs[field].split()]
    except ValueError as exc:
        raise ExportError(f"{attrs.get('name', '<unnamed>')}: {field} must contain numbers.") from exc
    if count is not None and len(values) != count:
        raise ExportError(f"{attrs.get('name', '<unnamed>')}: {field} needs {count} numbers.")
    return values


def _z_to_quaternion(direction):
    """MuJoCo's minimal rotation taking +Z onto ``direction`` (zaxis and fromto)."""
    direction = direction.normalized()
    axis = Vector((0, 0, 1)).cross(direction)
    sine = axis.length
    axis = Vector((1, 0, 0)) if sine < 1e-10 else axis / sine
    return Quaternion(axis, math.atan2(sine, direction.z))


def _orientation(attrs, compiler=None):
    """Frame rotation of an MJCF element, honouring compiler angle and eulerseq."""
    degree = compiler is None or compiler.get("angle", "degree") == "degree"
    def angle(value):
        return math.radians(value) if degree else value
    if "quat" in attrs or not any(field in attrs for field in ("axisangle", "xyaxes", "zaxis", "euler")):
        return Quaternion(tuple(float(v) for v in attrs.get("quat", "1 0 0 0").split()))
    if "axisangle" in attrs:
        x, y, z, value = _floats(attrs, "axisangle", 4)
        return Quaternion(Vector((x, y, z)).normalized(), angle(value))
    if "xyaxes" in attrs:
        values = _floats(attrs, "xyaxes", 6)
        x_axis = Vector(values[:3]).normalized()
        y_axis = Vector(values[3:])
        y_axis = (y_axis - x_axis * x_axis.dot(y_axis)).normalized()
        return Matrix((x_axis, y_axis, x_axis.cross(y_axis))).transposed().to_quaternion()
    if "zaxis" in attrs:
        return _z_to_quaternion(Vector(_floats(attrs, "zaxis", 3)))
    sequence = compiler.get("eulerseq", "xyz") if compiler is not None else "xyz"
    rotation = Quaternion()
    for axis, value in zip(sequence, _floats(attrs, "euler", 3)):
        step = Quaternion(Vector(_AXES[axis.lower()]), angle(value))
        # Lower case rotates about the moving frame, upper case about the parent.
        rotation = rotation @ step if axis.islower() else step @ rotation
    return rotation


def _matrix(attrs, compiler=None):
    if attrs.get("fromto"):
        points = _floats(attrs, "fromto", 6)
        start, end = Vector(points[:3]), Vector(points[3:])
        # MuJoCo aligns the geom's +Z with from - to, centred on the segment.
        return Matrix.LocRotScale((start + end) / 2, _z_to_quaternion(start - end), Vector((1, 1, 1)))
    pos = Vector(tuple(float(v) for v in attrs.get("pos", "0 0 0").split()))
    return Matrix.LocRotScale(pos, _orientation(attrs, compiler), Vector((1, 1, 1)))


def _full_size(attrs):
    """Explicit primitive size, with the length a fromto segment implies."""
    from .exporter import ExportError
    kind = attrs.get("type", "sphere")
    size = _floats(attrs, "size") if attrs.get("size") else []
    if attrs.get("fromto") and kind in _ELONGATED and size:
        points = _floats(attrs, "fromto", 6)
        # Double precision: mathutils vectors are float32 and would write 0.340000007.
        half = math.dist(points[:3], points[3:]) / 2
        size = [size[0], half] if kind in {"capsule", "cylinder"} else [size[0], size[0], half]
    needed = {"sphere": 1, "capsule": 2, "cylinder": 2, "box": 3, "ellipsoid": 3}.get(kind)
    if needed is None:
        return size
    if len(size) < needed:
        raise ExportError(f"{attrs.get('name', '<unnamed>')}: a {kind} needs {needed} explicit size values to be moved or scaled; "
                          "write them on the geom in the source MJCF.")
    return size[:needed]


def _same_segment(transform, attrs, tolerance=2e-6):
    """Whether a pose still describes an unchanged fromto primitive.

    The shape is symmetric about its segment, so a Blender object built with +Z
    from 'from' to 'to', or rolled about its axis, is the same geom.
    """
    frame = _matrix(attrs)
    def close(a, b):
        return max(abs(x - y) for x, y in zip(a, b)) <= tolerance
    if not close(transform.translation, frame.translation):
        return False
    z_axis, frame_z = transform.col[2].to_3d(), frame.col[2].to_3d()
    if not (close(z_axis, frame_z) or close(z_axis, -frame_z)):
        return False
    if attrs.get("type", "sphere") in {"capsule", "cylinder"}:
        return True
    # Box and ellipsoid fromto shapes have equal X and Y sizes.
    x_axis = transform.col[0].to_3d()
    return any(close(x_axis, axis) or close(x_axis, -axis) for axis in (frame.col[0].to_3d(), frame.col[1].to_3d()))


def _check_segment_axis(obj, kind, size, unit):
    """Refuse to rewrite a fromto primitive whose Blender object is not modelled along local Z.

    A rewritten fromto geom takes the object's own frame, and MuJoCo lays a
    primitive's length along that frame's Z axis. An object modelled along
    local X or Y would turn the shape by 90 degrees, so stop instead.
    """
    from .exporter import ExportError
    values = [float(v) for v in size.split()]
    radius, half = values[0], values[1] if kind in {"capsule", "cylinder"} else values[2]
    length = 2 * (half + radius) if kind == "capsule" else 2 * half
    scale = _scale(obj.matrix_world)
    extents = [(max(v[i] for v in obj.bound_box) - min(v[i] for v in obj.bound_box)) * scale[i] * unit for i in range(3)]
    if max(extents) - min(extents) <= 0.01 * max(extents):
        return  # Nearly cubic: no modelling axis stands out.
    if min(range(3), key=lambda i: abs(extents[i] - length)) != 2:
        raise ExportError(f"{obj.name}: this fromto {kind} changed, but its Blender object is not modelled along local Z, "
                          "so the new pose would turn the shape. Rotate the mesh so its length is along local Z, "
                          "or edit the fromto in the source MJCF.")


def _resolved_path(value):
    return Path(bpy.path.abspath(str(value))).expanduser().resolve()


def adopt_mjcf(context, filepath, *, object_map=None):
    """Attach exact MJCF physics to an existing Blender world by object name.

    ``object_map`` optionally maps MJCF names to Blender object names. No mesh,
    material, object transform, camera, light or existing user edit is changed.
    Source assets remain relative to the blend file where possible.
    """
    from .exporter import ExportError
    filepath = _resolved_path(filepath)
    try:
        root = ET.parse(filepath).getroot()
    except ET.ParseError as exc:
        raise ExportError(f"The chosen file is not valid MJCF XML: {exc}") from exc
    if root.tag != "mujoco" or root.find("worldbody") is None:
        raise ExportError("Choose a standalone MJCF world containing <worldbody>.")
    if root.findall("include"):
        raise ExportError("Adoption needs a standalone XML. Expand MJCF includes before adopting this world.")
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("coordinate", "local") != "local":
        raise ExportError("Adoption supports local MJCF coordinates only.")
    # Inertia, joint limits, contact defaults and solver options are preserved in
    # their original units. Object poses are written as quaternions on export.
    mapping = object_map or {}
    scene = context.scene
    candidates = {obj.name: obj for obj in scene.objects}
    def find(name):
        return candidates.get(mapping.get(name, name))
    # First resolve and check everything, then change the scene, so a rejected
    # adoption leaves the previous adoption and every setting untouched.
    missing_bodies, missing_geoms = [], []
    body_plans, geom_plans = [], []
    for source in root.findall("./worldbody//body"):
        obj = find(source.get("name", ""))
        if obj is None or obj.type != "EMPTY":
            missing_bodies.append(source.get("name", "<unnamed>"))
            continue
        name = source.get("name", "<unnamed>")
        values = {"role": "FREE" if source.find("freejoint") is not None or source.find("joint[@type='free']") is not None else "HINGE" if source.find("joint") is not None else "STATIC"}
        try:
            inertial = source.find("inertial")
            if inertial is not None and inertial.get("mass"):
                values["mass"] = float(inertial.get("mass"))
            joint = source.find("joint[@type='hinge']")
            if joint is not None:
                degree = compiler is None or compiler.get("angle", "degree") == "degree"
                conversion = 1.0 if degree else 180.0 / math.pi
                # The fallback is the add-on's default door, already in degrees.
                limits = [float(v) * conversion for v in joint.get("range").split()] if joint.get("range") else [-110.0, 0.0]
                if len(limits) != 2:
                    raise ExportError(f"Body '{name}': its hinge range needs two numbers.")
                values["hinge_min"], values["hinge_max"] = limits
                values["hinge_start"] = float(joint.get("ref", "0")) * conversion
                values["damping"] = float(joint.get("damping", "0"))
                axis = [float(v) for v in joint.get("axis", "0 0 1").split()]
                if len(axis) != 3:
                    raise ExportError(f"Body '{name}': its hinge axis needs three numbers.")
                values["hinge_axis"] = "XYZ"[max(range(3), key=lambda i: abs(axis[i]))]
        except ValueError as exc:
            if isinstance(exc, ExportError):
                raise
            raise ExportError(f"Body '{name}': its inertial or joint attributes must contain numbers.") from exc
        children = [ET.tostring(node, encoding="unicode") for node in source if node.tag not in {"body", "geom"}]
        body_plans.append((obj, dict(source.attrib), children, values))
    world_geoms = list(root.findall("./worldbody/geom"))
    for source in root.findall("./worldbody//geom"):
        obj = find(source.get("name", ""))
        # A joined lighting atlas often has a descriptive Blender object name.
        if obj is None and source in world_geoms:
            atlases = [o for o in scene.objects if o.type == "MESH" and (o.get("baked_static_visual") or o.get("mjworld_baked_atlas"))]
            unmatched = [g for g in world_geoms if find(g.get("name", "")) is None]
            if len(atlases) == len(unmatched) == 1:
                obj = atlases[0]
        if obj is None or obj.type != "MESH":
            missing_geoms.append(source.get("name", "<unnamed>"))
            continue
        attrs = dict(source.attrib)
        scale = list(_scale(obj.matrix_world))
        bounds = [list(v) for v in obj.bound_box]
        # Legacy primitive authoring metadata identifies unit box/cylinder/
        # sphere meshes. Use their source dimensions as the baseline so manual
        # geometry/scale edits made before the first adoption are preserved too.
        kind = attrs.get("type", "sphere")
        size = None
        if obj.get("mujoco_type") == kind and kind not in {"mesh", "plane"}:
            try:
                size = _full_size(attrs)
            except ExportError:
                # Size inherited from a <default> class: measure the object instead.
                size = None
        if size:
            half = [1, 1, 1]
            if kind in {"box", "ellipsoid"}:
                scale = size[:3]
            elif kind == "sphere":
                scale = [size[0]] * 3
            elif kind == "cylinder":
                scale = [size[0], size[0], size[1]]
            elif kind == "capsule":
                scale, half = [1, 1, 1], [size[0], size[0], size[0] + size[1]]
            bounds = [[x * half[0], y * half[1], z * half[2]] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        collision_only = obj.get("collision_only") or attrs.get("rgba", "").split()[-1:] == ["0"]
        visual_only = obj.get("visual_only") or (attrs.get("contype") == attrs.get("conaffinity") == "0") or "visual" in attrs.get("class", "").lower()
        if obj.get("preserved_original_static_visual"):
            role = "VISUAL" if visual_only else "STATIC"
        else:
            role = "COLLIDER" if collision_only else "VISUAL" if visual_only else "STATIC"
        emission = None
        if obj.get("baked_static_visual") or obj.get("mjworld_baked_atlas"):
            material = root.find(f"./asset/material[@name='{attrs.get('material')}']")
            if material is not None:
                try:
                    emission = float(material.get("emission", ".2"))
                except ValueError as exc:
                    raise ExportError(f"Material '{attrs.get('material')}': emission must be a number.") from exc
        geom_plans.append((obj, attrs, scale, bounds, role, emission))
    if missing_bodies or missing_geoms:
        # A partial adoption can silently erase collision geometry. Do not allow
        # it to become the active project; report every missing physics source.
        raise ExportError("MJCF names were not found in this Blender scene. Bodies: " + ", ".join(missing_bodies[:10]) + "; geoms: " + ", ".join(missing_geoms[:10]) + ". Use object_map for renamed objects.")
    for obj in scene.objects:
        for key in ("mjworld_mjcf_body", "mjworld_mjcf_geom", "mjworld_mjcf_children", "mjworld_mjcf_original_scale",
                    "mjworld_mjcf_original_bounds", "mjworld_settings_baseline", "mjworld_mjcf_owner"):
            if key in obj:
                del obj[key]
        if obj.type == "MESH" and obj.get("simulation_geometry"):
            obj["mjworld_adopted_exclude"] = True
    for obj, attrs, children, values in body_plans:
        obj.mujoco_world.body = True
        obj["mjworld_mjcf_body"] = json.dumps(attrs)
        obj["mjworld_mjcf_children"] = json.dumps(children)
        # Copies made later with Shift+D carry this too; export uses it to keep
        # the adopted name on this object only.
        obj["mjworld_mjcf_owner"] = obj.name
        for key, value in values.items():
            setattr(obj.mujoco_world, key, value)
        obj["mjworld_settings_baseline"] = json.dumps({key: getattr(obj.mujoco_world, key) for key in ("role", "mass", "hinge_min", "hinge_max", "hinge_start", "hinge_axis", "damping")})
    for obj, attrs, scale, bounds, role, emission in geom_plans:
        obj["mjworld_mjcf_geom"] = json.dumps(attrs)
        obj["mjworld_mjcf_owner"] = obj.name
        obj["mjworld_adopted_exclude"] = False
        obj["mjworld_mjcf_original_scale"] = scale
        obj["mjworld_mjcf_original_bounds"] = json.dumps(bounds)
        # Explicit helpers remain exportable even inside hidden collections.
        if obj.get("preserved_original_static_visual"):
            obj["mjworld_bake_source"] = True
        obj.mujoco_world.role = role
        if emission is not None:
            scene.mujoco_world.baked_emission = emission
    scene["mjworld_mjcf_template"] = ET.tostring(root, encoding="unicode")
    scene["mjworld_mjcf_source"] = bpy.path.relpath(str(filepath)) if bpy.data.filepath else str(filepath)
    scene["mjworld_mjcf_adoption_version"] = 1
    from .authored_materials import capture_material_baselines
    capture_material_baselines(scene, root)
    return {"bodies": len(body_plans), "geoms": len(geom_plans), "source": str(filepath)}


def group_selected(context, role="FREE"):
    """Parent selected parts to one editable rigid body without moving them."""
    from .exporter import ExportError
    objects = [o for o in context.selected_objects if o.type in {"MESH", "EMPTY"}]
    if not objects:
        raise ExportError("Select the visible parts and collision helpers to group.")
    selected = set(objects)
    top = [o for o in objects if o.parent not in selected]
    parent = bpy.data.objects.new("Movable body" if role == "FREE" else "Door body" if role == "HINGE" else "Fixed body", None)
    context.collection.objects.link(parent)
    parent.empty_display_type = "PLAIN_AXES"
    parent.empty_display_size = 0.12 / max(context.scene.unit_settings.scale_length, 1e-12)
    parent.location = context.scene.cursor.location if role == "HINGE" else sum((o.matrix_world.translation for o in top), Vector()) / len(top)
    parent.mujoco_world.body = True
    parent.mujoco_world.role = role
    parent.mujoco_world.mass = 8.0 if role == "HINGE" else 1.0
    context.view_layer.update()
    for obj in top:
        world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = world
        if obj.type == "MESH" and obj.mujoco_world.role in {"FREE", "HINGE"}:
            obj.mujoco_world.role = "STATIC"
    bpy.ops.object.select_all(action="DESELECT")
    parent.select_set(True)
    context.view_layer.objects.active = parent
    return parent


def validate_parts(scene, objects, labels):
    """Reject grouped parts that would export incorrectly, before any file is written.

    ``labels`` are the body names the ordinary exporter gives ``objects``.
    """
    from .exporter import ExportError, check_door
    owners = set()
    for obj in objects:
        owner = body_owner(obj)
        if owner is None:
            continue
        if obj.mujoco_world.role in {"FREE", "HINGE"}:
            kind = "Movable" if obj.mujoco_world.role == "FREE" else "Door"
            raise ExportError(f"{obj.name}: a {kind} part inside body group '{owner.name}' would lose its joint. "
                              "Set the part to Fixed and give the group that role, or use a nested body Empty for a door inside a group.")
        while owner is not None:
            owners.add(owner)
            owner = body_owner(owner)
    # Body groups keep their Blender names; ungrouped parts get generated
    # labels. Both share MuJoCo's body namespace, which also contains 'world'.
    names = {owner.name: owner for owner in owners}
    if "world" in names:
        raise ExportError("A body group is named 'world', which MuJoCo reserves for the world body. Rename it.")
    for obj, label in zip(objects, labels):
        if body_owner(obj) is None and label in names:
            raise ExportError(f"{obj.name}: its exported body name '{label}' is also the name of body group '{label}'. Rename one of them.")
    adopted = bool(scene.get("mjworld_mjcf_template"))
    for owner in sorted(owners, key=lambda o: o.name):
        # Adopted joints keep their source limits; export_adopted checks edits.
        if owner.mujoco_world.role == "HINGE" and not (adopted and owner.get("mjworld_mjcf_body")):
            check_door(owner)


def moving_role(owner):
    """FREE or HINGE when body ``owner`` moves with such a body (itself included), else None.

    Matches robot_bringup.world_loader, which treats every geom below a joint
    as a moving object.
    """
    while owner is not None:
        if owner.mujoco_world.role in {"FREE", "HINGE"}:
            return owner.mujoco_world.role
        owner = body_owner(owner)
    return None


def check_evaluated(context, objects):
    """Refuse objects whose placement and modifiers Blender does not evaluate.

    Objects disabled in viewports (the monitor icon, on the object or any of its
    collections) or in a collection excluded from the view layer keep a stale
    matrix_world when a parent moves, and their modifiers are not applied.
    Hiding with the eye icon keeps them evaluated, so that is the supported way
    to hide collision helpers.
    """
    from .exporter import ExportError
    enabled = set()
    def walk(layer, parent_enabled):
        on = parent_enabled and not layer.exclude and not layer.collection.hide_viewport
        if on:
            enabled.update(layer.collection.objects)
        for child in layer.children:
            walk(child, on)
    walk(context.view_layer.layer_collection, True)
    stale = [obj.name for obj in objects if obj.hide_viewport or obj not in enabled]
    if stale:
        raise ExportError(", ".join(stale[:10]) + (" and others" if len(stale) > 10 else "") +
                          ": disabled in viewports (monitor icon) or in a collection excluded from the view layer, so Blender "
                          "does not update their placement or modifiers. Enable them, and use the eye icon (Hide in Viewport) "
                          "to hide helpers instead.")


def regroup_export(context, root, report):
    """Lift exported parts into compound bodies, retaining relative transforms.

    Works on the in-memory world from the ordinary exporter, before any file
    is written.
    """
    from .exporter import ExportError, check_door, door_radians, numbers, property_numbers, property_value, role_masks
    world = root.find("worldbody")
    unit = context.scene.unit_settings.scale_length
    settings = context.scene.mujoco_world
    bodies = {}
    def make(obj):
        if obj in bodies:
            return bodies[obj]
        owner = body_owner(obj)
        parent = make(owner) if owner else world
        transform = _rigid(obj.matrix_world, unit)
        if owner:
            transform = _rigid(owner.matrix_world, unit).inverted() @ transform
        body = ET.SubElement(parent, "body", name=obj.name, **_pose(transform))
        cfg = obj.mujoco_world
        if cfg.role == "FREE":
            if owner:
                raise ExportError(f"{obj.name}: a movable free body must be a top-level body.")
            ET.SubElement(body, "freejoint", name=obj.name + "_free")
        elif cfg.role == "HINGE":
            check_door(obj)
            # The authored pose is the closed pose. Rotate the body at the
            # start, and set ref to that angle so limits stay in closed-space.
            axis = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis])
            low, high, start = door_radians(cfg)
            body.set("quat", numbers(transform.to_quaternion() @ Quaternion(axis, start)))
            ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((low, high)), ref=numbers([start]), damping=property_numbers([cfg.damping]))
        bodies[obj] = body
        return body
    groups = {}
    for detail in report["object_details"]:
        obj = context.scene.objects.get(detail["source"])
        owner = body_owner(obj) if obj else None
        if owner is not None:
            groups.setdefault(owner, []).append((obj, detail))
    for owner, parts in groups.items():
        target = make(owner)
        inverse = _rigid(owner.matrix_world, unit).inverted()
        colliders = [d for _o, d in parts if d["collision"]]
        if owner.mujoco_world.role in {"FREE", "HINGE"} and not colliders:
            raise ExportError(f"{owner.name}: add at least one collision helper before making this body movable.")
        for obj, detail in parts:
            original = world.find(f"body[@name='{detail['body']}']")
            if original is None:
                continue
            relative = inverse @ _rigid(obj.matrix_world, unit)
            # Doors and movable objects inside a moving body move with it, so
            # they take its contact masks (robot_bringup.world_loader agrees).
            contype, conaffinity = role_masks(settings, moving_role(owner))
            for geom in list(original.findall("geom")):
                pose = relative @ _matrix(geom.attrib)
                geom.attrib.update(_pose(pose))
                if geom.get("contype") != "0":
                    geom.set("contype", contype)
                    geom.set("conaffinity", conaffinity)
                    geom.set("mass", numbers([property_value(owner.mujoco_world.mass) / len(colliders)]) if owner.mujoco_world.role in {"FREE", "HINGE"} else "0")
                target.append(geom)
            world.remove(original)
            detail["body"] = owner.name
            detail["body_role"] = owner.mujoco_world.role
    report["compound_bodies"] = len(groups)
    grouped_names = {obj.name for parts in groups.values() for obj, _d in parts}
    report["warnings"] = [w for w in report["warnings"] if not any(w.startswith(name + ": parent transforms") for name in grouped_names)]
    return report


def _append_new_parts(context, objects, root, world, asset, assets, make_body, warnings, asset_prefix):
    """Use the ordinary exporter for newly added furniture and compound props.

    Their files go through ``assets``, the adopted export's AssetWriter. Parts
    grouped under a body get mass 0 here; export_adopted distributes the
    body's Weight afterwards when the body needs it.
    """
    from .exporter import _export_simple, property_numbers, role_masks, slug
    if not objects:
        return []
    scene = context.scene
    selected, active = list(context.selected_objects), context.view_layer.objects.active
    only = scene.mujoco_world.selected_only
    asset_directory = scene.mujoco_world.asset_directory
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        scene.mujoco_world.selected_only = True
        scene.mujoco_world.asset_directory = ""
        with tempfile.TemporaryDirectory(prefix="mjworld-parts-") as temporary:
            path = Path(temporary) / "parts.xml"
            report = _export_simple(context, str(path))
            parts_root = ET.parse(path).getroot()
            names = {element.get("name") for element in asset}
            rename = {}
            for element in parts_root.find("asset"):
                name = "added_" + element.get("name")
                while name in names:
                    name = "added_" + name
                names.add(name)
                rename[element.get("name")] = name
            for element in parts_root.find("asset"):
                old_name = element.get("name")
                element.set("name", rename[old_name])
                for field in ("mesh", "material", "texture"):
                    if element.get(field) in rename:
                        element.set(field, rename[element.get(field)])
                if element.get("file"):
                    original = Path(temporary) / "parts_assets" / element.get("file")
                    element.set("file", asset_prefix + "/" + assets.copy(original, slug(rename[old_name]) + original.suffix))
                asset.append(element)
            details = report["object_details"]
            groups = {}
            for detail in details:
                obj = scene.objects.get(detail["source"])
                owner = body_owner(obj)
                groups.setdefault(owner, []).append((obj, detail))
            for owner, members in groups.items():
                target = make_body(owner) if owner else world
                if owner:
                    contype, conaffinity = role_masks(scene.mujoco_world, moving_role(owner))
                for obj, detail in members:
                    body = parts_root.find(f"./worldbody/body[@name='{detail['body']}']")
                    # The part export is always in radians; follow the adopted
                    # world. Degrees come from the properties, not the 9-digit
                    # radian text, so -90 stays -90.
                    if root.find("compiler").get("angle", "degree") == "degree":
                        cfg = obj.mujoco_world
                        for joint in body.findall("joint[@type='hinge']"):
                            joint.set("range", property_numbers([cfg.hinge_min, cfg.hinge_max]))
                            joint.set("ref", property_numbers([cfg.hinge_start]))
                    for geom in body.findall("geom"):
                        for field in ("mesh", "material"):
                            if geom.get(field) in rename:
                                geom.set(field, rename[geom.get(field)])
                    if owner:
                        relative = _rigid(owner.matrix_world, scene.unit_settings.scale_length).inverted() @ _rigid(obj.matrix_world, scene.unit_settings.scale_length)
                        for geom in list(body.findall("geom")):
                            geom.attrib.update(_pose(relative @ _matrix(geom.attrib)))
                            if geom.get("contype") != "0":
                                geom.set("contype", contype)
                                geom.set("conaffinity", conaffinity)
                                geom.set("mass", "0")
                            target.append(geom)
                    else:
                        target.append(body)
            warnings.extend(w for w in report["warnings"] if "parent transforms were baked" not in w)
            return details
    finally:
        scene.mujoco_world.asset_directory = asset_directory
        scene.mujoco_world.selected_only = only
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active


def _primitive_size(obj, attrs):
    """Scale an adopted analytic primitive without replacing hollow compounds."""
    from .exporter import ExportError, numbers
    kind = attrs.get("type", "sphere")
    if kind in {"mesh", "plane"}:
        return attrs.get("size")
    axes = [obj.matrix_world.col[i].to_3d().normalized() for i in range(3)]
    if max(abs(axes[i].dot(axes[j])) for i in range(3) for j in range(i + 1, 3)) > 1e-5:
        raise ExportError(f"{obj.name}: an analytic {kind} cannot represent parent-induced shear. Apply the parent scale, or use separate convex collision helpers.")
    old = Vector(obj.get("mjworld_mjcf_original_scale", (1, 1, 1)))
    current_scale = _scale(obj.matrix_world)
    ratio = Vector(tuple(abs(current_scale[i] / old[i]) if old[i] else 1 for i in range(3)))
    original_bounds = _json(obj, "mjworld_mjcf_original_bounds")
    if original_bounds:
        for i in range(3):
            span = max(v[i] for v in original_bounds) - min(v[i] for v in original_bounds)
            current = max(v[i] for v in obj.bound_box) - min(v[i] for v in obj.bound_box)
            if span:
                ratio[i] *= current / span
    if max(abs(ratio[i] - 1) for i in range(3)) < 2e-6:
        return attrs.get("size")
    size = _full_size(attrs)
    if kind in {"box", "ellipsoid"}:
        size = [v * ratio[i] for i, v in enumerate(size)]
    elif kind in {"cylinder", "capsule"}:
        if abs(ratio.x - ratio.y) > 1e-5:
            raise ExportError(f"{obj.name}: an analytic {kind} needs equal X/Y scaling. Use a convex collision helper for a distorted shape.")
        size = [size[0] * ratio.x, size[1] * ratio.z]
    elif kind == "sphere":
        if max(ratio) - min(ratio) > 1e-5:
            raise ExportError(f"{obj.name}: sphere collision needs uniform scale. Use an ellipsoid or convex helper.")
        size = [size[0] * ratio.x]
    return numbers(size)


def _default_chains(root):
    """Map each <default> class to its chain of <default> elements, outermost first."""
    chains = {}
    def walk(element, chain):
        chain = chain + [element]
        chains[element.get("class", "main")] = chain
        for child in element.findall("default"):
            walk(child, chain)
    for top in root.findall("default"):
        walk(top, [])
    return chains


def _effective(attrs, chains, class_name, tag, field):
    """An MJCF attribute, explicit or inherited from the element's default class."""
    if field in attrs:
        return attrs[field]
    for default in reversed(chains.get(class_name or "main", chains.get("main", []))):
        element = default.find(tag)
        if element is not None and field in element.attrib:
            return element.get(field)
    return None


# Mesh asset attributes that change physics rather than geometry. The exporter
# rewrites the vertices, so scale and reference frames are not carried.
_MESH_PHYSICS = ("inertia", "maxhullvert", "smoothnormal")


def _mesh_physics(source_mesh, chains):
    """Physics attributes of an adopted mesh asset, including class defaults."""
    if source_mesh is None:
        return {}
    attrs = dict(source_mesh.attrib)
    values = {field: _effective(attrs, chains, attrs.get("class", "main"), "mesh", field) for field in _MESH_PHYSICS}
    return {field: value for field, value in values.items() if value is not None}


def _contributes_inertia(attrs, chains, class_name, compiler):
    """Whether a geom adds mass and inertia to its body."""
    def value(field):
        return _effective(attrs, chains, class_name, "geom", field)
    if compiler.get("inertiafromgeom") == "false":
        return False
    try:
        if value("mass") is not None:
            return float(value("mass")) > 0
        if value("density") is not None and float(value("density")) == 0:
            return False
        low, high = (int(v) for v in compiler.get("inertiagrouprange", "0 5").split())
        return low <= int(value("group") or 0) <= high
    except ValueError:
        return True


def _collides(attrs, chains, class_name):
    """Whether a geom takes part in contacts (contype or conaffinity non-zero)."""
    try:
        return any(int(_effective(attrs, chains, class_name, "geom", field) or 1) for field in ("contype", "conaffinity"))
    except ValueError:
        return True


def _adopted_copies(bodies, geoms, root):
    """Fresh MJCF names for copies of adopted objects.

    Duplicating an adopted object (Shift+D) also copies its adopted physics,
    including the MJCF name. Where several objects claim one name, the object
    adopted under it keeps it: its ``mjworld_mjcf_owner`` (the Blender name at
    adoption) matches its current name or, in projects adopted before that
    property existed, its Blender name is the MJCF name. Every other claimant
    is a new copy with a fresh name. Returns ``({obj: name}, unique)``, where
    ``unique(base)`` reserves another unused name.
    """
    from .exporter import ExportError, slug
    taken = {element.get("name") for element in root.iter() if element.get("name")}
    def unique(base):
        candidate, number = base, 1
        while candidate in taken:
            candidate, number = f"{base}_{number}", number + 1
        taken.add(candidate)
        return candidate
    claims = {}
    for kind, key, objects in (("body", "mjworld_mjcf_body", bodies), ("geom", "mjworld_mjcf_geom", geoms)):
        for obj in objects:
            name = _json(obj, key, {}).get("name")
            if name:
                claims.setdefault((kind, name), []).append(obj)
    fresh = {}
    for (kind, name), claimants in sorted(claims.items()):
        if len(claimants) == 1:
            continue
        owners = [obj for obj in claimants if obj.get("mjworld_mjcf_owner", name) == obj.name]
        if len(owners) != 1:
            raise ExportError(f"Several objects carry the adopted MJCF {kind} '{name}': " +
                              ", ".join(sorted(obj.name for obj in claimants)[:10]) +
                              ". Give the original object its adopted name again (or adopt the world again); "
                              "the others then export as new copies.")
        for obj in sorted(claimants, key=lambda o: o.name):
            if obj is not owners[0]:
                fresh[obj] = unique(slug(obj.name))
    return fresh, unique


def export_adopted(context, filepath):
    """Export retained physics and live Blender geometry as a portable world.

    Everything is validated and built in memory first; asset files are staged
    or placed through an AssetWriter, and the XML and report replace the
    previous ones only at the end, so a failed export keeps the last good one.
    """
    from .exporter import AssetWriter, ExportError, _asset_paths, write_text, write_xml, xml_path
    if context.mode != "OBJECT":
        raise ExportError("Switch to Object Mode before exporting.")
    scene = context.scene
    settings = scene.mujoco_world
    if settings.selected_only:
        raise ExportError("An adopted physics project exports the complete world. Turn off 'Export selected objects only' so joint, contact and keyframe references remain valid.")
    path = xml_path(filepath)
    unit = float(scene.unit_settings.scale_length)
    if unit <= 0:
        raise ExportError("Scene unit scale must be positive.")
    objects = sorted({o for o in scene.objects if o.type == "MESH" and not o.get("mjworld_adopted_exclude") and _included(o) and (o.get("mjworld_mjcf_geom") or o.get("mjworld_baked_atlas") or (o.visible_get() and not o.hide_render))}, key=lambda o: o.name)
    if not objects:
        raise ExportError("No mesh objects selected for export.")
    adopted_bodies = [o for o in scene.objects if is_body(o) and o.get("mjworld_mjcf_body") and _included(o)]
    owners = {owner for obj in objects for owner in [body_owner(obj)] if owner}
    check_evaluated(context, objects + sorted(owners | set(adopted_bodies), key=lambda o: o.name))
    root = ET.fromstring(scene["mjworld_mjcf_template"])
    def skipped(obj):
        # Bake sources that only provided appearance are replaced by the atlas.
        return bool(obj.get("mjworld_bake_source") or obj.get("preserved_original_static_visual")) and obj.mujoco_world.role == "VISUAL"
    fresh, unique = _adopted_copies(adopted_bodies, [o for o in objects if o.get("mjworld_mjcf_geom") and not skipped(o)], root)
    chains = _default_chains(root)
    source = _resolved_path(scene["mjworld_mjcf_source"])
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    source_meshdir, source_texturedir = compiler.get("meshdir", ""), compiler.get("texturedir", "")
    compiler.attrib.pop("meshdir", None)
    compiler.attrib.pop("texturedir", None)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    warnings = []
    path.parent.mkdir(parents=True, exist_ok=True)
    assets_path, asset_prefix, shared_assets = _asset_paths(scene, path)
    assets = AssetWriter(assets_path, shared_assets)
    try:
        report = _build_adopted(context, path, root, compiler, asset, source, source_meshdir, source_texturedir, objects,
                                adopted_bodies, fresh, unique, chains, unit, assets, asset_prefix, warnings)
        assets.commit()
        write_xml(root, path)
        write_text(path.parent / "export_report.json", json.dumps(report, indent=2) + "\n")
    finally:
        assets.close()
    return report


def _build_adopted(context, path, root, compiler, asset, source, source_meshdir, source_texturedir, objects,
                   adopted_bodies, fresh, unique, chains, unit, assets, asset_prefix, warnings):
    """Build the adopted world in memory; asset files go through ``assets``."""
    from .exporter import (ExportError, _extract_geometry, _write_obj, _write_image, check_door, numbers,
                           inspect_material, property_numbers, property_value, slug)
    scene = context.scene
    settings = scene.mujoco_world
    # Preserve original material/solver/default declarations. Asset files are
    # copied into one sibling directory so exports never depend on author paths.
    # Every referenced mesh is exported from current Blender geometry below;
    # carrying the old files too would duplicate task meshes on every export.
    # The source declarations still supply each mesh's physics attributes.
    def source_file(element):
        filename = element.get("file")
        directory = source_meshdir if element.tag == "mesh" else source_texturedir
        return Path(filename) if Path(filename).is_absolute() else source.parent / directory / filename
    # An export may target the source world's own folder; its files are never removed.
    for element in asset:
        if element.get("file"):
            assets.protect(source_file(element))
    source_meshes = {element.get("name"): element for element in asset.findall("mesh")}
    for element in list(asset.findall("mesh")):
        asset.remove(element)
    for index, element in enumerate(asset):
        filename = element.get("file")
        if filename:
            original = source_file(element)
            if not original.exists():
                raise ExportError(f"Missing adopted asset: {original}. Keep the source world assets beside the saved project.")
            name = slug(element.get("name", f"asset_{index}")) + Path(filename).suffix.lower()
            element.set("file", asset_prefix + "/" + assets.copy(original, name))
    original_world = root.find("worldbody")
    world = ET.Element("worldbody", original_world.attrib)
    # Retain lights/cameras/sites and world-level physics declarations. Bodies
    # and geometry are reconstructed from the current Blender object hierarchy.
    for element in original_world:
        if element.tag not in {"body", "geom"}:
            world.append(copy.deepcopy(element))
    root.remove(original_world)
    root.append(world)
    depsgraph = context.evaluated_depsgraph_get()
    bodies = {}
    # The childclass each body passes on to its geoms, for default lookups.
    childclass = {world: None}
    angle_factor = math.pi / 180 if compiler.get("angle", "degree") == "radian" else 1.0
    def door_angles(cfg):
        """Door limits and starting angle in the adopted world's angle unit."""
        return [property_value(v) * angle_factor for v in (cfg.hinge_min, cfg.hinge_max, cfg.hinge_start)]
    def make_body(obj):
        if obj in bodies:
            return bodies[obj]
        owner = body_owner(obj)
        parent = make_body(owner) if owner else world
        attrs = _json(obj, "mjworld_mjcf_body", {"name": obj.name})
        transform = _rigid(obj.matrix_world, unit)
        if owner:
            transform = _rigid(owner.matrix_world, unit).inverted() @ transform
        old = _matrix(attrs, compiler)
        # Avoid float32 Blender rounding changing untouched source poses.
        if max(abs(transform[i][j] - old[i][j]) for i in range(4) for j in range(4)) > 2e-6:
            for field in ("euler", "axisangle", "xyaxes", "zaxis"):
                attrs.pop(field, None)
            attrs.update(_pose(transform))
        children = [ET.fromstring(xml) for xml in _json(obj, "mjworld_mjcf_children", [])]
        if obj in fresh:
            # A copy of an adopted body: same physics, new body, joint and site names.
            old_name = attrs.get("name", "")
            attrs["name"] = fresh[obj]
            for element in (node for child in children for node in child.iter()):
                if element.get("name"):
                    name = element.get("name")
                    suffix = name[len(old_name):] if old_name and name.startswith(old_name) else "_" + name
                    element.set("name", unique(fresh[obj] + suffix))
            warnings.append(f"{obj.name}: copy of adopted body '{old_name}', exported as body '{fresh[obj]}'.")
        body = ET.SubElement(parent, "body", **attrs)
        childclass[body] = attrs.get("childclass") or childclass[parent]
        for child in children:
            body.append(child)
        cfg = obj.mujoco_world
        baseline = _json(obj, "mjworld_settings_baseline", {})
        if not obj.get("mjworld_mjcf_body"):
            if obj.mujoco_world.role == "FREE":
                if owner:
                    raise ExportError(f"{obj.name}: a free body must be top-level.")
                ET.SubElement(body, "freejoint", name=obj.name + "_free")
            elif obj.mujoco_world.role == "HINGE":
                check_door(obj)
                axis = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis])
                low, high, start = door_angles(cfg)
                body.set("quat", numbers(transform.to_quaternion() @ Quaternion(axis, math.radians(property_value(cfg.hinge_start)))))
                ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((low, high)), ref=numbers([start]), damping=property_numbers([cfg.damping]))
        else:
            if cfg.role != baseline.get("role", cfg.role):
                for node in list(body):
                    if node.tag in {"joint", "freejoint"}:
                        body.remove(node)
                if cfg.role == "FREE":
                    if owner:
                        raise ExportError(f"{obj.name}: a free body must be top-level.")
                    ET.SubElement(body, "freejoint", name=obj.name + "_free")
                elif cfg.role == "HINGE":
                    # The body keeps its adopted pose at the starting angle, so
                    # the closed pose need not be at 0.
                    check_door(obj, closed_at_zero=False)
                    axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis]
                    low, high, start = door_angles(cfg)
                    ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((low, high)), ref=numbers([start]), damping=property_numbers([cfg.damping]))
            inertial = body.find("inertial")
            if inertial is not None and abs(cfg.mass - baseline.get("mass", cfg.mass)) > 1e-6:
                ratio = property_value(cfg.mass) / float(inertial.get("mass"))
                inertial.set("mass", property_numbers([cfg.mass]))
                for field in ("diaginertia", "fullinertia"):
                    if inertial.get(field):
                        inertial.set(field, numbers(float(v) * ratio for v in inertial.get(field).split()))
            joint = body.find("joint[@type='hinge']")
            if joint is not None and cfg.role == "HINGE":
                if not cfg.hinge_min <= cfg.hinge_start <= cfg.hinge_max:
                    raise ExportError(f"{obj.name}: starting angle must be within the door limits.")
                low, high, start = door_angles(cfg)
                if cfg.hinge_min != baseline.get("hinge_min") or cfg.hinge_max != baseline.get("hinge_max"):
                    # Source joints may legitimately exclude 0, so only order is required.
                    if not low < high:
                        raise ExportError(f"{obj.name}: the open limit must be below the closed limit.")
                    joint.set("range", numbers((low, high)))
                if cfg.damping != baseline.get("damping"):
                    joint.set("damping", property_numbers([cfg.damping]))
                if cfg.hinge_axis != baseline.get("hinge_axis"):
                    joint.set("axis", numbers({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis]))
                if abs(cfg.hinge_start - baseline.get("hinge_start", cfg.hinge_start)) > 1e-5:
                    joint.set("ref", numbers([start]))
                    axis = Vector(tuple(float(v) for v in joint.get("axis", "0 0 1").split()))
                    delta = math.radians(cfg.hinge_start - baseline["hinge_start"])
                    body.set("quat", numbers(transform.to_quaternion() @ Quaternion(axis, delta)))
        bodies[obj] = body
        return body
    # Retain source body/joint indexing, including bodies that only contain
    # children or sensors, so unchanged keyframes keep their original meaning.
    imported_bodies = {_json(o, "mjworld_mjcf_body", {}).get("name"): o for o in adopted_bodies if o not in fresh}
    for definition in original_world.iter("body"):
        obj = imported_bodies.get(definition.get("name"))
        if obj is not None:
            make_body(obj)
    details = []
    mesh_cache = {}
    new_parts = []
    plane_count = 0
    geom_elements = {}
    for index, obj in enumerate(objects):
        attrs = _json(obj, "mjworld_mjcf_geom")
        owner = body_owner(obj)
        target = make_body(owner) if owner else world
        baked_source = obj.get("mjworld_bake_source") or obj.get("preserved_original_static_visual")
        if attrs:
            attrs = dict(attrs)
            kind = attrs.get("type", "sphere")
            if baked_source and obj.mujoco_world.role == "VISUAL":
                continue
            if obj in fresh:
                warnings.append(f"{obj.name}: copy of adopted geom '{attrs.get('name')}', exported as geom '{fresh[obj]}'.")
                attrs["name"] = fresh[obj]
            if baked_source or obj.mujoco_world.role == "COLLIDER":
                attrs.update(group="3", rgba="0 0 0 0")
            transform = _rigid(obj.matrix_world, unit)
            if kind not in {"mesh", "plane"}:
                reference_bounds = _json(obj, "mjworld_mjcf_original_bounds")
                if reference_bounds:
                    reference_center = sum((Vector(v) for v in reference_bounds), Vector()) / 8
                    current_center = sum((Vector(v) for v in obj.bound_box), Vector()) / 8
                    transform.translation += obj.matrix_world.to_3x3() @ (current_center - reference_center) * unit
            if owner:
                transform = _rigid(owner.matrix_world, unit).inverted() @ transform
            size = _primitive_size(obj, attrs)
            segment = bool(attrs.get("fromto")) and kind in _ELONGATED
            if segment:
                # A fromto shape stays verbatim until it moves or is resized; then
                # pos/quat and an explicit full size replace the segment.
                resized = size != attrs.get("size")
                moved = resized or not _same_segment(transform, attrs)
                if moved and not resized:
                    size = numbers(_full_size(attrs))
                if moved:
                    _check_segment_axis(obj, kind, size, unit)
            else:
                old = _matrix(attrs, compiler)
                moved = max(abs(transform[i][j] - old[i][j]) for i in range(4) for j in range(4)) > 2e-6
            if moved:
                for field in ("fromto", "euler", "axisangle", "xyaxes", "zaxis"):
                    attrs.pop(field, None)
                attrs.update(_pose(transform))
            if size:
                attrs["size"] = size
            if kind == "plane":
                plane_count += 1
            elif kind == "mesh":
                geometry = _extract_geometry(obj, depsgraph, unit)
                # Geometry is expressed in its object's rigid rotation frame;
                # meshes retain arbitrary evaluated scale, modifiers and shear.
                triangles = [tri for _material, tri in geometry["triangles"]]
                uv_name = geometry["active_uv"]
                # Share identical evaluated shapes across repeated task props.
                # Rounding only the cache key absorbs Blender float32 rotation
                # noise; the written mesh retains full evaluated precision.
                # The regenerated mesh keeps the source mesh's inertia mode,
                # convex hull limit and normals. Without a source mode, a mesh
                # that adds no mass uses shell, which also accepts open or flat
                # visual meshes; other meshes keep MuJoCo's default.
                physics = _mesh_physics(source_meshes.get(attrs.get("mesh")), chains)
                if "inertia" not in physics and not _contributes_inertia(attrs, chains, attrs.get("class") or childclass[target], compiler):
                    physics["inertia"] = "shell"
                physics = {field: physics[field] for field in _MESH_PHYSICS if field in physics}
                key =hashlib.sha256(json.dumps({"vertices": [[round(v, 5) for v in p] for p in geometry["vertices"]], "triangles": triangles, "uv": geometry["uv_layers"].get(uv_name), "physics": physics}, sort_keys=True).encode()).hexdigest()
                if key not in mesh_cache:
                    # The atlas may be matched to an unnamed source geom.
                    mesh_name = slug(attrs.get("name") or obj.name) + "_authored_mesh"
                    mesh_file = assets.write(mesh_name + ".obj", lambda target_path: _write_obj(target_path, geometry, triangles, uv_name))
                    ET.SubElement(asset, "mesh", name=mesh_name, file=asset_prefix + "/" + mesh_file, **physics)
                    mesh_cache[key] = mesh_name
                mesh_name = mesh_cache[key]
                attrs["mesh"] = mesh_name
            geom_elements[obj] = ET.SubElement(target, "geom", **attrs)
            details.append({"source": obj.name, "body": owner.name if owner else "world", "role": obj.mujoco_world.role, "geometry": kind})
        else:
            # New static visual atlas: material comes from the current packed
            # HDR image, converted through the scene's current view transform.
            if obj.mujoco_world.role != "VISUAL":
                new_parts.append(obj)
                continue
            geometry = _extract_geometry(obj, depsgraph, unit)
            mat = obj.material_slots[0].material if obj.material_slots else None
            info = inspect_material(mat)
            warnings.extend(info["warnings"])
            label = slug(obj.name) + "_" + str(index)
            material_args = {"name": label + "_material", "rgba": numbers(info["color"]), "specular": "0", "emission": property_numbers([settings.baked_emission])}
            image = info["image"]
            if image:
                image_file = assets.write(label + ".png", lambda target_path: _write_image(image, target_path, baked=True, scene=scene))
                ET.SubElement(asset, "texture", name=label + "_texture", type="2d", file=asset_prefix + "/" + image_file, colorspace="linear")
                material_args.update(texture=label + "_texture", rgba="1 1 1 1", texuniform="false")
            ET.SubElement(asset, "material", **material_args)
            mesh_file = assets.write(label + ".obj", lambda target_path: _write_obj(target_path, geometry, [tri for _m, tri in geometry["triangles"]], info.get("uv_map") or geometry["active_uv"]))
            ET.SubElement(asset, "mesh", name=label + "_mesh", file=asset_prefix + "/" + mesh_file, inertia="shell")
            ET.SubElement(target, "geom", name=label, type="mesh", mesh=label + "_mesh", material=material_args["name"], contype="0", conaffinity="0", group="1", mass="0", **_pose(_rigid(obj.matrix_world, unit) if owner is None else _rigid(owner.matrix_world, unit).inverted() @ _rigid(obj.matrix_world, unit)))
            details.append({"source": obj.name, "body": owner.name if owner else "world", "role": "VISUAL", "geometry": "mesh"})
    # Copies of adopted bodies are exported even when they hold no geoms.
    for obj in sorted((o for o in fresh if is_body(o)), key=lambda o: o.name):
        make_body(obj)
    details.extend(_append_new_parts(context, new_parts, root, world, asset, assets, make_body, warnings, asset_prefix))
    if plane_count != 1:
        raise ExportError(f"This adopted world needs exactly one analytic floor plane; found {plane_count}.")
    _apply_weights(bodies, childclass, chains, warnings)
    from .authored_materials import update_authored_materials
    warnings.extend(update_authored_materials(scene, root, assets, asset_prefix, geom_elements))
    # No duplicate legacy meshes/textures in runtime bundles. Defaults can carry
    # material references, so gather references across the entire XML tree.
    mesh_refs = {element.get("mesh") for element in root.iter() if element.get("mesh")}
    material_refs = {element.get("material") for element in root.iter() if element.get("material")}
    for element in list(asset):
        if (element.tag == "mesh" and element.get("name") not in mesh_refs) or (element.tag == "material" and element.get("name") not in material_refs):
            asset.remove(element)
            if element.get("file"):
                assets.discard(path.parent / element.get("file"))
    texture_refs = {element.get("texture") for element in asset if element.get("texture")}
    for element in list(asset.findall("texture")):
        if element.get("name") not in texture_refs and element.get("type") != "skybox":
            asset.remove(element)
            if element.get("file"):
                assets.discard(path.parent / element.get("file"))
    return {"addon_version": "0.2.0", "blender_version": bpy.app.version_string, "file": path.name, "objects": len(details), "bodies": len(bodies), "meshes": len(asset.findall("mesh")), "textures": len(asset.findall("texture")), "triangles": sum(len(o.data.polygons) for o in objects), "adopted_physics": True, "metres_per_blender_unit": unit, "object_details": details, "warnings": list(dict.fromkeys(warnings)), "plane": {"enabled": True, "name": "floor", "count": plane_count}}


def _apply_weights(bodies, childclass, chains, warnings):
    """Give movable bodies their Weight where the source inertia no longer applies.

    New Movable/Door bodies, and adopted bodies without <inertial> whose role
    or Weight changed, get the Weight split equally over their colliding geoms;
    their other geoms get mass 0. Untouched adopted bodies keep their source
    physics, and bodies with <inertial> are rescaled in make_body.
    """
    from .exporter import ExportError, numbers, property_value
    for obj, body in bodies.items():
        cfg = obj.mujoco_world
        if cfg.role not in {"FREE", "HINGE"}:
            continue
        adopted = bool(obj.get("mjworld_mjcf_body"))
        if adopted:
            baseline = _json(obj, "mjworld_settings_baseline", {})
            changed = cfg.role != baseline.get("role", cfg.role) or abs(cfg.mass - baseline.get("mass", cfg.mass)) > 1e-6
            if body.find("inertial") is not None or not changed:
                continue
        geoms = body.findall("geom")
        colliders = [geom for geom in geoms if _collides(geom.attrib, chains, geom.get("class") or childclass[body])]
        if not colliders:
            if not adopted:
                raise ExportError(f"{obj.name}: add at least one collision helper before making this body movable.")
            warnings.append(f"{obj.name}: its Weight was not applied because the body has no colliding geoms of its own.")
            continue
        share = numbers([property_value(cfg.mass) / len(colliders)])
        for geom in geoms:
            geom.set("mass", share if geom in colliders else "0")
