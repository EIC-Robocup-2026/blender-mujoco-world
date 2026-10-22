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
import numpy as np


def is_body(obj):
    return obj.type == "EMPTY" and (obj.mujoco_world.body or obj.get("simulation_body", False))


def editable(obj):
    """Whether edits to ``obj`` are kept: not linked from a library (overrides are editable)."""
    return obj.library is None and obj.is_editable


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
# Attributes naming asset files: one file, or the six faces of a cube texture.
_FILE_FIELDS = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")
# Source files an adopted export copies, by element, with the suffixes MuJoCo
# reads for it. Source mesh files are never read or copied: every mesh is
# regenerated from Blender geometry, so their paths are only protected.
_COPIED_SUFFIXES = {"texture": (".png",), "hfield": (".png", ".bin"), "skin": (".skn",)}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_OUTSIDE_SETTING = "Allow adopted assets outside the MJCF folder"


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


def _source_path(scene):
    """The adopted MJCF file: the blend-relative path, else the absolute one stored beside it.

    The relative path is a plain string that Save As does not remap, so a blend
    saved elsewhere falls back to the absolute path recorded at adoption.
    """
    relative = _resolved_path(scene["mjworld_mjcf_source"])
    absolute = scene.get("mjworld_mjcf_source_absolute")
    if not relative.exists() and absolute and Path(absolute).exists():
        return Path(absolute)
    return relative


# ID properties that adoption writes on objects; a new adoption clears them first.
_ADOPTION_KEYS = ("mjworld_mjcf_body", "mjworld_mjcf_geom", "mjworld_mjcf_children", "mjworld_mjcf_original_scale",
                  "mjworld_mjcf_original_bounds", "mjworld_settings_baseline", "mjworld_mjcf_owner",
                  "mjworld_mjcf_geom_offset")


def _element_classes(worldbody):
    """Default class and enclosing body of every element below an MJCF worldbody.

    ``classes[element]`` is the class the element uses: its own ``class``, else
    the nearest ancestor body's ``childclass``, else ``main``. A body has no
    class of its own, so for a body it is the class inherited from its parent.
    ``parents[element]`` is the enclosing body element, or None in the worldbody.
    """
    classes, parents = {}, {}
    def walk(element, inherited, body):
        for child in element:
            classes[child] = child.get("class") or inherited
            parents[child] = body
            walk(child, child.get("childclass") or inherited, child if child.tag == "body" else body)
    walk(worldbody, "main", None)
    return classes, parents


# Geom attributes read through the default classes for classification and sizing.
_RESOLVED = ("type", "size", "mesh", "rgba", "contype", "conaffinity")


def _resolved_geom(attrs, chains, class_name):
    """Geom attributes completed with the type, size and contact fields its default class sets.

    For reading only: exports keep the source attributes and write a size only
    when it changes.
    """
    resolved = dict(attrs)
    for field in _RESOLVED:
        if field not in resolved:
            value = _effective(attrs, chains, class_name, "geom", field)
            if value is not None:
                resolved[field] = value
    return resolved


def _joint_type(joint, chains, class_name):
    """A joint's type, explicit or from its default class; MuJoCo's default type is hinge."""
    return _effective(joint.attrib, chains, class_name, "joint", "type") or "hinge"


def _origin_offset(obj, owner, attrs, compiler, unit):
    """An adopted primitive's centre in its object's local coordinates, if the origin is elsewhere.

    MuJoCo places a primitive at its geom frame, while Blender objects are
    often modelled with the origin at their base. The export adds this offset
    to the object's pose, so an untouched geom keeps its source position.
    Returns None when the origin is within 10 micrometres of the geom centre.
    Computed in double precision: Blender matrices are float32.
    """
    if attrs.get("fromto"):
        points = _floats(attrs, "fromto", 6)
        position = [(points[i] + points[i + 3]) / 2 for i in range(3)]
    else:
        position = _floats(attrs, "pos", 3) if attrs.get("pos") else [0.0, 0.0, 0.0]
    body = np.array(_rigid(owner.matrix_world, unit) if owner is not None else Matrix.Identity(4), dtype=float)
    centre = body @ np.array(position + [1.0])
    matrix = np.array(obj.matrix_world, dtype=float)
    if np.linalg.norm(centre[:3] - matrix[:3, 3] * unit) <= 1e-5:
        return None
    return [float(v) for v in np.linalg.solve(matrix, np.append(centre[:3] / unit, 1.0))[:3]]


def adopt_mjcf(context, filepath, *, object_map=None, body_map=None, geom_map=None):
    """Attach exact MJCF physics to an existing Blender world by object name.

    ``object_map`` optionally maps MJCF names to Blender object names, for
    bodies and geoms alike. ``body_map`` and ``geom_map`` map one kind only and
    take precedence, so a body and a geom sharing an MJCF name can match
    different objects. Unnamed geoms cannot be matched, except a lone unnamed
    world geom on a baked atlas. Two MJCF elements matching one object are
    refused. No mesh, material, object transform, camera, light or existing
    user edit is changed. Source assets remain relative to the blend file where
    possible; the absolute source path is kept as a fallback for a moved blend.
    Returns counts and ``warnings`` (geoms whose Blender object sits under a
    different body than in the MJCF).
    """
    from .exporter import ExportError
    filepath = _resolved_path(filepath)
    try:
        root = ET.parse(filepath).getroot()
    except ET.ParseError as exc:
        raise ExportError(f"The chosen file is not valid MJCF XML: {exc}") from exc
    if root.tag != "mujoco" or root.find("worldbody") is None:
        raise ExportError("Choose a standalone MJCF world containing <worldbody>.")
    if next(root.iter("include"), None) is not None:
        raise ExportError("Adoption needs a standalone XML. Expand MJCF includes before adopting this world.")
    for tag in ("frame", "replicate"):
        if next(root.iter(tag), None) is not None:
            raise ExportError(f"Adoption does not support <{tag}>: its contents would be exported twice. "
                              "Flatten frames and replicated bodies into ordinary bodies and geoms before adopting.")
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("coordinate", "local") != "local":
        raise ExportError("Adoption supports local MJCF coordinates only.")
    # Inertia, joint limits, contact defaults and solver options are preserved in
    # their original units. Object poses are written as quaternions on export.
    world_element = root.find("worldbody")
    chains = _default_chains(root)
    classes, parents = _element_classes(world_element)
    scene = context.scene
    unit = float(scene.unit_settings.scale_length)
    candidates = {obj.name: obj for obj in scene.objects}
    def find(name, kind_map):
        for mapping in (kind_map or {}, object_map or {}):
            if name in mapping:
                return candidates.get(mapping[name])
        return candidates.get(name)
    claims, duplicates = {}, []
    def claim(obj, label):
        previous = claims.setdefault(obj, label)
        if previous != label:
            duplicates.append(f"{previous} and {label} both match '{obj.name}'")
    # First resolve and check everything, then change the scene, so a rejected
    # adoption leaves the previous adoption and every setting untouched.
    missing_bodies, missing_geoms = [], []
    body_plans, geom_plans = [], []
    body_names = {}
    for source in root.findall("./worldbody//body"):
        obj = find(source.get("name", ""), body_map)
        if obj is None or obj.type != "EMPTY":
            missing_bodies.append(source.get("name", "<unnamed>"))
            continue
        name = source.get("name", "<unnamed>")
        claim(obj, f"body '{name}'")
        body_names[obj] = source.get("name")
        joints = source.findall("joint")
        kinds = [_joint_type(joint, chains, classes[joint]) for joint in joints]
        # Slide and ball joints are kept as they are; the role marks the body as moving.
        values = {"role": "FREE" if source.find("freejoint") is not None or "free" in kinds else "HINGE" if joints else "STATIC"}
        try:
            inertial = source.find("inertial")
            if inertial is not None and inertial.get("mass"):
                values["mass"] = float(inertial.get("mass"))
            joint = next((joint for joint, kind in zip(joints, kinds) if kind == "hinge"), None)
            if joint is not None:
                def field(key, fallback=None, joint=joint):
                    return _effective(joint.attrib, chains, classes[joint], "joint", key) or fallback
                degree = compiler is None or compiler.get("angle", "degree") == "degree"
                conversion = 1.0 if degree else 180.0 / math.pi
                # The fallback is the add-on's default door, already in degrees.
                limits = [float(v) * conversion for v in field("range").split()] if field("range") else [-110.0, 0.0]
                if len(limits) != 2:
                    raise ExportError(f"Body '{name}': its hinge range needs two numbers.")
                values["hinge_min"], values["hinge_max"] = limits
                values["hinge_start"] = float(field("ref", "0")) * conversion
                values["damping"] = float(field("damping", "0"))
                axis = [float(v) for v in field("axis", "0 0 1").split()]
                if len(axis) != 3:
                    raise ExportError(f"Body '{name}': its hinge axis needs three numbers.")
                values["hinge_axis"] = "XYZ"[max(range(3), key=lambda i: abs(axis[i]))]
        except ValueError as exc:
            if isinstance(exc, ExportError):
                raise
            raise ExportError(f"Body '{name}': its inertial or joint attributes must contain numbers.") from exc
        children = [ET.tostring(node, encoding="unicode") for node in source if node.tag not in {"body", "geom"}]
        body_plans.append((obj, dict(source.attrib), children, values))
    def planned_owner(obj):
        """body_owner() as it will be once the planned bodies are marked."""
        parent = obj.parent
        while parent:
            if parent in body_names or is_body(parent):
                return parent
            parent = parent.parent
        return None
    def describe(name):
        return "the world body" if name is None else f"body '{name}'"
    notes = []
    world_geoms = list(world_element.findall("geom"))
    materials = {material.get("name"): material for material in root.iterfind("./asset/material")}
    for source in root.findall("./worldbody//geom"):
        obj = find(source.get("name", ""), geom_map)
        # A joined lighting atlas often has a descriptive Blender object name.
        if obj is None and source in world_geoms:
            atlases = [o for o in scene.objects if o.type == "MESH" and (o.get("baked_static_visual") or o.get("mjworld_baked_atlas"))]
            unmatched = [g for g in world_geoms if find(g.get("name", ""), geom_map) is None]
            if len(atlases) == len(unmatched) == 1:
                obj = atlases[0]
        if obj is None or obj.type != "MESH":
            missing_geoms.append(source.get("name", "<unnamed>"))
            continue
        label = source.get("name", "<unnamed>")
        claim(obj, f"geom '{label}'")
        attrs = dict(source.attrib)
        resolved = _resolved_geom(attrs, chains, classes[source])
        scale = list(_scale(obj.matrix_world))
        bounds = [list(v) for v in obj.bound_box]
        # Legacy primitive authoring metadata identifies unit box/cylinder/
        # sphere meshes. Use their source dimensions as the baseline so manual
        # geometry/scale edits made before the first adoption are preserved too.
        kind = resolved.get("type", "sphere")
        size = None
        if obj.get("mujoco_type") == kind and kind not in {"mesh", "plane"}:
            try:
                size = _full_size(resolved)
            except ExportError:
                # No size, even from a <default> class: measure the object instead.
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
        collision_only = obj.get("collision_only") or resolved.get("rgba", "").split()[-1:] == ["0"]
        visual_only = obj.get("visual_only") or (resolved.get("contype") == resolved.get("conaffinity") == "0") or "visual" in classes[source].lower()
        if obj.get("preserved_original_static_visual"):
            role = "VISUAL" if visual_only else "STATIC"
        else:
            role = "COLLIDER" if collision_only else "VISUAL" if visual_only else "STATIC"
        emission = None
        if obj.get("baked_static_visual") or obj.get("mjworld_baked_atlas"):
            material = materials.get(attrs.get("material"))
            if material is not None:
                try:
                    emission = float(material.get("emission", ".2"))
                except ValueError as exc:
                    raise ExportError(f"Material '{attrs.get('material')}': emission must be a number.") from exc
        # Export takes body membership from the Blender hierarchy; say where it differs.
        owner = planned_owner(obj)
        current = body_names.get(owner, owner.name if owner is not None else None)
        expected = parents[source].get("name", "<unnamed>") if parents[source] is not None else None
        same_parent = current == expected and (owner is None or owner in body_names)
        if not same_parent:
            notes.append(f"{obj.name}: MJCF geom '{label}' belongs to {describe(expected)}, but its Blender object is under "
                         f"{describe(current)}; the export follows Blender.")
        # Measured objects keep their geom centre when the origin is elsewhere;
        # legacy unit primitives are centred, and their bounds already say so.
        offset = None
        if same_parent and size is None and kind not in {"mesh", "plane"}:
            offset = _origin_offset(obj, owner, attrs, compiler, unit)
        geom_plans.append((obj, attrs, scale, bounds, role, emission, offset))
    if missing_bodies or missing_geoms:
        # A partial adoption can silently erase collision geometry. Do not allow
        # it to become the active project; report every missing physics source.
        raise ExportError("MJCF names were not found in this Blender scene. Bodies: " + ", ".join(missing_bodies[:10]) + "; geoms: " + ", ".join(missing_geoms[:10]) + ". Use object_map, body_map or geom_map for renamed objects.")
    if duplicates:
        # The second element would overwrite the first one's physics on the object.
        raise ExportError("Several MJCF elements match one Blender object: " + "; ".join(duplicates[:10]) +
                          ". Map them to separate objects with object_map, body_map or geom_map.")
    for obj in scene.objects:
        for key in _ADOPTION_KEYS:
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
    for obj, attrs, scale, bounds, role, emission, offset in geom_plans:
        obj["mjworld_mjcf_geom"] = json.dumps(attrs)
        obj["mjworld_mjcf_owner"] = obj.name
        obj["mjworld_adopted_exclude"] = False
        obj["mjworld_mjcf_original_scale"] = scale
        obj["mjworld_mjcf_original_bounds"] = json.dumps(bounds)
        if offset is not None:
            obj["mjworld_mjcf_geom_offset"] = offset
        # Explicit helpers remain exportable even inside hidden collections.
        if obj.get("preserved_original_static_visual"):
            obj["mjworld_bake_source"] = True
        obj.mujoco_world.role = role
        if emission is not None:
            scene.mujoco_world.baked_emission = emission
    scene["mjworld_mjcf_template"] = ET.tostring(root, encoding="unicode")
    scene["mjworld_mjcf_source"] = bpy.path.relpath(str(filepath)) if bpy.data.filepath else str(filepath)
    scene["mjworld_mjcf_source_absolute"] = str(filepath)
    scene["mjworld_mjcf_adoption_version"] = 1
    from .authored_materials import capture_material_baselines
    capture_material_baselines(scene, root)
    return {"bodies": len(body_plans), "geoms": len(geom_plans), "source": str(filepath), "warnings": notes}


def group_selected(context, role="FREE"):
    """Parent selected parts to one editable rigid body without moving them."""
    from .exporter import ExportError
    objects = [o for o in context.selected_objects if o.type in {"MESH", "EMPTY"}]
    if not objects:
        raise ExportError("Select the visible parts and collision helpers to group.")
    # Linked objects cannot be reparented: the change is lost when the file reloads.
    objects = [o for o in objects if editable(o)]
    if not objects:
        raise ExportError("The selected parts are linked from another file. Make a library override first.")
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
    # One lookup table instead of a search per part keeps large scenes linear.
    exported = {body.get("name"): body for body in world.findall("body")}
    depsgraph = context.evaluated_depsgraph_get()
    for owner, parts in groups.items():
        target = make(owner)
        inverse = _rigid(owner.matrix_world, unit).inverted()
        colliders = [d for _o, d in parts if d["collision"]]
        if owner.mujoco_world.role in {"FREE", "HINGE"} and not colliders:
            raise ExportError(f"{owner.name}: add at least one collision helper before making this body movable.")
        weighted = []
        for obj, detail in parts:
            original = exported.get(detail["body"])
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
                    weighted.append((geom, obj))
                target.append(geom)
            world.remove(original)
            detail["body"] = owner.name
            detail["body_role"] = owner.mujoco_world.role
        if owner.mujoco_world.role in {"FREE", "HINGE"}:
            # Uniform density over the colliders, so the centre of mass and
            # inertia follow where the body's bulk is.
            volumes = [_geom_volume(geom.attrib, {}, None, obj, depsgraph, unit) for geom, obj in weighted]
            for (geom, _obj), mass in zip(weighted, _split_weight(property_value(owner.mujoco_world.mass), volumes)):
                geom.set("mass", numbers([mass]))
        else:
            for geom, _obj in weighted:
                geom.set("mass", "0")
    report["compound_bodies"] = len(groups)
    grouped = {obj.name + ": parent transforms" for parts in groups.values() for obj, _d in parts}
    report["warnings"] = [w for w in report["warnings"] if w.split(" were baked", 1)[0] not in grouped]
    return report


def _append_new_parts(context, objects, root, world, asset, assets, make_body, warnings, asset_prefix, geom_sources):
    """Use the ordinary exporter for newly added furniture and compound props.

    Their files go through ``assets``, the adopted export's AssetWriter. Parts
    grouped under a body get mass 0 here; export_adopted distributes the
    body's Weight afterwards when the body needs it, using ``geom_sources``,
    which this fills with each grouped geom's Blender object.
    """
    from .exporter import _export_simple, property_numbers, role_masks, slug
    if not objects:
        return []
    scene = context.scene
    asset_directory = scene.mujoco_world.asset_directory
    try:
        scene.mujoco_world.asset_directory = ""
        with tempfile.TemporaryDirectory(prefix="mjworld-parts-") as temporary:
            path = Path(temporary) / "parts.xml"
            # An explicit object list also reaches hidden bake sources, which no selection holds.
            report = _export_simple(context, str(path), candidates=objects)
            parts_root = ET.parse(path).getroot()
            details = report["object_details"]
            groups = {}
            for detail in details:
                obj = scene.objects.get(detail["source"])
                owner = body_owner(obj)
                groups.setdefault(owner, []).append((obj, detail))
            exported = {body.get("name"): body for body in parts_root.find("worldbody").findall("body")}
            # Checked before the new parts' own meshes and textures are copied. The
            # adopted world's files are already written by now, so a refusal can
            # still leave those in a shared folder.
            _check_part_names(world, groups, exported)
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
            for owner, members in groups.items():
                target = make_body(owner) if owner else world
                if owner:
                    contype, conaffinity = role_masks(scene.mujoco_world, moving_role(owner))
                for obj, detail in members:
                    body = exported[detail["body"]]
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
                            geom_sources[geom] = obj
                            target.append(geom)
                    else:
                        target.append(body)
            warnings.extend(w for w in report["warnings"] if "parent transforms were baked" not in w)
            return details
    finally:
        scene.mujoco_world.asset_directory = asset_directory


# MuJoCo name namespaces of the elements new parts add; a joint and a free joint share one.
_NAME_KINDS = {"body": "body", "geom": "geom", "joint": "joint", "freejoint": "joint", "site": "site"}


def _check_part_names(world, groups, exported):
    """Refuse new parts whose body, geom, joint or site names the adopted world already uses.

    ``groups`` maps each owner body (None for top-level parts) to its
    ``(object, detail)`` pairs, and ``exported`` the part export's bodies by
    name. A grouped part contributes only its geoms to the owner's body.
    MuJoCo rejects a repeated name, so the export names the objects instead.
    """
    taken = {kind: set() for kind in _NAME_KINDS.values()}
    for element in world.iter():
        if element.tag in _NAME_KINDS and element.get("name"):
            taken[_NAME_KINDS[element.tag]].add(element.get("name"))
    clashes = []
    for owner, members in groups.items():
        for obj, detail in members:
            body = exported[detail["body"]]
            for element in (body.findall("geom") if owner else body.iter()):
                kind, name = _NAME_KINDS.get(element.tag), element.get("name")
                if kind is None or not name:
                    continue
                if name in taken[kind]:
                    clashes.append(f"'{obj.name}' ({element.tag} '{name}')")
                taken[kind].add(name)
    if clashes:
        raise _name_clash_error(clashes)


def _check_generated_names(world, generated):
    """Refuse bodies and joints named after Blender objects when the world already uses the name.

    make_body names a new body group's body after its Empty, and the free
    joint or hinge it adds to a new body, or to an adopted body whose role
    changed, after the object (``<name>_free``, ``<name>_hinge``). A renamed
    adopted object keeps its MJCF names, so another object can take its old
    Blender name. ``generated`` lists those ``(object, element)`` pairs;
    ``world`` is the complete worldbody before new parts are added.
    """
    counts = {}
    for element in world.iter():
        if element.tag in _NAME_KINDS and element.get("name"):
            key = _NAME_KINDS[element.tag], element.get("name")
            counts[key] = counts.get(key, 0) + 1
    clashes = [f"'{obj.name}' ({element.tag} '{element.get('name')}')" for obj, element in generated
               if counts.get((_NAME_KINDS[element.tag], element.get("name")), 0) > 1]
    if clashes:
        raise _name_clash_error(clashes)


def _name_clash_error(clashes):
    from .exporter import ExportError
    return ExportError("Objects would repeat names the adopted world already uses, which MuJoCo rejects: "
                       + ", ".join(clashes) + ". Rename these objects, then export again.")


def _primitive_size(obj, attrs, resolved=None):
    """Scale an adopted analytic primitive without replacing hollow compounds.

    ``resolved`` is ``attrs`` completed from the default classes; it supplies
    the type and size when the geom inherits them. Returns the geom's own size
    attribute (possibly None) while the object is unscaled.
    """
    from .exporter import ExportError, numbers
    resolved = attrs if resolved is None else resolved
    kind = resolved.get("type", "sphere")
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
    size = _full_size(resolved)
    if kind in {"box", "ellipsoid"}:
        size = [v * ratio[i] for i, v in enumerate(size)]
    elif kind in {"cylinder", "capsule"}:
        if abs(ratio.x - ratio.y) > 1e-5:
            raise ExportError(f"{obj.name}: an analytic {kind} needs equal X/Y scaling. Use a convex collision helper for a distorted shape.")
        if kind == "cylinder":
            size = [size[0] * ratio.x, size[1] * ratio.z]
        else:
            # The capsule's caps add its radius to each end, so stretch the
            # overall length and keep the rest as the cylinder part.
            radius = size[0] * ratio.x
            half = ratio.z * (size[0] + size[1]) - radius
            if half <= 0:
                raise ExportError(f"{obj.name}: this capsule is now shorter than it is wide. Scale it along its length, "
                                  "or use a sphere or a convex collision helper.")
            size = [radius, half]
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


def _hull_volume(obj, depsgraph, unit):
    """Volume in cubic metres of the convex hull of an object's evaluated mesh, or None."""
    import bmesh
    if obj is None or obj.type != "MESH":
        return None
    matrix = Matrix.Diagonal((unit, unit, unit, 1.0)) @ obj.matrix_world
    source = bmesh.new()
    try:
        source.from_object(obj, depsgraph)
        points = [matrix @ vertex.co for vertex in source.verts]
    finally:
        source.free()
    if len(points) < 4:
        return None
    hull = bmesh.new()
    try:
        for point in points:
            hull.verts.new(point)
        # Points inside the hull stay loose and add no volume.
        bmesh.ops.convex_hull(hull, input=list(hull.verts))
        return abs(hull.calc_volume())
    finally:
        hull.free()


def _geom_volume(attrs, chains, class_name, obj, depsgraph, unit):
    """A collider's volume in cubic metres, for splitting a body's Weight; None if unknown.

    Primitives use their MJCF size, meshes the convex hull of their Blender
    object. Planes, height fields and unknown shapes return None.
    """
    from .exporter import ExportError
    resolved = _resolved_geom(attrs, chains, class_name)
    kind = resolved.get("type", "sphere")
    if kind == "mesh":
        return _hull_volume(obj, depsgraph, unit)
    try:
        size = _full_size(resolved)
    except ExportError:
        return None
    if kind == "box" and len(size) == 3:
        return 8 * size[0] * size[1] * size[2]
    if kind == "sphere" and len(size) == 1:
        return 4 / 3 * math.pi * size[0] ** 3
    if kind == "ellipsoid" and len(size) == 3:
        return 4 / 3 * math.pi * size[0] * size[1] * size[2]
    if kind == "cylinder" and len(size) == 2:
        return 2 * math.pi * size[0] ** 2 * size[1]
    if kind == "capsule" and len(size) == 2:
        return 2 * math.pi * size[0] ** 2 * size[1] + 4 / 3 * math.pi * size[0] ** 3
    return None


def _mesh_asset_name(base, taken):
    """An unused mesh asset name for a regenerated adopted mesh.

    Keeps the readable ``<slug>_authored_mesh`` while it is free, so existing
    exports keep their file names. A short hash of the full name is added when
    that name is taken ('a b' and 'a_b' meet) or when nothing of the name
    survives slug() (Thai names all become 'object'), keeping it unique and
    stable.
    """
    from .exporter import slug
    name = slug(base) + "_authored_mesh"
    if name in taken or (slug(base) == "object" and base != "object"):
        name = slug(base) + "_" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:8] + "_authored_mesh"
    candidate, number = name, 1
    while candidate in taken:
        candidate, number = f"{name}_{number}", number + 1
    return candidate


def _split_weight(total, volumes):
    """Shares of ``total`` proportional to ``volumes``, as with one uniform density.

    Explicit masses keep the body's total exact even where MuJoCo's own volume
    of a mesh differs slightly from ours. Falls back to equal shares when any
    volume is unknown or the volumes add up to nothing.
    """
    if not volumes:
        return []
    if any(volume is None for volume in volumes) or sum(volumes) <= 0:
        return [total / len(volumes)] * len(volumes)
    whole = sum(volumes)
    return [total * volume / whole for volume in volumes]


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
    from .baking import bake_covered
    from .exporter import (AssetWriter, ExportError, _asset_paths, dropped_object_notes, publish, skip_faceless, xml_bytes,
                           xml_path)
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
    # Hidden bake sources still export their colliders, as in an ordinary export.
    objects = sorted({o for o in scene.objects if o.type == "MESH" and not o.get("mjworld_adopted_exclude") and not o.get("mjworld_bake_clone") and _included(o) and (o.get("mjworld_mjcf_geom") or o.get("mjworld_baked_atlas") or o.get("mjworld_bake_source") or (o.visible_get() and not o.hide_render))}, key=lambda o: o.name)
    # Adopted mesh geoms keep their physics, so only new meshes may be skipped.
    new, faceless_notes = skip_faceless(context, [o for o in objects if not o.get("mjworld_mjcf_geom")])
    objects = [o for o in objects if o.get("mjworld_mjcf_geom") or o in new]
    if not objects:
        raise ExportError("No mesh objects selected for export.")
    adopted_bodies = [o for o in scene.objects if is_body(o) and o.get("mjworld_mjcf_body") and _included(o)]
    owners = {owner for obj in objects for owner in [body_owner(obj)] if owner}
    check_evaluated(context, objects + sorted(owners | set(adopted_bodies), key=lambda o: o.name))
    root = ET.fromstring(scene["mjworld_mjcf_template"])
    # Bake sources whose appearance an atlas provides; see bake_covered().
    covered = bake_covered(scene)
    def skipped(obj):
        # Bake sources that only provided appearance are replaced by the atlas.
        return obj.name in covered and obj.mujoco_world.role == "VISUAL"
    fresh, unique = _adopted_copies(adopted_bodies, [o for o in objects if o.get("mjworld_mjcf_geom") and not skipped(o)], root)
    chains = _default_chains(root)
    source = _source_path(scene)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    # The exported files sit in one asset folder that every path names in full,
    # so the source's directory attributes are resolved here and dropped.
    directories = {field: compiler.attrib.pop(field, None) for field in ("assetdir", "meshdir", "texturedir", "strippath")}
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    warnings = list(faceless_notes)
    # Helpers kept out of the export, and bake sources the atlas replaces, need no note.
    warnings.extend(dropped_object_notes(context, [o for o in scene.objects if not o.get("mjworld_adopted_exclude")
                                                   and not o.get("mjworld_bake_clone") and not skipped(o)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    assets_path, asset_prefix, shared_assets = _asset_paths(scene, path)
    assets = AssetWriter(assets_path, shared_assets)
    try:
        report = _build_adopted(context, path, root, compiler, asset, source, directories, objects,
                                adopted_bodies, fresh, unique, chains, unit, assets, asset_prefix, warnings, covered)
        publish(assets, [(path, xml_bytes(root)),
                         (path.parent / "export_report.json", (json.dumps(report, indent=2) + "\n").encode("utf-8"))])
    finally:
        assets.close()
    return report


def _build_adopted(context, path, root, compiler, asset, source, directories, objects,
                   adopted_bodies, fresh, unique, chains, unit, assets, asset_prefix, warnings, covered):
    """Build the adopted world in memory; asset files go through ``assets``.

    ``directories`` holds the source compiler's assetdir, meshdir, texturedir
    and strippath, already removed from ``compiler``. ``covered`` names the
    bake sources whose appearance an atlas provides.
    """
    from .exporter import (ExportError, _extract_geometry, _write_obj, _write_image, addon_version, check_door, image_problem,
                           numbers, inspect_material, profile_text, property_numbers, property_value, slug, texture_colorspace,
                           uncovered_bake_source)
    scene = context.scene
    settings = scene.mujoco_world
    # Preserve original material/solver/default declarations. Asset files are
    # copied into one sibling directory so exports never depend on author paths.
    # Every referenced mesh is exported from current Blender geometry below;
    # carrying the old files too would duplicate task meshes on every export.
    # The source declarations still supply each mesh's physics attributes.
    # MuJoCo's rules: assetdir sets both meshdir and texturedir unless they are
    # given; meshes, height fields and skins load from meshdir, textures from
    # texturedir; strippath drops the path written in the file attribute.
    assetdir = directories.get("assetdir") or ""
    meshdir = directories["meshdir"] if directories.get("meshdir") is not None else assetdir
    texturedir = directories["texturedir"] if directories.get("texturedir") is not None else assetdir
    strip = (directories.get("strippath") or "false").lower() == "true"
    def source_file(element, field="file"):
        filename = element.get(field)
        if strip:
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if Path(filename).is_absolute():
            return Path(filename)
        directory = meshdir if element.tag in {"mesh", "hfield", "skin"} else texturedir if element.tag == "texture" else ""
        return source.parent / directory / filename
    source_root = source.parent.resolve()
    def checked_file(element, field):
        """The resolved source file ``element`` names in ``field``, once it is safe to copy.

        The MJCF names these paths, and the export copies the files into the
        asset folder, so an absolute or '..' path from someone else's world
        could otherwise copy any local file there. A file must therefore have a
        suffix MuJoCo reads for the element, PNG content when it is a .png,
        and lie, after following links, inside the adopted world's folder or
        the export's asset folder (whose files are already part of the export).
        The scene setting allow_outside_assets allows other folders, with a
        warning per file.
        """
        value = element.get(field)
        label = f"Adopted {element.tag} '{element.get('name', '<unnamed>')}', {field}=\"{value}\""
        suffixes = _COPIED_SUFFIXES.get(element.tag)
        if suffixes is None:
            raise ExportError(f"{label}: an adopted export copies only texture, height field and skin files. "
                              f"Remove this <{element.tag}> from the source MJCF and adopt it again.")
        original = source_file(element, field)
        if original.suffix.lower() not in suffixes:
            raise ExportError(f"{label}: a {element.tag} file must end in {' or '.join(suffixes)}.")
        resolved = original.resolve()
        if not (resolved.is_relative_to(source_root) or resolved.is_relative_to(assets.directory)):
            if not settings.allow_outside_assets:
                raise ExportError(f"{label}: the file {resolved} is outside the adopted world's folder {source_root}. "
                                  "A crafted MJCF can name any local file this way, and the export would copy it into the "
                                  "asset folder. Move the file into the world's folder and adopt it again, or, if you "
                                  f"trust this world, turn on '{_OUTSIDE_SETTING}' under World advanced settings.")
            warnings.append(f"{label}: copied {resolved} from outside the adopted world's folder {source_root}, "
                            f"because '{_OUTSIDE_SETTING}' is on.")
        if not resolved.is_file():
            recorded = scene.get("mjworld_mjcf_source_absolute")
            where = str(source) + (f" (adopted from {recorded})" if recorded and Path(recorded) != source else "")
            raise ExportError(f"Missing adopted asset: {original}. Asset files are read beside the adopted MJCF world, "
                              f"{where}; keep them there, or adopt the world again from where it is now.")
        if original.suffix.lower() == ".png":
            with open(resolved, "rb") as stream:
                if stream.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                    raise ExportError(f"{label}: {resolved} is not a PNG file.")
        return resolved
    file_elements = list(asset) + root.findall("./deformable/skin")
    # Every file is checked before any is copied, so a refused export leaves
    # the asset folder unchanged. An export may target the source world's own
    # folder; its files are never removed.
    copied = {}
    for element in file_elements:
        for field in _FILE_FIELDS:
            if element.get(field):
                assets.protect(source_file(element, field))
                if element.tag != "mesh":
                    copied[element, field] = checked_file(element, field)
    def copy_files(element, fallback):
        for field in _FILE_FIELDS:
            filename = element.get(field)
            if not filename:
                continue
            # Cube texture faces keep the texture's name with the face appended.
            side = "" if field == "file" else "_" + field[len("file"):]
            name = slug(element.get("name", fallback)) + side + Path(filename).suffix.lower()
            element.set(field, asset_prefix + "/" + assets.copy(copied[element, field], name))
        if element.tag == "texture" and element.get("colorspace") is None and any(element.get(f) for f in _FILE_FIELDS):
            # MuJoCo's 'auto' reads the PNG's sRGB chunk, which Blender 5.2
            # and many editors write when the file is saved again. Pin the
            # texture mode instead, so a resaved texture keeps its look.
            element.set("colorspace", texture_colorspace(settings, False))
    source_meshes = {element.get("name"): element for element in asset.findall("mesh")}
    for element in list(asset.findall("mesh")):
        asset.remove(element)
    for index, element in enumerate(asset):
        copy_files(element, f"asset_{index}")
    for index, element in enumerate(root.findall("./deformable/skin")):
        copy_files(element, f"skin_{index}")
    original_world = root.find("worldbody")
    # The default class each source body and geom used, so one that ends up
    # under another body can name it; keyed by MJCF tag and name.
    template_classes, template_parents = _element_classes(original_world)
    source_class = {(element.tag, element.get("name")): name for element, name in template_classes.items()
                    if element.tag in {"body", "geom"} and element.get("name")}
    source_parent = {element.get("name"): (body.get("name") if body is not None else None)
                     for element, body in template_parents.items() if element.tag == "geom" and element.get("name")}
    template_bodies = {element.get("name"): element for element in original_world.iter("body") if element.get("name")}
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
    # (object, element) of each body and joint make_body names after a Blender object.
    generated = []
    # The childclass each body passes on to its geoms, for default lookups.
    childclass = {world: None}
    angle_factor = math.pi / 180 if compiler.get("angle", "degree") == "radian" else 1.0
    def describe(name):
        return "the world body" if name is None else f"body '{name}'"
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
        inherited = source_class.get(("body", attrs.get("name"))) if obj.get("mjworld_mjcf_body") else None
        if "childclass" not in attrs and inherited is not None and inherited != (childclass[parent] or "main"):
            # The body now sits where another class applies; keep passing on
            # the class its joints, sites and geoms used in the source.
            attrs["childclass"] = inherited
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
            generated.append((obj, body))
            if obj.mujoco_world.role == "FREE":
                if owner:
                    raise ExportError(f"{obj.name}: a free body must be top-level.")
                generated.append((obj, ET.SubElement(body, "freejoint", name=obj.name + "_free")))
            elif obj.mujoco_world.role == "HINGE":
                check_door(obj)
                axis = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis])
                low, high, start = door_angles(cfg)
                body.set("quat", numbers(transform.to_quaternion() @ Quaternion(axis, math.radians(property_value(cfg.hinge_start)))))
                generated.append((obj, ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((low, high)), ref=numbers([start]), damping=property_numbers([cfg.damping]))))
        else:
            if cfg.role != baseline.get("role", cfg.role):
                for node in list(body):
                    if node.tag in {"joint", "freejoint"}:
                        body.remove(node)
                if cfg.role == "FREE":
                    if owner:
                        raise ExportError(f"{obj.name}: a free body must be top-level.")
                    generated.append((obj, ET.SubElement(body, "freejoint", name=obj.name + "_free")))
                elif cfg.role == "HINGE":
                    # The body keeps its adopted pose at the starting angle, so
                    # the closed pose need not be at 0.
                    check_door(obj, closed_at_zero=False)
                    axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis]
                    low, high, start = door_angles(cfg)
                    generated.append((obj, ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((low, high)), ref=numbers([start]), damping=property_numbers([cfg.damping]))))
            inertial = body.find("inertial")
            if inertial is not None and abs(cfg.mass - baseline.get("mass", cfg.mass)) > 1e-6:
                ratio = property_value(cfg.mass) / float(inertial.get("mass"))
                inertial.set("mass", property_numbers([cfg.mass]))
                for field in ("diaginertia", "fullinertia"):
                    if inertial.get(field):
                        inertial.set(field, numbers(float(v) * ratio for v in inertial.get(field).split()))
            # A joint is a hinge when it says so, when its class does, or when it
            # names no type at all (MuJoCo's default).
            joint = next((node for node in body.findall("joint")
                          if _joint_type(node, chains, node.get("class") or childclass[body]) == "hinge"), None)
            door_fields = ("hinge_min", "hinge_max", "hinge_start", "hinge_axis", "damping")
            if (joint is None and cfg.role == "HINGE" and baseline.get("role") == "HINGE"
                    and any(getattr(cfg, key) != baseline.get(key, getattr(cfg, key)) for key in door_fields)):
                warnings.append(f"{obj.name}: its door settings were not applied, because the adopted body has no hinge joint; "
                                "its slide or ball joints are exported as in the source MJCF.")
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
                    # MuJoCo accepts one orientation per body; the quaternion replaces the source's.
                    for field in ("euler", "axisangle", "xyaxes", "zaxis"):
                        body.attrib.pop(field, None)
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
    # Every source mesh was removed above, so only the meshes written here count.
    mesh_names = {element.get("name") for element in asset.findall("mesh")}
    new_parts = []
    plane_count = 0
    geom_elements = {}
    for index, obj in enumerate(objects):
        attrs = _json(obj, "mjworld_mjcf_geom")
        owner = body_owner(obj)
        target = make_body(owner) if owner else world
        baked_source = obj.name in covered
        if attrs:
            attrs = dict(attrs)
            if baked_source and obj.mujoco_world.role == "VISUAL":
                continue
            if obj.get("mjworld_bake_source") and not baked_source and obj.mujoco_world.role != "COLLIDER":
                warnings.append(uncovered_bake_source(obj))
            source_name = attrs.get("name")
            if obj in fresh:
                warnings.append(f"{obj.name}: copy of adopted geom '{attrs.get('name')}', exported as geom '{fresh[obj]}'.")
                attrs["name"] = fresh[obj]
            elif source_name in source_parent and source_parent[source_name] != target.get("name"):
                warnings.append(f"{obj.name}: adopted geom '{source_name}' belongs to {describe(source_parent[source_name])} "
                                f"in the source MJCF, but is exported in {describe(target.get('name'))}, following Blender.")
            inherited = source_class.get(("geom", source_name)) if source_name else None
            if "class" not in attrs and inherited is not None and inherited != (childclass[target] or "main"):
                # The geom left the body whose childclass it used; name that class itself.
                attrs["class"] = inherited
            geom_class = attrs.get("class") or childclass[target]
            resolved = _resolved_geom(attrs, chains, geom_class)
            kind = resolved.get("type", "sphere")
            if baked_source or obj.mujoco_world.role == "COLLIDER":
                attrs.update(group="3", rgba="0 0 0 0")
            transform = _rigid(obj.matrix_world, unit)
            # The geom centre in double precision, for objects whose origin is
            # elsewhere (see _origin_offset); None otherwise.
            position = None
            if kind not in {"mesh", "plane"}:
                reference_bounds = _json(obj, "mjworld_mjcf_original_bounds")
                offset = obj.get("mjworld_mjcf_geom_offset")
                if offset is not None:
                    local = np.array(list(offset), dtype=float)
                    if reference_bounds:
                        local += (np.mean(np.array([list(v) for v in obj.bound_box], dtype=float), axis=0)
                                  - np.mean(np.array(reference_bounds, dtype=float), axis=0))
                    position = (np.array(obj.matrix_world, dtype=float) @ np.append(local, 1.0))[:3] * unit
                    transform.translation = Vector(position)
                elif reference_bounds:
                    reference_center = sum((Vector(v) for v in reference_bounds), Vector()) / 8
                    current_center = sum((Vector(v) for v in obj.bound_box), Vector()) / 8
                    transform.translation += obj.matrix_world.to_3x3() @ (current_center - reference_center) * unit
            if owner:
                frame = _rigid(owner.matrix_world, unit)
                transform = frame.inverted() @ transform
                if position is not None:
                    position = np.linalg.solve(np.array(frame, dtype=float), np.append(position, 1.0))[:3]
            size = _primitive_size(obj, attrs, resolved)
            segment = bool(attrs.get("fromto")) and kind in _ELONGATED
            if segment:
                # A fromto shape stays verbatim until it moves or is resized; then
                # pos/quat and an explicit full size replace the segment.
                resized = size != attrs.get("size")
                moved = resized or not _same_segment(transform, resolved)
                if moved and not resized:
                    size = numbers(_full_size(resolved))
                if moved:
                    _check_segment_axis(obj, kind, size, unit)
            else:
                old = _matrix(attrs, compiler)
                if position is None:
                    moved = max(abs(transform[i][j] - old[i][j]) for i in range(4) for j in range(4)) > 2e-6
                else:
                    source_position = _floats(attrs, "pos", 3) if attrs.get("pos") else [0.0, 0.0, 0.0]
                    moved = (max(abs(transform[i][j] - old[i][j]) for i in range(3) for j in range(3)) > 2e-6
                             or max(abs(position[i] - source_position[i]) for i in range(3)) > 2e-6)
            if moved:
                for field in ("fromto", "euler", "axisangle", "xyaxes", "zaxis"):
                    attrs.pop(field, None)
                attrs.update(_pose(transform))
                if position is not None:
                    attrs["pos"] = numbers(position)
            if size:
                attrs["size"] = size
            if kind == "plane":
                plane_count += 1
            elif kind == "mesh":
                geometry = _extract_geometry(obj, depsgraph, unit)
                # Geometry is expressed in its object's rigid rotation frame;
                # meshes retain arbitrary evaluated scale, modifiers and shear.
                triangles = [tri for _material, tri in geometry["triangles"]]
                uv_name = geometry["render_uv"]
                # Share identical evaluated shapes across repeated task props.
                # Rounding only the cache key absorbs Blender float32 rotation
                # noise; the written mesh retains full evaluated precision.
                # The regenerated mesh keeps the source mesh's inertia mode,
                # convex hull limit and normals. Without a source mode, a mesh
                # that adds no mass uses shell, which also accepts open or flat
                # visual meshes; other meshes keep MuJoCo's default.
                physics = _mesh_physics(source_meshes.get(resolved.get("mesh")), chains)
                if "inertia" not in physics and not _contributes_inertia(attrs, chains, geom_class, compiler):
                    physics["inertia"] = "shell"
                physics = {field: physics[field] for field in _MESH_PHYSICS if field in physics}
                key =hashlib.sha256(json.dumps({"vertices": [[round(v, 5) for v in p] for p in geometry["vertices"]], "triangles": triangles, "uv": geometry["uv_layers"].get(uv_name), "physics": physics}, sort_keys=True).encode()).hexdigest()
                if key not in mesh_cache:
                    # The atlas may be matched to an unnamed source geom.
                    mesh_name = _mesh_asset_name(attrs.get("name") or obj.name, mesh_names)
                    mesh_names.add(mesh_name)
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
            # New bake sources that also collide keep their colliders through
            # the ordinary exporter, which leaves out a covered visual.
            if obj.mujoco_world.role != "VISUAL":
                new_parts.append(obj)
                continue
            if baked_source:
                continue
            if obj.get("mjworld_bake_source"):
                warnings.append(uncovered_bake_source(obj))
            geometry = _extract_geometry(obj, depsgraph, unit)
            mat = obj.material_slots[0].material if obj.material_slots else None
            info = inspect_material(mat)
            warnings.extend(info["warnings"])
            label = slug(obj.name) + "_" + str(index)
            material_args = {"name": label + "_material", "rgba": numbers(info["color"]), "specular": "0", "emission": property_numbers([settings.baked_emission])}
            image = info["image"]
            problem = image_problem(image) if image else None
            if problem:
                warnings.append(f"{obj.name}: image '{image.name}' {problem}; export used the flat material colour.")
                image = None
            if image:
                written = {}
                image_file = assets.write(label + ".png", lambda target_path: written.update(
                    _write_image(image, target_path, baked=True, scene=scene, where=obj.name)))
                ET.SubElement(asset, "texture", name=label + "_texture", type="2d", file=asset_prefix + "/" + image_file,
                              colorspace=texture_colorspace(settings, written["linear"]))
                if written["hdr"]:
                    warnings.append(f"{obj.name}: HDR bake '{image.name}' converted to display PNG using {profile_text(written['profile'])}.")
                warnings.extend(f"{obj.name}: {note}" for note in written["notes"])
                material_args.update(texture=label + "_texture", rgba="1 1 1 1", texuniform="false")
            ET.SubElement(asset, "material", **material_args)
            mesh_file = assets.write(label + ".obj", lambda target_path: _write_obj(target_path, geometry, [tri for _m, tri in geometry["triangles"]], info.get("uv_map") or geometry["render_uv"]))
            ET.SubElement(asset, "mesh", name=label + "_mesh", file=asset_prefix + "/" + mesh_file, inertia="shell")
            ET.SubElement(target, "geom", name=label, type="mesh", mesh=label + "_mesh", material=material_args["name"], contype="0", conaffinity="0", group="1", mass="0", **_pose(_rigid(obj.matrix_world, unit) if owner is None else _rigid(owner.matrix_world, unit).inverted() @ _rigid(obj.matrix_world, unit)))
            details.append({"source": obj.name, "body": owner.name if owner else "world", "role": "VISUAL", "geometry": "mesh"})
    # Copies of adopted bodies are exported even when they hold no geoms.
    for obj in sorted((o for o in fresh if is_body(o)), key=lambda o: o.name):
        make_body(obj)
    # Every body exists now: the object loop made each object's owner, new parts' owners
    # included, and the loop above made the copies.
    _check_generated_names(world, generated)
    geom_sources = {element: obj for obj, element in geom_elements.items()}
    details.extend(_append_new_parts(context, new_parts, root, world, asset, assets, make_body, warnings, asset_prefix,
                                     geom_sources))
    if plane_count != 1:
        raise ExportError(f"This adopted world needs exactly one analytic floor plane; found {plane_count}.")
    for obj, body in bodies.items():
        # A moving body whose colliders were all parented elsewhere would fall
        # through the floor; the source gave it some, so say so.
        template = template_bodies.get(_json(obj, "mjworld_mjcf_body", {}).get("name")) if obj not in fresh else None
        if template is None or obj.mujoco_world.role not in {"FREE", "HINGE"}:
            continue
        if (any(_collides(geom.attrib, chains, template_classes[geom]) for geom in template.findall("geom"))
                and not any(_collides(geom.attrib, chains, geom.get("class") or childclass[body]) for geom in body.findall("geom"))):
            warnings.append(f"{obj.name}: this moving body has no colliding geoms left, though the source MJCF gave it some; "
                            "parent its collision objects under it again, or it passes through everything.")
    _apply_weights(bodies, childclass, chains, warnings, geom_sources, depsgraph, unit)
    from .authored_materials import update_authored_materials
    warnings.extend(update_authored_materials(scene, root, assets, asset_prefix, geom_elements))
    # No duplicate legacy meshes/textures in runtime bundles. Defaults can carry
    # material references, so gather references across the entire XML tree.
    mesh_refs = {element.get("mesh") for element in root.iter() if element.get("mesh")}
    material_refs = {element.get("material") for element in root.iter() if element.get("material")}
    removed = []
    for element in list(asset):
        if (element.tag == "mesh" and element.get("name") not in mesh_refs) or (element.tag == "material" and element.get("name") not in material_refs):
            asset.remove(element)
            removed.append(element)
    # Gathered once the unused materials are gone, so their textures go too;
    # material layers and defaults name textures as well.
    texture_refs = {element.get("texture") for element in root.iter() if element.get("texture")}
    for element in list(asset.findall("texture")):
        if element.get("name") not in texture_refs and element.get("type") != "skybox":
            asset.remove(element)
            removed.append(element)
    # Assets can share one file, so a file is dropped only when nothing left names it.
    kept_files = {element.get(field) for element in root.iter() for field in _FILE_FIELDS if element.get(field)}
    for element in removed:
        for field in _FILE_FIELDS:
            if element.get(field) and element.get(field) not in kept_files:
                assets.discard(path.parent / element.get(field))
    return {"addon_version": addon_version(), "blender_version": bpy.app.version_string, "file": path.name, "objects": len(details), "bodies": len(bodies), "meshes": len(asset.findall("mesh")), "textures": len(asset.findall("texture")), "triangles": sum(len(o.data.polygons) for o in objects), "adopted_physics": True, "metres_per_blender_unit": unit, "object_details": details, "warnings": list(dict.fromkeys(warnings)), "plane": {"enabled": True, "name": "floor", "count": plane_count}}


def _apply_weights(bodies, childclass, chains, warnings, geom_sources, depsgraph, unit):
    """Give movable bodies their Weight where the source inertia no longer applies.

    New Movable/Door bodies, and adopted bodies without <inertial> whose role
    or Weight changed, get the Weight split over their colliding geoms in
    proportion to collider volume (one uniform density); their other geoms get
    mass 0. ``geom_sources`` maps geom elements to their Blender objects, for
    mesh hull volumes. Untouched adopted bodies keep their source physics, and
    bodies with <inertial> are rescaled in make_body.
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
        volumes = [_geom_volume(geom.attrib, chains, geom.get("class") or childclass[body], geom_sources.get(geom), depsgraph, unit)
                   for geom in colliders]
        shares = dict(zip(colliders, _split_weight(property_value(cfg.mass), volumes)))
        for geom in geoms:
            geom.set("mass", numbers([shares[geom]]) if geom in shares else "0")
