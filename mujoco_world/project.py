"""Body hierarchies and lossless physics adoption for existing MJCF worlds.

No room layouts live here. Adoption attaches physics to same-named Blender
objects; Blender remains the editable geometry and placement source.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
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


def _matrix(attrs):
    pos = Vector(tuple(float(v) for v in attrs.get("pos", "0 0 0").split()))
    quat = Quaternion(tuple(float(v) for v in attrs.get("quat", "1 0 0 0").split()))
    return Matrix.LocRotScale(pos, quat, Vector((1, 1, 1)))


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
    missing_bodies, missing_geoms = [], []
    body_count = geom_count = 0
    for obj in scene.objects:
        for key in ("mjworld_mjcf_body", "mjworld_mjcf_geom", "mjworld_mjcf_children", "mjworld_mjcf_original_scale"):
            if key in obj:
                del obj[key]
        if obj.type == "MESH" and obj.get("simulation_geometry"):
            obj["mjworld_adopted_exclude"] = True
    for source in root.findall("./worldbody//body"):
        obj = find(source.get("name", ""))
        if obj is None or obj.type != "EMPTY":
            missing_bodies.append(source.get("name", "<unnamed>"))
            continue
        obj.mujoco_world.body = True
        obj["mjworld_mjcf_body"] = json.dumps(dict(source.attrib))
        children = [ET.tostring(node, encoding="unicode") for node in source if node.tag not in {"body", "geom"}]
        obj["mjworld_mjcf_children"] = json.dumps(children)
        role = "FREE" if source.find("freejoint") is not None or source.find("joint[@type='free']") is not None else "HINGE" if source.find("joint") is not None else "STATIC"
        obj.mujoco_world.role = role
        inertial = source.find("inertial")
        if inertial is not None and inertial.get("mass"):
            obj.mujoco_world.mass = float(inertial.get("mass"))
        joint = source.find("joint[@type='hinge']")
        if joint is not None:
            degree = compiler is None or compiler.get("angle", "degree") == "degree"
            conversion = 1.0 if degree else 180.0 / math.pi
            limits = [float(v) * conversion for v in joint.get("range", "-110 0").split()]
            obj.mujoco_world.hinge_min, obj.mujoco_world.hinge_max = limits
            obj.mujoco_world.hinge_start = float(joint.get("ref", "0")) * conversion
            obj.mujoco_world.damping = float(joint.get("damping", "0"))
            axis = [float(v) for v in joint.get("axis", "0 0 1").split()]
            obj.mujoco_world.hinge_axis = "XYZ"[max(range(3), key=lambda i: abs(axis[i]))]
        obj["mjworld_settings_baseline"] = json.dumps({key: getattr(obj.mujoco_world, key) for key in ("role", "mass", "hinge_min", "hinge_max", "hinge_start", "hinge_axis", "damping")})
        body_count += 1
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
        obj["mjworld_mjcf_geom"] = json.dumps(attrs)
        obj["mjworld_adopted_exclude"] = False
        scale = list(_scale(obj.matrix_world))
        bounds = [list(v) for v in obj.bound_box]
        # Legacy primitive authoring metadata identifies unit box/cylinder/
        # sphere meshes. Use their source dimensions as the baseline so manual
        # geometry/scale edits made before the first adoption are preserved too.
        kind = attrs.get("type", "sphere")
        if obj.get("mujoco_type") == kind and kind not in {"mesh", "plane"}:
            size = [float(v) for v in attrs.get("size", "").split()]
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
        obj["mjworld_mjcf_original_scale"] = scale
        obj["mjworld_mjcf_original_bounds"] = json.dumps(bounds)
        # Explicit helpers remain exportable even inside hidden collections.
        collision_only = obj.get("collision_only") or attrs.get("rgba", "").split()[-1:] == ["0"]
        visual_only = obj.get("visual_only") or (attrs.get("contype") == attrs.get("conaffinity") == "0") or "visual" in attrs.get("class", "").lower()
        if obj.get("preserved_original_static_visual"):
            obj["mjworld_bake_source"] = True
            obj.mujoco_world.role = "VISUAL" if visual_only else "STATIC"
        else:
            obj.mujoco_world.role = "COLLIDER" if collision_only else "VISUAL" if visual_only else "STATIC"
        if obj.get("baked_static_visual") or obj.get("mjworld_baked_atlas"):
            material = root.find(f"./asset/material[@name='{attrs.get('material')}']")
            if material is not None:
                scene.mujoco_world.baked_emission = float(material.get("emission", ".2"))
        geom_count += 1
    if missing_bodies or missing_geoms:
        # A partial adoption can silently erase collision geometry. Do not allow
        # it to become the active project; report every missing physics source.
        raise ExportError("MJCF names were not found in this Blender scene. Bodies: " + ", ".join(missing_bodies[:10]) + "; geoms: " + ", ".join(missing_geoms[:10]) + ". Use object_map for renamed objects.")
    scene["mjworld_mjcf_template"] = ET.tostring(root, encoding="unicode")
    scene["mjworld_mjcf_source"] = bpy.path.relpath(str(filepath)) if bpy.data.filepath else str(filepath)
    scene["mjworld_mjcf_adoption_version"] = 1
    from .authored_materials import capture_material_baselines
    capture_material_baselines(scene, root)
    return {"bodies": body_count, "geoms": geom_count, "source": str(filepath)}


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


def regroup_export(context, filepath, report):
    """Lift exported parts into compound bodies, retaining relative transforms."""
    from .exporter import ExportError, numbers
    path = _resolved_path(filepath).with_suffix(".xml")
    root = ET.parse(path).getroot()
    world = root.find("worldbody")
    unit = context.scene.unit_settings.scale_length
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
            if not cfg.hinge_min <= cfg.hinge_start <= cfg.hinge_max:
                raise ExportError(f"{obj.name}: door starting angle must be inside its limits.")
            # The authored pose is the closed pose. Rotate the body at the
            # start, and set ref to that angle so limits stay in closed-space.
            axis = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis])
            angle = math.radians(cfg.hinge_start)
            body.set("quat", numbers(transform.to_quaternion() @ Quaternion(axis, angle)))
            ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((cfg.hinge_min, cfg.hinge_max)), ref=str(cfg.hinge_start), damping=str(cfg.damping))
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
            for geom in list(original.findall("geom")):
                pose = relative @ _matrix(geom.attrib)
                geom.attrib.update(_pose(pose))
                if geom.get("contype") != "0":
                    movable = owner.mujoco_world.role == "FREE"
                    geom.set("contype", getattr(context.scene.mujoco_world, "movable_contype" if movable else "object_contype"))
                    geom.set("conaffinity", getattr(context.scene.mujoco_world, "movable_conaffinity" if movable else "object_conaffinity"))
                    geom.set("mass", str(owner.mujoco_world.mass / len(colliders)) if owner.mujoco_world.role in {"FREE", "HINGE"} else "0")
                target.append(geom)
            world.remove(original)
            detail["body"] = owner.name
            detail["body_role"] = owner.mujoco_world.role
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    report["compound_bodies"] = len(groups)
    grouped_names = {obj.name for parts in groups.values() for obj, _d in parts}
    report["warnings"] = [w for w in report["warnings"] if not any(w.startswith(name + ": parent transforms") for name in grouped_names)]
    (path.parent / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _append_new_parts(context, objects, root, world, asset, assets_path, make_body, warnings, asset_prefix=None):
    """Use the ordinary exporter for newly added furniture and compound props."""
    from .exporter import _export_simple, ExportError, slug
    if not objects:
        return []
    scene = context.scene
    asset_prefix = assets_path.name if asset_prefix is None else asset_prefix
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
                    target = assets_path / (slug(rename[old_name]) + original.suffix)
                    if original.resolve() != target.resolve():
                        shutil.copyfile(original, target)
                    element.set("file", asset_prefix + "/" + target.name)
                asset.append(element)
            details = report["object_details"]
            groups = {}
            for detail in details:
                obj = scene.objects.get(detail["source"])
                owner = body_owner(obj)
                groups.setdefault(owner, []).append((obj, detail))
            for owner, members in groups.items():
                target = make_body(owner) if owner else world
                collision_count = sum(bool(d["collision"]) for _o, d in members)
                if owner and owner.mujoco_world.role in {"FREE", "HINGE"} and not owner.get("mjworld_mjcf_body") and not collision_count:
                    raise ExportError(f"{owner.name}: a movable body needs at least one collision helper.")
                for obj, detail in members:
                    body = parts_root.find(f"./worldbody/body[@name='{detail['body']}']")
                    if root.find("compiler").get("angle", "degree") == "radian":
                        for joint in body.findall("joint"):
                            if joint.get("type", "hinge") in {"hinge", "ball"}:
                                for field in ("range", "ref", "springref"):
                                    if joint.get(field):
                                        joint.set(field, " ".join(str(math.radians(float(v))) for v in joint.get(field).split()))
                    for geom in body.findall("geom"):
                        for field in ("mesh", "material"):
                            if geom.get(field) in rename:
                                geom.set(field, rename[geom.get(field)])
                    if owner:
                        relative = _rigid(owner.matrix_world, scene.unit_settings.scale_length).inverted() @ _rigid(obj.matrix_world, scene.unit_settings.scale_length)
                        for geom in list(body.findall("geom")):
                            geom.attrib.update(_pose(relative @ _matrix(geom.attrib)))
                            if geom.get("contype") != "0":
                                movable = owner.mujoco_world.role == "FREE"
                                geom.set("contype", getattr(scene.mujoco_world, "movable_contype" if movable else "object_contype"))
                                geom.set("conaffinity", getattr(scene.mujoco_world, "movable_conaffinity" if movable else "object_conaffinity"))
                                mass = owner.mujoco_world.mass / collision_count if owner.mujoco_world.role in {"FREE", "HINGE"} and not owner.get("mjworld_mjcf_body") else 0
                                geom.set("mass", str(mass))
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
    size = [float(x) for x in attrs.get("size", "").split()]
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


def export_adopted(context, filepath):
    """Export retained physics and live Blender geometry as a portable world."""
    from .exporter import ExportError, _extract_geometry, _write_obj, _write_image, _asset_paths, numbers, inspect_material, slug
    if context.mode != "OBJECT":
        raise ExportError("Switch to Object Mode before exporting.")
    scene = context.scene
    settings = scene.mujoco_world
    if settings.selected_only:
        raise ExportError("An adopted physics project exports the complete world. Turn off 'Export selected objects only' so joint, contact and keyframe references remain valid.")
    path = _resolved_path(filepath).with_suffix(".xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    assets_path, asset_prefix, shared_assets = _asset_paths(scene, path)
    root = ET.fromstring(scene["mjworld_mjcf_template"])
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
    # Preserve original material/solver/default declarations. Asset files are
    # copied into one sibling directory so exports never depend on author paths.
    # Every referenced mesh is exported from current Blender geometry below;
    # carrying the old files too would duplicate task meshes on every export.
    for element in list(asset.findall("mesh")):
        asset.remove(element)
    for index, element in enumerate(asset):
        filename = element.get("file")
        if filename:
            directory = source_meshdir if element.tag == "mesh" else source_texturedir
            original = Path(filename) if Path(filename).is_absolute() else source.parent / directory / filename
            target = assets_path / (slug(element.get("name", f"asset_{index}")) + Path(filename).suffix.lower())
            if not original.exists():
                raise ExportError(f"Missing adopted asset: {original}. Keep the source world assets beside the saved project.")
            if original.resolve() != target.resolve():
                shutil.copyfile(original, target)
            element.set("file", asset_prefix + "/" + target.name)
    original_world = root.find("worldbody")
    world = ET.Element("worldbody", original_world.attrib)
    # Retain lights/cameras/sites and world-level physics declarations. Bodies
    # and geometry are reconstructed from the current Blender object hierarchy.
    for element in original_world:
        if element.tag not in {"body", "geom"}:
            world.append(copy.deepcopy(element))
    root.remove(original_world)
    root.append(world)
    unit = float(scene.unit_settings.scale_length)
    if unit <= 0:
        raise ExportError("Scene unit scale must be positive.")
    depsgraph = context.evaluated_depsgraph_get()
    candidates = list(context.selected_objects) if settings.selected_only else list(scene.objects)
    if settings.selected_only:
        for obj in list(candidates):
            if is_body(obj):
                candidates.extend(obj.children_recursive)
    objects = sorted({o for o in candidates if o.type == "MESH" and not o.get("mjworld_adopted_exclude") and _included(o) and (o.get("mjworld_mjcf_geom") or o.get("mjworld_baked_atlas") or (o.visible_get() and not o.hide_render))}, key=lambda o: o.name)
    if not objects:
        raise ExportError("No mesh objects selected for export.")
    bodies = {}
    angle_factor = math.pi / 180 if compiler.get("angle", "degree") == "radian" else 1.0
    def make_body(obj):
        if obj in bodies:
            return bodies[obj]
        owner = body_owner(obj)
        parent = make_body(owner) if owner else world
        attrs = _json(obj, "mjworld_mjcf_body", {"name": obj.name})
        transform = _rigid(obj.matrix_world, unit)
        if owner:
            transform = _rigid(owner.matrix_world, unit).inverted() @ transform
        old = _matrix(attrs)
        # Avoid float32 Blender rounding changing untouched source poses.
        if max(abs(transform[i][j] - old[i][j]) for i in range(4) for j in range(4)) > 2e-6:
            for field in ("euler", "axisangle", "xyaxes", "zaxis"):
                attrs.pop(field, None)
            attrs.update(_pose(transform))
        body = ET.SubElement(parent, "body", **attrs)
        for xml in _json(obj, "mjworld_mjcf_children", []):
            body.append(ET.fromstring(xml))
        cfg = obj.mujoco_world
        baseline = _json(obj, "mjworld_settings_baseline", {})
        if not obj.get("mjworld_mjcf_body"):
            if obj.mujoco_world.role == "FREE":
                if owner:
                    raise ExportError(f"{obj.name}: a free body must be top-level.")
                ET.SubElement(body, "freejoint", name=obj.name + "_free")
            elif obj.mujoco_world.role == "HINGE":
                if not cfg.hinge_min <= cfg.hinge_start <= cfg.hinge_max:
                    raise ExportError(f"{obj.name}: starting angle must be within the door limits.")
                axis = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis])
                body.set("quat", numbers(transform.to_quaternion() @ Quaternion(axis, math.radians(cfg.hinge_start))))
                ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((cfg.hinge_min * angle_factor, cfg.hinge_max * angle_factor)), ref=str(cfg.hinge_start * angle_factor), damping=str(cfg.damping))
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
                    axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis]
                    ET.SubElement(body, "joint", name=obj.name + "_hinge", type="hinge", axis=numbers(axis), range=numbers((cfg.hinge_min * angle_factor, cfg.hinge_max * angle_factor)), ref=str(cfg.hinge_start * angle_factor), damping=str(cfg.damping))
            inertial = body.find("inertial")
            if inertial is not None and abs(cfg.mass - baseline.get("mass", cfg.mass)) > 1e-6:
                ratio = cfg.mass / float(inertial.get("mass"))
                inertial.set("mass", str(cfg.mass))
                for field in ("diaginertia", "fullinertia"):
                    if inertial.get(field):
                        inertial.set(field, numbers(float(v) * ratio for v in inertial.get(field).split()))
            joint = body.find("joint[@type='hinge']")
            if joint is not None and cfg.role == "HINGE":
                if not cfg.hinge_min <= cfg.hinge_start <= cfg.hinge_max:
                    raise ExportError(f"{obj.name}: starting angle must be within the door limits.")
                if cfg.hinge_min != baseline.get("hinge_min") or cfg.hinge_max != baseline.get("hinge_max"):
                    joint.set("range", numbers((cfg.hinge_min * angle_factor, cfg.hinge_max * angle_factor)))
                if cfg.damping != baseline.get("damping"):
                    joint.set("damping", str(cfg.damping))
                if cfg.hinge_axis != baseline.get("hinge_axis"):
                    joint.set("axis", numbers({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[cfg.hinge_axis]))
                if abs(cfg.hinge_start - baseline.get("hinge_start", cfg.hinge_start)) > 1e-5:
                    joint.set("ref", str(cfg.hinge_start * angle_factor))
                    axis = Vector(tuple(float(v) for v in joint.get("axis", "0 0 1").split()))
                    delta = math.radians(cfg.hinge_start - baseline["hinge_start"])
                    body.set("quat", numbers(transform.to_quaternion() @ Quaternion(axis, delta)))
        bodies[obj] = body
        return body
    # Retain source body/joint indexing, including bodies that only contain
    # children or sensors, so unchanged keyframes keep their original meaning.
    imported_bodies = {_json(o, "mjworld_mjcf_body", {}).get("name"): o for o in scene.objects if is_body(o) and o.get("mjworld_mjcf_body") and _included(o)}
    for definition in original_world.iter("body"):
        obj = imported_bodies.get(definition.get("name"))
        if obj is not None:
            make_body(obj)
    details = []
    mesh_cache = {}
    new_parts = []
    plane_count = 0
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
            old = _matrix(attrs)
            if max(abs(transform[i][j] - old[i][j]) for i in range(4) for j in range(4)) > 2e-6:
                for field in ("fromto", "euler", "axisangle", "xyaxes", "zaxis"):
                    attrs.pop(field, None)
                attrs.update(_pose(transform))
            size = _primitive_size(obj, attrs)
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
                key = hashlib.sha256(json.dumps({"vertices": [[round(v, 5) for v in p] for p in geometry["vertices"]], "triangles": triangles, "uv": geometry["uv_layers"].get(uv_name)}, sort_keys=True).encode()).hexdigest()
                if key not in mesh_cache:
                    mesh_name = slug(attrs["name"]) + "_authored_mesh"
                    _write_obj(assets_path / (mesh_name + ".obj"), geometry, triangles, uv_name)
                    ET.SubElement(asset, "mesh", name=mesh_name, file=asset_prefix + "/" + mesh_name + ".obj", inertia="shell")
                    mesh_cache[key] = mesh_name
                mesh_name = mesh_cache[key]
                attrs["mesh"] = mesh_name
            ET.SubElement(target, "geom", **attrs)
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
            material_args = {"name": label + "_material", "rgba": numbers(info["color"]), "specular": "0", "emission": str(settings.baked_emission)}
            image = info["image"]
            if image:
                image_path = assets_path / (label + ".png")
                _write_image(image, image_path, baked=True, scene=scene)
                ET.SubElement(asset, "texture", name=label + "_texture", type="2d", file=asset_prefix + "/" + image_path.name, colorspace="linear")
                material_args.update(texture=label + "_texture", rgba="1 1 1 1", texuniform="false")
            ET.SubElement(asset, "material", **material_args)
            _write_obj(assets_path / (label + ".obj"), geometry, [tri for _m, tri in geometry["triangles"]], info.get("uv_map") or geometry["active_uv"])
            ET.SubElement(asset, "mesh", name=label + "_mesh", file=asset_prefix + "/" + label + ".obj", inertia="shell")
            ET.SubElement(target, "geom", name=label, type="mesh", mesh=label + "_mesh", material=material_args["name"], contype="0", conaffinity="0", group="1", mass="0", **_pose(_rigid(obj.matrix_world, unit) if owner is None else _rigid(owner.matrix_world, unit).inverted() @ _rigid(obj.matrix_world, unit)))
            details.append({"source": obj.name, "body": owner.name if owner else "world", "role": "VISUAL", "geometry": "mesh"})
    details.extend(_append_new_parts(context, new_parts, root, world, asset, assets_path, make_body, warnings, asset_prefix))
    if plane_count != 1:
        raise ExportError(f"This adopted world needs exactly one analytic floor plane; found {plane_count}.")
    from .authored_materials import update_authored_materials
    warnings.extend(update_authored_materials(scene, root, assets_path, asset_prefix))
    # No duplicate legacy meshes/textures in runtime bundles. Defaults can carry
    # material references, so gather references across the entire XML tree.
    mesh_refs = {element.get("mesh") for element in root.iter() if element.get("mesh")}
    material_refs = {element.get("material") for element in root.iter() if element.get("material")}
    for element in list(asset):
        if (element.tag == "mesh" and element.get("name") not in mesh_refs) or (element.tag == "material" and element.get("name") not in material_refs):
            file = element.get("file")
            asset.remove(element)
            if file and not shared_assets:
                (path.parent / file).unlink(missing_ok=True)
    texture_refs = {element.get("texture") for element in asset if element.get("texture")}
    for element in list(asset.findall("texture")):
        if element.get("name") not in texture_refs and element.get("type") != "skybox":
            file = element.get("file")
            asset.remove(element)
            if file and not shared_assets:
                (path.parent / file).unlink(missing_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    report = {"addon_version": "0.2.0", "blender_version": bpy.app.version_string, "file": path.name, "objects": len(details), "bodies": len(bodies), "meshes": len(asset.findall("mesh")), "textures": len(asset.findall("texture")), "triangles": sum(len(o.data.polygons) for o in objects), "adopted_physics": True, "metres_per_blender_unit": unit, "object_details": details, "warnings": list(dict.fromkeys(warnings)), "plane": {"enabled": True, "name": "floor", "count": plane_count}}
    (path.parent / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
