"""Non-destructive diffuse-light atlases for an authored Blender world.

The user's materials, lights and world supply the appearance. Only fixed visible
surfaces participate: movable bodies, doors and collision helpers neither enter
the atlas nor leave permanent shadows in it. Original objects and their physics
metadata remain in the file for editing and collision export.
"""

import math
from pathlib import Path
import tempfile

import bpy


class BakeError(ValueError):
    pass


# The lightmap PNG is pinned to AgX on the sRGB display, whatever the user's
# view settings are; the summary and the master image record exactly this.
LIGHTMAP_PROFILE = {"view_transform": "AgX", "look": "None", "exposure": 0.0, "gamma": 1.0}
BAKE_PASSES = ("Combined diffuse and emission, direct and indirect light; metallic surfaces baked as diffuse, "
               "glossy and transmission left out; fixed geometry only")
# Object-level inputs that change how a material shades; bake clones copy them.
_SHADING_FIELDS = ("color", "pass_index", "visible_camera", "visible_diffuse", "visible_glossy",
                   "visible_transmission", "visible_volume_scatter", "visible_shadow",
                   "is_holdout", "is_shadow_catcher")
_SOURCE_KEYS = ("mjworld_bake_source", "mjworld_bake_original_hide_render", "mjworld_bake_original_hide_viewport",
                "preserved_original_static_visual")


def _role(obj):
    settings = getattr(obj, "mujoco_world", None)
    return getattr(settings, "role", "STATIC")


def _dynamic(obj):
    """A visual inherits motion from any ancestor, including an Empty body."""
    current = obj
    while current is not None:
        if _role(current) in {"FREE", "HINGE"} or bool(current.get("free_body", False)):
            return True
        # Compatibility with the room projects made before the add-on importer.
        if current.get("mujoco_joint") or current.get("joint_type"):
            return True
        current = current.parent
    return False


def _excluded(obj):
    current = obj
    while current is not None:
        if _role(current) == "EXCLUDE":
            return True
        current = current.parent
    return False


def _source_visible(obj, view_layer):
    if obj.get("mjworld_bake_source", False) or obj.get("preserved_original_static_visual", False):
        return True
    return obj.visible_get(view_layer=view_layer) and not obj.hide_render


def is_atlas(obj):
    return bool(obj.get("mjworld_baked_atlas", False) or obj.get("baked_static_visual", False))


def is_bake_source(obj):
    return bool(obj.get("mjworld_bake_source", False) or obj.get("preserved_original_static_visual", False))


def bake_covered(scene):
    """Names of the marked bake sources whose appearance an atlas provides.

    An atlas lists its sources in ``mjworld_bake_sources``. A legacy atlas
    without that list (the Chula realistic room) covers every marked source.
    Without an atlas that is exported, nothing is covered, and the exporters
    write the sources' own visuals again.
    """
    marked = {obj.name for obj in scene.objects if is_bake_source(obj)}
    covered = set()
    for atlas in scene.objects:
        if atlas.type != "MESH" or not is_atlas(atlas) or _excluded(atlas) or atlas.get("mjworld_adopted_exclude"):
            continue
        names = atlas.get("mjworld_bake_sources")
        if names is None:
            return marked
        covered.update(str(name) for name in names)
    return covered & marked


def bake_leftovers(scene):
    """Objects that 'Edit original surfaces' can restore or remove."""
    return [obj for obj in scene.objects if is_atlas(obj) or is_bake_source(obj) or obj.get("mjworld_bake_clone", False)]


def _select(context, objects):
    view_layer = context.view_layer
    # Removing objects leaves stale entries until the view layer updates.
    view_layer.update()
    for obj in view_layer.objects:
        if obj is not None:
            obj.select_set(False)
    live = [obj for obj in objects if _exists(obj) and obj.name in view_layer.objects]
    for obj in live:
        obj.select_set(True)
    view_layer.objects.active = live[0] if live else None


def _remove_objects(objects):
    meshes, materials, collections = [], set(), set()
    for obj in objects:
        if not _exists(obj):
            continue
        if obj.type == "MESH":
            meshes.append(obj.data)
            materials.update(material for material in obj.data.materials
                             if material and (material.get("mjworld_generated_bake", False)
                                              or obj.get("baked_static_visual", False)))
        collections.update(collection for collection in obj.users_collection
                           if collection.get("mjworld_generated_bake", False))
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in materials:
        if material.users == 0:
            images = {node.image for node in material.node_tree.nodes
                      if node.type == "TEX_IMAGE" and node.image
                      and (node.image.get("mjworld_generated_bake", False)
                           or Path(node.image.filepath).name == "static_lighting_linear.exr")}
            bpy.data.materials.remove(material)
            for image in images:
                if image.users == 0:
                    bpy.data.images.remove(image)
    for collection in collections:
        if not collection.objects and not collection.children:
            bpy.data.collections.remove(collection)


def _exists(obj, collection=None):
    try:
        return obj is not None and obj.name in (collection if collection is not None else bpy.data.objects)
    except ReferenceError:
        return False


def _material_copy(source, image, copies):
    key = source.as_pointer() if source else 0
    if key in copies:
        return copies[key]
    material = source.copy() if source else bpy.data.materials.new("MuJoCo bake default")
    material.name = "MuJoCo bake working material"
    copies[key] = material
    if bpy.app.version < (5, 0, 0):
        # Blender 5 always renders through the node tree; 4.x needs it enabled.
        material.use_nodes = True
    tree = material.node_tree
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED" and node.inputs.get("Metallic") is not None:
            # A diffuse bake sees (1 - metallic) x base colour, so a metal
            # bakes black. Bake its base colour as diffuse albedo instead.
            # Transmission stays: glass must still let light into the room.
            metallic = node.inputs["Metallic"]
            for link in list(metallic.links):
                tree.links.remove(link)
            metallic.default_value = 0.0
    target = tree.nodes.new("ShaderNodeTexImage")
    target.name = "MuJoCo atlas bake target"
    target.image = image
    for node in tree.nodes:
        node.select = node == target
    tree.nodes.active = target
    return material


def _shading_inputs(materials):
    """Custom properties read by Attribute (Object) nodes, and whether Object Info Random is used."""
    names, random = set(), False
    def walk(tree, seen):
        nonlocal random
        for node in tree.nodes:
            if node.type == "ATTRIBUTE" and node.attribute_type in {"OBJECT", "INSTANCER"}:
                name = node.attribute_name
                names.add(name[2:-2] if name.startswith('["') and name.endswith('"]') else name)
            elif node.type == "OBJECT_INFO" and node.outputs["Random"].is_linked:
                random = True
            elif node.type == "GROUP" and node.node_tree and node.node_tree.as_pointer() not in seen:
                seen.add(node.node_tree.as_pointer())
                walk(node.node_tree, seen)
    for material in materials:
        if material is not None and material.node_tree is not None:
            walk(material.node_tree, set())
    return names, random


def _copy_shading(source, clone, warnings):
    """Give a bake clone the object-level shading inputs of its source."""
    for name in _SHADING_FIELDS:
        if hasattr(source, name):
            setattr(clone, name, getattr(source, name))
    linking = getattr(source, "light_linking", None)
    if linking is not None:
        clone.light_linking.receiver_collection = linking.receiver_collection
        clone.light_linking.blocker_collection = linking.blocker_collection
    names, random = _shading_inputs([slot.material for slot in source.material_slots])
    for name in names:
        # Add-on metadata would make the clone look like a source or an atlas.
        if name in source.keys() and not name.startswith(("mjworld_", "mujoco")):
            clone[name] = source[name]
    if random:
        warnings.append(f"{source.name}: its material uses Object Info Random, which a bake cannot reproduce; "
                        "the atlas shows a different random value.")
    # Generated coordinates follow the source mesh's texture space, not the
    # bounds of the evaluated copy.
    data = source.data
    if hasattr(data, "texspace_location") and hasattr(clone.data, "use_auto_texspace"):
        clone.data.use_auto_texspace = False
        clone.data.texspace_location = data.texspace_location
        clone.data.texspace_size = data.texspace_size


def _save_atlases(image, directory):
    from .exporter import save_image
    master_path = directory / "static_lighting_linear.exr"
    display_path = directory / "static_lightmap.png"
    # Float OpenEXR is scene-linear; the display transform is not baked in.
    save_image(image, master_path, file_format="OPEN_EXR", color_depth="16")
    profile = save_image(image, display_path, profile=LIGHTMAP_PROFILE)
    return master_path, display_path, profile


def _restore_source(obj, view_layer):
    obj.hide_render = bool(obj.get("mjworld_bake_original_hide_render", False))
    obj.hide_set(bool(obj.get("mjworld_bake_original_hide_viewport", False)), view_layer=view_layer)
    for key in _SOURCE_KEYS:
        if key in obj:
            del obj[key]


def _bake_directory(output_directory):
    raw = str(output_directory)
    if raw.startswith("//") and not bpy.data.filepath:
        raise BakeError(f"The bake folder '{raw}' is relative to the .blend file, which is not saved yet. "
                        "Save the .blend first, or choose an absolute folder.")
    directory = Path(bpy.path.abspath(raw)).expanduser().resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".mjworld-write-check-"):
            pass
    except OSError as exc:
        raise BakeError(f"Cannot write to the bake folder {directory}: {exc}") from exc
    return directory


def bake_static(context, output_directory, *, atlas_size=2048, samples=128, margin=8):
    """Bake fixed visual surfaces without changing source meshes or materials.

    Returns a JSON-safe summary. The generated object uses a packed linear HDR
    Emission texture; its display PNG is saved through LIGHTMAP_PROFILE (AgX)
    for MuJoCo. Exporters treat hidden ``mjworld_bake_source`` objects that an
    atlas lists as collision-only, and the ``mjworld_baked_atlas`` object as
    visual-only. Calling again replaces the previous atlas after a successful
    bake. ``restore_sources`` restores editing.

    Everything that can fail cheaply (the bake folder, Cycles, the AgX view)
    is checked before the Cycles pass. Faceless meshes are skipped and listed
    in the summary's ``skipped``.
    """
    from .exporter import ExportError, check_profile
    if context.mode != "OBJECT":
        raise BakeError("Switch to Object Mode before baking static lighting.")
    if not isinstance(atlas_size, int) or not 32 <= atlas_size <= 16384:
        raise BakeError("Atlas size must be an integer between 32 and 16384.")
    if not isinstance(samples, int) or samples < 1:
        raise BakeError("Bake samples must be a positive integer.")
    if not isinstance(margin, int) or not 0 <= margin < atlas_size // 2:
        raise BakeError("Bake margin must be nonnegative and smaller than half the atlas.")
    scene = context.scene
    if not hasattr(scene, "cycles"):
        raise BakeError("Enable the Cycles render add-on to bake static lighting.")
    try:
        check_profile(LIGHTMAP_PROFILE)
    except ExportError as exc:
        raise BakeError(f"Cannot save the lightmap: {exc}") from exc
    sources = sorted((obj for obj in scene.objects if obj.type == "MESH"
                      and not is_atlas(obj)
                      and not obj.get("mjworld_bake_clone", False)
                      and not obj.get("collision_only", False)
                      and _role(obj) not in {"COLLIDER", "EXCLUDE"}
                      and not _excluded(obj) and not _dynamic(obj)
                      and _source_visible(obj, context.view_layer)), key=lambda obj: obj.name)
    if not sources:
        raise BakeError("No fixed visible meshes to bake. Mark furniture Fixed and a decorative floor Visual only.")
    directory = _bake_directory(output_directory)
    previous_selection = list(context.selected_objects)
    previous_active = context.view_layer.objects.active
    previous_engine, previous_samples = scene.render.engine, scene.cycles.samples
    bake = scene.render.bake
    bake_fields = ("use_pass_color", "use_pass_direct", "use_pass_indirect", "use_pass_diffuse", "use_pass_emit",
                   "use_pass_glossy", "use_pass_transmission", "margin", "use_clear", "target",
                   "use_selected_to_active", "use_cage")
    previous_bake = {name: getattr(bake, name) for name in bake_fields if hasattr(bake, name)}
    hidden = {obj: (obj.hide_render, obj.hide_get(view_layer=context.view_layer)) for obj in scene.objects}
    previous_atlases = [obj for obj in scene.objects if is_atlas(obj)]
    dynamic_count = sum(1 for obj in scene.objects if obj.type == "MESH" and _dynamic(obj))
    copies, pieces, clone_meshes, baked, skipped, warnings = {}, [], [], [], [], []
    image = master = atlas = emission_material = None
    collection = bpy.data.collections.new("MuJoCo baked static visual")
    collection["mjworld_generated_bake"] = True
    scene.collection.children.link(collection)
    succeeded = False
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        image = bpy.data.images.new("MuJoCo working static-light atlas", width=atlas_size,
                                    height=atlas_size, alpha=False, float_buffer=True)
        image.colorspace_settings.name = "Linear Rec.709"
        image.generated_color = (0, 0, 0, 1)
        graph = context.evaluated_depsgraph_get()
        for source in sources:
            # Evaluate modifiers while preserving each object's local coordinate
            # system until after baking. Joining first changes Generated/Object
            # texture coordinates and can silently alter authored materials.
            source.hide_set(False, view_layer=context.view_layer)
            evaluated = source.evaluated_get(graph)
            mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
                                                   depsgraph=graph)
            clone_meshes.append(mesh)
            if not mesh.polygons:
                skipped.append(source.name)
                continue
            mesh.materials.clear()
            slots = [slot.material for slot in source.material_slots] or [None]
            for material in slots:
                mesh.materials.append(_material_copy(material, image, copies))
            clone = bpy.data.objects.new("MuJoCo bake piece " + source.name, mesh)
            # Tracked before anything else can fail, so cleanup always finds it.
            pieces.append(clone)
            clone["mjworld_bake_clone"] = True
            collection.objects.link(clone)
            clone.matrix_world = source.matrix_world.copy()
            _copy_shading(source, clone, warnings)
            uv = mesh.uv_layers.new(name="MuJoCoBakeUV")
            if uv is None:
                raise BakeError(f"{source.name}: the mesh already has the maximum of 8 UV maps; remove one before baking.")
            mesh.uv_layers.active = uv
            uv.active_render = True
            baked.append(source)
        if not baked:
            raise BakeError("No fixed visible meshes with faces to bake: " + ", ".join(skipped[:10]) + ".")
        _select(context, pieces)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0,
                                 area_weight=0.25, correct_aspect=True, scale_to_bounds=True)
        # Pack the multi-object UV selection once, so all sources share one atlas.
        bpy.ops.uv.pack_islands(rotate=True, rotate_method="CARDINAL", shape_method="AABB", margin_method="FRACTION",
                               margin=min(0.25, (margin + 1) / atlas_size))
        bpy.ops.object.mode_set(mode="OBJECT")
        for obj in hidden:
            # Only authored lights/world and fixed clones participate. In
            # particular, an open door or a can cannot leave a fixed shadow.
            if obj.type not in {"LIGHT", "CAMERA", "EMPTY"} or obj.is_instancer:
                obj.hide_render = True
        for name in ("use_pass_color", "use_pass_direct", "use_pass_indirect", "use_pass_diffuse", "use_pass_emit",
                     "use_clear"):
            setattr(bake, name, True)
        bake.use_pass_glossy = bake.use_pass_transmission = False
        bake.margin, bake.target = margin, "IMAGE_TEXTURES"
        bake.use_selected_to_active = bake.use_cage = False
        _select(context, pieces)
        # Combined diffuse + emission: lit albedo plus emissive panels and
        # screens. Glossy depends on the viewer, so it stays out of the atlas.
        bpy.ops.object.bake(type="COMBINED", pass_filter={"DIFFUSE", "EMIT", "DIRECT", "INDIRECT"},
                            margin=margin, use_clear=True, uv_layer="MuJoCoBakeUV")
        master_path, display_path, display_profile = _save_atlases(image, directory)
        master = bpy.data.images.load(str(master_path), check_existing=False)
        master.colorspace_settings.name = "Linear Rec.709"
        master["mjworld_display_png"] = str(display_path)
        for name, value in display_profile.items():
            master["mjworld_display_" + name] = value
        master["mjworld_generated_bake"] = True
        master.pack()
        emission_material = bpy.data.materials.new("MuJoCo baked static lighting")
        emission_material["mjworld_generated_bake"] = True
        if bpy.app.version < (5, 0, 0):
            emission_material.use_nodes = True
        nodes, links = emission_material.node_tree.nodes, emission_material.node_tree.links
        nodes.clear()
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = master
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = "MuJoCoBakeUV"
        emission = nodes.new("ShaderNodeEmission")
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(uv_node.outputs["UV"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs[0], output.inputs["Surface"])
        _select(context, pieces)
        bpy.ops.object.join()
        atlas = context.view_layer.objects.active
        pieces = [atlas]
        # The joined object keeps the active piece's properties.
        del atlas["mjworld_bake_clone"]
        atlas.name = "MuJoCo baked static visual"
        atlas.data.materials.clear()
        atlas.data.materials.append(emission_material)
        for polygon in atlas.data.polygons:
            polygon.material_index = 0
        atlas.mujoco_world.role = "VISUAL"
        atlas.mujoco_world.baked = True
        atlas["mjworld_baked_atlas"] = True
        atlas["mjworld_bake_sources"] = [source.name for source in baked]
        atlas["mjworld_bake_passes"] = BAKE_PASSES
        atlas["mjworld_master_exr"] = str(master_path)
        atlas["mjworld_display_png"] = str(display_path)
        for source in baked:
            if not source.get("mjworld_bake_source", False):
                legacy = bool(source.get("preserved_original_static_visual", False))
                source["mjworld_bake_original_hide_render"] = False if legacy else hidden[source][0]
                source["mjworld_bake_original_hide_viewport"] = False if legacy else hidden[source][1]
            source["mjworld_bake_source"] = True
        # Marked sources this bake no longer contains (now movable, left out
        # or faceless) are restored, so they are not hidden without an atlas.
        stale = [obj for obj in scene.objects if is_bake_source(obj) and obj not in baked]
        _remove_objects(previous_atlases)
        # The base names are free only now that the previous bake is gone.
        # Taking them again keeps a rebake from alternating with '.001' names.
        atlas.name = atlas.data.name = collection.name = "MuJoCo baked static visual"
        master.name = master_path.name
        emission_material.name = "MuJoCo baked static lighting"
        succeeded = True
        return {"object": atlas.name, "source_count": len(baked), "atlas_size": atlas_size,
                "samples": samples, "master_exr": str(master_path), "display_png": str(display_path),
                "display_view_transform": display_profile["view_transform"], "display_profile": display_profile,
                "passes": BAKE_PASSES, "skipped": skipped, "restored": [obj.name for obj in stale],
                "warnings": warnings, "excluded_dynamic_count": dynamic_count}
    finally:
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        scene.render.engine, scene.cycles.samples = previous_engine, previous_samples
        for name, value in previous_bake.items():
            setattr(bake, name, value)
        for obj, (hide_render, hide_viewport) in hidden.items():
            if _exists(obj):
                obj.hide_render = hide_render
                obj.hide_set(hide_viewport, view_layer=context.view_layer)
        if succeeded:
            for source in baked:
                source.hide_render = True
                source.hide_set(True, view_layer=context.view_layer)
            for obj in stale:
                _restore_source(obj, context.view_layer)
        else:
            _remove_objects(pieces)
            if _exists(emission_material, bpy.data.materials) and emission_material.users == 0:
                bpy.data.materials.remove(emission_material)
            if _exists(master, bpy.data.images) and master.users == 0:
                bpy.data.images.remove(master)
        # The working data is private to this bake. Joined pieces leave their
        # meshes orphaned, and those still hold the material copies, so remove
        # orphans first, then the copies and the working image unconditionally.
        for mesh in clone_meshes:
            if _exists(mesh, bpy.data.meshes) and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for material in copies.values():
            if _exists(material, bpy.data.materials):
                bpy.data.materials.remove(material)
        if _exists(image, bpy.data.images):
            bpy.data.images.remove(image)
        if _exists(collection, bpy.data.collections) and not collection.objects:
            bpy.data.collections.remove(collection)
        if succeeded:
            _select(context, [atlas])
        else:
            _select(context, [obj for obj in previous_selection if _exists(obj)])
            if previous_active and _exists(previous_active):
                context.view_layer.objects.active = previous_active


def restore_sources(context):
    """Remove the generated atlas and restore source object visibility.

    Leftover pieces of a failed bake are removed too. Returns the number of
    restored source objects.
    """
    scene = context.scene
    count = 0
    for obj in list(scene.objects):
        if is_bake_source(obj):
            _restore_source(obj, context.view_layer)
            count += 1
    _remove_objects([obj for obj in scene.objects if is_atlas(obj) or obj.get("mjworld_bake_clone", False)])
    return count
