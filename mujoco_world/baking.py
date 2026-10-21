"""Non-destructive diffuse-light atlases for an authored Blender world.

The user's materials, lights and world supply the appearance. Only fixed visible
surfaces participate: movable bodies, doors and collision helpers neither enter
the atlas nor leave permanent shadows in it. Original objects and their physics
metadata remain in the file for editing and collision export.
"""

import math
from pathlib import Path

import bpy


class BakeError(ValueError):
    pass


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


def _select(context, objects):
    for obj in context.view_layer.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    context.view_layer.objects.active = objects[0] if objects else None


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
    material.use_nodes = True
    nodes = material.node_tree.nodes
    target = nodes.new("ShaderNodeTexImage")
    target.name = "MuJoCo atlas bake target"
    target.image = image
    for node in nodes:
        node.select = node == target
    nodes.active = target
    copies[key] = material
    return material


def _save_atlases(image, scene, directory):
    settings = scene.render.image_settings
    previous = {name: getattr(settings, name) for name in
                ("file_format", "color_mode", "color_depth", "exr_codec")}
    view = scene.view_settings
    previous_view, previous_look = view.view_transform, view.look
    master_path = directory / "static_lighting_linear.exr"
    display_path = directory / "static_lightmap.png"
    try:
        settings.file_format = "OPEN_EXR"
        settings.color_mode = "RGB"
        settings.color_depth = "16"
        settings.exr_codec = "ZIP"
        # Float OpenEXR is scene-linear; the display transform is not baked in.
        image.save_render(str(master_path), scene=scene)
        view.view_transform = "AgX"
        display_profile = {name: getattr(view, name) for name in ("view_transform", "look", "exposure", "gamma")}
        settings.file_format = "PNG"
        settings.color_mode = "RGB"
        settings.color_depth = "8"
        image.save_render(str(display_path), scene=scene)
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)
        view.view_transform = previous_view
        view.look = previous_look
    return master_path, display_path, display_profile


def bake_static(context, output_directory, *, atlas_size=2048, samples=128, margin=8):
    """Bake fixed visual surfaces without changing source meshes or materials.

    Returns a JSON-safe summary. The generated object uses a packed linear HDR
    Emission texture; its display PNG is saved through AgX for MuJoCo. Exporters
    should treat hidden ``mjworld_bake_source`` objects as collision-only and the
    ``mjworld_baked_atlas`` object as visual-only. Calling again replaces the
    previous atlas after a successful bake. ``restore_sources`` restores editing.
    """
    if context.mode != "OBJECT":
        raise BakeError("Switch to Object Mode before baking static lighting.")
    if not isinstance(atlas_size, int) or not 32 <= atlas_size <= 16384:
        raise BakeError("Atlas size must be an integer between 32 and 16384.")
    if not isinstance(samples, int) or samples < 1:
        raise BakeError("Bake samples must be a positive integer.")
    if not isinstance(margin, int) or not 0 <= margin < atlas_size // 2:
        raise BakeError("Bake margin must be nonnegative and smaller than half the atlas.")
    scene = context.scene
    directory = Path(bpy.path.abspath(str(output_directory))).expanduser().resolve()
    sources = sorted((obj for obj in scene.objects if obj.type == "MESH"
                      and not obj.get("mjworld_baked_atlas", False)
                      and not obj.get("baked_static_visual", False)
                      and not obj.get("collision_only", False)
                      and _role(obj) not in {"COLLIDER", "EXCLUDE"}
                      and not _excluded(obj) and not _dynamic(obj)
                      and _source_visible(obj, context.view_layer)), key=lambda obj: obj.name)
    if not sources:
        raise BakeError("No fixed visible meshes to bake. Mark furniture Fixed and a decorative floor Visual only.")
    directory.mkdir(parents=True, exist_ok=True)
    previous_selection = list(context.selected_objects)
    previous_active = context.view_layer.objects.active
    previous_engine, previous_samples = scene.render.engine, scene.cycles.samples
    bake = scene.render.bake
    bake_fields = ("use_pass_color", "use_pass_direct", "use_pass_indirect", "margin", "use_clear", "target",
                   "use_selected_to_active", "use_cage")
    previous_bake = {name: getattr(bake, name) for name in bake_fields}
    hidden = {obj: (obj.hide_render, obj.hide_get(view_layer=context.view_layer)) for obj in scene.objects}
    previous_atlases = [obj for obj in scene.objects if obj.get("mjworld_baked_atlas", False)
                       or obj.get("baked_static_visual", False)]
    dynamic_count = sum(1 for obj in scene.objects if obj.type == "MESH" and _dynamic(obj))
    copies, pieces = {}, []
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
            if not mesh.polygons:
                bpy.data.meshes.remove(mesh)
                raise BakeError(f"{source.name}: mesh has no faces to bake.")
            mesh.materials.clear()
            slots = [slot.material for slot in source.material_slots] or [None]
            for material in slots:
                mesh.materials.append(_material_copy(material, image, copies))
            clone = bpy.data.objects.new("MuJoCo bake piece " + source.name, mesh)
            collection.objects.link(clone)
            clone.matrix_world = source.matrix_world.copy()
            uv = mesh.uv_layers.new(name="MuJoCoBakeUV")
            mesh.uv_layers.active = uv
            uv.active_render = True
            pieces.append(clone)
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
        for name in ("use_pass_color", "use_pass_direct", "use_pass_indirect", "use_clear"):
            setattr(bake, name, True)
        bake.margin, bake.target = margin, "IMAGE_TEXTURES"
        bake.use_selected_to_active = bake.use_cage = False
        _select(context, pieces)
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR", "DIRECT", "INDIRECT"},
                            margin=margin, use_clear=True, uv_layer="MuJoCoBakeUV")
        master_path, display_path, display_profile = _save_atlases(image, scene, directory)
        master = bpy.data.images.load(str(master_path), check_existing=False)
        master.colorspace_settings.name = "Linear Rec.709"
        master["mjworld_display_png"] = str(display_path)
        for name, value in display_profile.items():
            master["mjworld_display_" + name] = value
        master["mjworld_generated_bake"] = True
        master.pack()
        emission_material = bpy.data.materials.new("MuJoCo baked static lighting")
        emission_material["mjworld_generated_bake"] = True
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
        atlas.name = "MuJoCo baked static visual"
        atlas.data.materials.clear()
        atlas.data.materials.append(emission_material)
        for polygon in atlas.data.polygons:
            polygon.material_index = 0
        atlas.mujoco_world.role = "VISUAL"
        atlas.mujoco_world.baked = True
        atlas["mjworld_baked_atlas"] = True
        atlas["mjworld_bake_sources"] = [source.name for source in sources]
        atlas["mjworld_bake_passes"] = "Diffuse color + direct + indirect; fixed geometry only"
        atlas["mjworld_master_exr"] = str(master_path)
        atlas["mjworld_display_png"] = str(display_path)
        for source in sources:
            if not source.get("mjworld_bake_source", False):
                legacy = bool(source.get("preserved_original_static_visual", False))
                source["mjworld_bake_original_hide_render"] = False if legacy else hidden[source][0]
                source["mjworld_bake_original_hide_viewport"] = False if legacy else hidden[source][1]
            source["mjworld_bake_source"] = True
        _remove_objects(previous_atlases)
        succeeded = True
        return {"object": atlas.name, "source_count": len(sources), "atlas_size": atlas_size,
                "samples": samples, "master_exr": str(master_path), "display_png": str(display_path),
                "display_view_transform": "AgX",
                "excluded_dynamic_count": dynamic_count}
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
            for source in sources:
                source.hide_render = True
                source.hide_set(True, view_layer=context.view_layer)
            _select(context, [atlas])
        else:
            _remove_objects(pieces)
            if _exists(emission_material, bpy.data.materials) and emission_material.users == 0:
                bpy.data.materials.remove(emission_material)
            if _exists(master, bpy.data.images) and master.users == 0:
                bpy.data.images.remove(master)
            _select(context, [obj for obj in previous_selection if _exists(obj)])
            if previous_active and _exists(previous_active):
                context.view_layer.objects.active = previous_active
        for material in copies.values():
            if material.users == 0:
                bpy.data.materials.remove(material)
        if _exists(image, bpy.data.images) and image.users == 0:
            bpy.data.images.remove(image)
        if _exists(collection, bpy.data.collections) and not collection.objects:
            bpy.data.collections.remove(collection)


def restore_sources(context):
    """Remove the generated atlas and restore source object visibility."""
    scene = context.scene
    count = 0
    for obj in list(scene.objects):
        if obj.get("mjworld_bake_source", False) or obj.get("preserved_original_static_visual", False):
            obj.hide_render = bool(obj.get("mjworld_bake_original_hide_render", False))
            obj.hide_set(bool(obj.get("mjworld_bake_original_hide_viewport", False)), view_layer=context.view_layer)
            for key in ("mjworld_bake_source", "mjworld_bake_original_hide_render", "mjworld_bake_original_hide_viewport",
                        "preserved_original_static_visual"):
                if key in obj:
                    del obj[key]
            count += 1
    _remove_objects([obj for obj in scene.objects if obj.get("mjworld_baked_atlas", False)
                     or obj.get("baked_static_visual", False)])
    return count
