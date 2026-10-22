"""A small, dependency-free Blender world exporter for MuJoCo 3.15+."""

bl_info = {
    "name": "MuJoCo World — Easy Export",
    "author": "Walkie / Codex",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > MuJoCo",
    "description": "Make fixed furniture, movable objects, doors and a plane floor; export a portable MuJoCo world",
    "category": "Import-Export",
}

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper, ImportHelper


ROLES = [
    ("STATIC", "Fixed furniture", "Stays in place and has collision"),
    ("FREE", "Movable object", "Falls, can be picked up and dragged"),
    ("HINGE", "Door", "Rotates around the object origin; no motor or handle required"),
    ("VISUAL", "Visual only", "Visible but never collides, useful for a decorative floor"),
    ("COLLIDER", "Collision helper", "Invisible fixed collision shape; use with a separate detailed visual mesh"),
    ("EXCLUDE", "Leave out", "Do not export this object"),
]


def role_changed(settings, _context):
    obj = settings.id_data
    if not isinstance(obj, bpy.types.Object):
        return
    if settings.role == "COLLIDER":
        if not settings.get("_helper_wire", False):
            settings["_before_helper_display"] = obj.display_type
        obj.display_type = "WIRE"
        settings["_helper_wire"] = True
    elif settings.get("_helper_wire", False):
        obj.display_type = settings.get("_before_helper_display", "TEXTURED")
        settings["_helper_wire"] = False


class MJWORLD_ObjectSettings(PropertyGroup):
    body: BoolProperty(name="Rigid body group", default=False,
                       description="An Empty owns all descendant visual and collision parts as one rigid body")
    role: EnumProperty(name="Object type", items=ROLES, default="STATIC", update=role_changed)
    collision: EnumProperty(name="Collision shape", items=[
        ("BOX", "Box", "Fast bounding box; use separate tabletop and legs to leave the space under a table open"),
        ("SPHERE", "Sphere", "Enclosing sphere; useful for a ball or apple"),
        ("CYLINDER", "Cylinder (local Z)", "Enclosing cylinder along the object's local Z axis"),
        ("CONVEX", "Convex mesh", "Convex hull, NOT the original concave mesh; hollow areas will be filled"),
    ], default="BOX")
    mass: FloatProperty(name="Weight (kg)", default=1.0, min=0.001, soft_max=30)
    friction: FloatVectorProperty(name="Friction", size=3, default=(0.8, 0.005, 0.0001), min=0.0)
    hinge_axis: EnumProperty(name="Hinge direction", items=[("Z", "Vertical (local Z)", "Usually correct for a door"), ("X", "Local X", ""), ("Y", "Local Y", "")], default="Z")
    hinge_min: FloatProperty(name="Open limit (degrees)", default=-110.0, min=-360.0, max=360.0)
    hinge_max: FloatProperty(name="Closed limit (degrees)", default=0.0, min=-360.0, max=360.0)
    hinge_start: FloatProperty(name="Starting angle (degrees)", default=0.0, min=-360.0, max=360.0)
    damping: FloatProperty(name="Hinge resistance", default=0.3, min=0.0, soft_max=10)
    baked: BoolProperty(name="Image already contains lighting", default=False,
                        description="Use the baked texture without treating it as an ordinary dark albedo; also detected for an Emission shader")
    advanced: BoolProperty(name="Advanced", default=False)


class MJWORLD_SceneSettings(PropertyGroup):
    selected_only: BoolProperty(name="Export selected objects only", default=False)
    plane: BoolProperty(name="Add a real MuJoCo floor plane", default=True,
                        description="One infinite invisible collision plane; decorative floor meshes should be Visual only")
    plane_height: FloatProperty(name="Floor height (m)", default=0.0)
    # Environment integration can require masks different from an isolated world.
    plane_contype: StringProperty(name="Plane contact type", default="1")
    plane_conaffinity: StringProperty(name="Plane contact mask", default="1")
    object_contype: StringProperty(name="Fixed contact type", default="1")
    object_conaffinity: StringProperty(name="Fixed contact mask", default="1")
    # Blank door masks follow the fixed masks, as doors did before they had their own.
    hinge_contype: StringProperty(name="Door contact type", default="",
                                  description="Contact type of door collision geoms, and of everything that moves with a door. Empty uses the fixed contact type")
    hinge_conaffinity: StringProperty(name="Door contact mask", default="",
                                      description="Contact mask of door collision geoms, and of everything that moves with a door. Empty uses the fixed contact mask")
    movable_contype: StringProperty(name="Movable contact type", default="1")
    movable_conaffinity: StringProperty(name="Movable contact mask", default="1")
    texture_mode: EnumProperty(name="Texture colour", items=[
        ("DISPLAY", "Match PNG appearance", "Explicit linear texture input preserves display-referred PNG values in the native MuJoCo viewer"),
        ("SRGB", "sRGB decode (custom renderer)", "Decode PNG values into linear light; your renderer must apply its display transform"),
    ], default="DISPLAY")
    baked_emission: FloatProperty(name="Baked texture brightness", default=0.2, min=0.0, max=1.0)
    advanced: BoolProperty(name="World advanced settings", default=False)
    last_export: StringProperty(name="Last export", default="", subtype="FILE_PATH")
    asset_directory: StringProperty(name="Shared asset folder", default="", subtype="DIR_PATH",
                                   description="Optional shared folder, relative to the blend file. Empty uses an assets folder beside each exported XML")
    bake_directory: StringProperty(name="Bake folder", default="//baked_lighting", subtype="DIR_PATH")
    bake_size: EnumProperty(name="Atlas resolution", items=[("512", "512", "Quick preview"), ("1024", "1024", "Small world"), ("2048", "2048", "Room details"), ("4096", "4096", "Large world")], default="2048")
    bake_samples: IntProperty(name="Bake samples", default=128, min=1, max=4096)


class MJWORLD_OT_role(Operator):
    bl_idname = "mjworld.set_role"
    bl_label = "Set object type"
    bl_options = {"REGISTER", "UNDO"}
    role: EnumProperty(items=ROLES)

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(o.type in {"MESH", "EMPTY"} for o in context.selected_objects)

    def execute(self, context):
        count = 0
        for obj in context.selected_objects:
            if obj.type not in {"MESH", "EMPTY"}:
                continue
            if obj.type == "EMPTY":
                obj.mujoco_world.body = True
            obj.mujoco_world.role = self.role
            if self.role == "HINGE":
                obj.mujoco_world.mass = 8.0
                obj.mujoco_world.collision = "BOX"
            count += 1
        self.report({"INFO"}, f"Set {count} object(s) to {dict((r[0], r[1]) for r in ROLES)[self.role]}")
        return {"FINISHED"}


class MJWORLD_OT_hinge_cursor(Operator):
    bl_idname = "mjworld.hinge_at_cursor"
    bl_label = "Set hinge at 3D cursor"
    bl_description = "Move the selected door's origin to the 3D cursor without moving its geometry"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and context.object is not None and context.object.type in {"MESH", "EMPTY"}

    def execute(self, context):
        # Use a narrow temporary selection so an adjacent doorframe is unaffected.
        selected = list(context.selected_objects)
        active = context.object
        if active.type == "EMPTY":
            children = [(obj, obj.matrix_world.copy()) for obj in active.children]
            world = active.matrix_world.copy()
            world.translation = context.scene.cursor.location
            active.matrix_world = world
            context.view_layer.update()
            for obj, matrix in children:
                obj.matrix_world = matrix
            active.mujoco_world.body = True
        else:
            # origin_set moves the mesh data, which would shift every other
            # object that uses the same mesh.
            mesh = active.data
            users = mesh.users - int(mesh.use_fake_user)
            if users > 1:
                self.report({"ERROR"}, f"{active.name}: its mesh '{mesh.name}' is also used by {users - 1} other object(s), "
                                       "which would move too. Run Object > Relation > Make Single User > Object & Data first, "
                                       "or group the door parts under a body Empty and set the hinge on it.")
                return {"CANCELLED"}
            try:
                bpy.ops.object.select_all(action="DESELECT")
                active.select_set(True)
                result = bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
            except RuntimeError as exc:
                result, error = {"CANCELLED"}, str(exc)
            else:
                error = "Blender could not move the origin."
            finally:
                bpy.ops.object.select_all(action="DESELECT")
                for obj in selected:
                    obj.select_set(True)
                context.view_layer.objects.active = active
            if "FINISHED" not in result:
                self.report({"ERROR"}, f"{active.name}: {error}")
                return {"CANCELLED"}
        active.mujoco_world.role = "HINGE"
        self.report({"INFO"}, "Hinge set. Local Z is the default vertical hinge direction.")
        return {"FINISHED"}


class MJWORLD_OT_group(Operator):
    bl_idname = "mjworld.group_parts"
    bl_label = "Group selected parts"
    bl_description = "Combine visible pieces and collision helpers into one movable body without joining or moving their meshes"
    bl_options = {"REGISTER", "UNDO"}
    role: EnumProperty(items=[("FREE", "Movable object", "One mass and free joint for all selected parts"), ("STATIC", "Fixed furniture", "One fixed body with compound collision"), ("HINGE", "Door", "One hinged body; pivot starts at the 3D cursor")], default="FREE")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and bool(context.selected_objects)

    def execute(self, context):
        from .project import group_selected
        try:
            obj = group_selected(context, self.role)
        except (ValueError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {obj.name}. Set its total mass; child Collision helpers stay invisible.")
        return {"FINISHED"}


class MJWORLD_OT_adopt(Operator, ImportHelper):
    bl_idname = "mjworld.adopt_mjcf"
    bl_label = "Link existing world physics…"
    bl_description = "Attach exact body, joint and compound collider metadata from a standalone MJCF; match existing Blender object names"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".xml"
    filter_glob: StringProperty(default="*.xml", options={"HIDDEN"})

    def execute(self, context):
        from .project import adopt_mjcf
        try:
            report = adopt_mjcf(context, self.filepath)
        except (OSError, ValueError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Linked {report['bodies']} bodies and {report['geoms']} analytic/visual parts. Blender edits preserved.")
        return {"FINISHED"}


class MJWORLD_OT_bake(Operator):
    bl_idname = "mjworld.bake_lighting"
    bl_label = "Bake static lighting"
    bl_description = "Bake fixed furniture and walls to one UV atlas; keep movable props and the door dynamic"

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        from .baking import bake_static
        settings = context.scene.mujoco_world
        try:
            report = bake_static(context, bpy.path.abspath(settings.bake_directory), atlas_size=int(settings.bake_size), samples=settings.bake_samples)
        except (OSError, ValueError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Baked {report['source_count']} fixed parts. Export world to use the lighting in MuJoCo.")
        return {"FINISHED"}


class MJWORLD_OT_restore(Operator):
    bl_idname = "mjworld.restore_sources"
    bl_label = "Edit original surfaces"
    bl_description = "Restore the original fixed surfaces and remove the generated lighting atlas so you can edit and rebake"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .baking import restore_sources
        restore_sources(context)
        return {"FINISHED"}


class MJWORLD_OT_walkie_contacts(Operator):
    bl_idname = "mjworld.walkie_contacts"
    bl_label = "Use Walkie contact masks"
    bl_description = "Set the current Walkie environment masks; its wheel/plane pairs still belong in the robot integration"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.mujoco_world
        settings.plane_contype, settings.plane_conaffinity = "1", "32"
        settings.object_contype, settings.object_conaffinity = "64", "56"
        # robot_bringup.world_loader treats every geom that moves with a joint,
        # doors included, as a movable object.
        settings.hinge_contype, settings.hinge_conaffinity = "32", "121"
        settings.movable_contype, settings.movable_conaffinity = "32", "121"
        self.report({"INFO"}, "Walkie masks set. Preserve the robot's explicit wheel/floor contact pairs when merging.")
        return {"FINISHED"}


class MJWORLD_OT_export(Operator, ExportHelper):
    bl_idname = "mjworld.export_world"
    bl_label = "Export MuJoCo world"
    bl_description = "Write scene.xml, relative mesh/texture assets and a readable export report"
    filename_ext = ".xml"
    filter_glob: StringProperty(default="*.xml", options={"HIDDEN"})

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.abspath("//mujoco_world/scene.xml")
        return super().invoke(context, event)

    def execute(self, context):
        from .exporter import ExportError, export_world
        try:
            result = export_world(context, self.filepath)
        except (ExportError, OSError, ValueError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.scene.mujoco_world.last_export = self.filepath
        warnings = result["warnings"]
        self.report({"WARNING" if warnings else "INFO"},
                    f"Exported {result['objects']} objects. " +
                    (f"{len(warnings)} notes: see export_report.json." if warnings else "Ready to open in MuJoCo."))
        return {"FINISHED"}


class MJWORLD_PT_main(Panel):
    bl_label = "MuJoCo World"
    bl_idname = "MJWORLD_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MuJoCo"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mujoco_world
        layout.label(text="1. Select an object and choose its type")
        grid = layout.grid_flow(columns=2, even_columns=True)
        for label, role in [("Fixed furniture", "STATIC"), ("Movable object", "FREE"), ("Door", "HINGE"), ("Visual floor", "VISUAL")]:
            op = grid.operator("mjworld.set_role", text=label)
            op.role = role
        obj = context.object
        if obj and obj.type in {"MESH", "EMPTY"}:
            box = layout.box()
            box.label(text=obj.name, icon="EMPTY_AXIS" if obj.type == "EMPTY" else "MESH_DATA")
            box.prop(obj.mujoco_world, "role", text="")
            role = obj.mujoco_world.role
            if role == "COLLIDER":
                box.label(text="Invisible in MuJoCo; keep visible here.", icon="INFO")
            if role == "HINGE":
                box.label(text="2. Put the 3D cursor at the hinge")
                box.operator("mjworld.hinge_at_cursor", icon="PIVOT_CURSOR")
                box.prop(obj.mujoco_world, "hinge_axis")
                box.prop(obj.mujoco_world, "hinge_start")
            if role in {"FREE", "HINGE"}:
                box.prop(obj.mujoco_world, "mass")
            if obj.type == "MESH" and role not in {"VISUAL", "EXCLUDE"}:
                box.prop(obj.mujoco_world, "collision")
            if role != "EXCLUDE":
                box.prop(obj.mujoco_world, "advanced", icon="TRIA_DOWN" if obj.mujoco_world.advanced else "TRIA_RIGHT", emboss=False)
                if obj.mujoco_world.advanced:
                    if role != "COLLIDER":
                        box.prop(obj.mujoco_world, "baked")
                    if role != "VISUAL":
                        box.prop(obj.mujoco_world, "friction")
                    if role == "HINGE":
                        box.prop(obj.mujoco_world, "hinge_min")
                        box.prop(obj.mujoco_world, "hinge_max")
                        box.prop(obj.mujoco_world, "damping")
        layout.separator()
        layout.operator("mjworld.group_parts", text="Group selected parts as one body", icon="OUTLINER_OB_GROUP_INSTANCE")
        layout.label(text="Hollow cup/plate: visual mesh + separate helpers.", icon="INFO")
        layout.separator()
        layout.label(text="2. Optional: bake the room lighting")
        layout.operator("mjworld.bake_lighting", icon="RENDER_STILL")
        if any(o.get("mjworld_baked_atlas") or o.get("baked_static_visual") for o in context.scene.objects):
            layout.operator("mjworld.restore_sources", icon="LOOP_BACK")
        layout.separator()
        layout.label(text="3. Export your world")
        layout.prop(settings, "selected_only")
        layout.prop(settings, "plane")
        if settings.plane:
            layout.prop(settings, "plane_height")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("mjworld.export_world", text="Export world…", icon="EXPORT")
        layout.label(text="Visuals and collision helpers export.", icon="INFO")
        layout.prop(settings, "advanced", icon="TRIA_DOWN" if settings.advanced else "TRIA_RIGHT", emboss=False)
        if settings.advanced:
            layout.prop(settings, "asset_directory")
            layout.prop(settings, "bake_directory")
            layout.prop(settings, "bake_size")
            layout.prop(settings, "bake_samples")
            layout.prop(settings, "texture_mode")
            layout.prop(settings, "baked_emission")
            layout.label(text="Contact masks (integers, for robot integration)")
            layout.prop(settings, "plane_contype")
            layout.prop(settings, "plane_conaffinity")
            layout.prop(settings, "object_contype")
            layout.prop(settings, "object_conaffinity")
            layout.prop(settings, "hinge_contype")
            layout.prop(settings, "hinge_conaffinity")
            layout.prop(settings, "movable_contype")
            layout.prop(settings, "movable_conaffinity")
            layout.operator("mjworld.walkie_contacts")
            layout.operator("mjworld.adopt_mjcf", icon="IMPORT")


def export_menu(self, context):
    self.layout.operator(MJWORLD_OT_export.bl_idname, text="MuJoCo World (.xml)")


CLASSES = (MJWORLD_ObjectSettings, MJWORLD_SceneSettings, MJWORLD_OT_role, MJWORLD_OT_hinge_cursor, MJWORLD_OT_group, MJWORLD_OT_adopt, MJWORLD_OT_bake, MJWORLD_OT_restore, MJWORLD_OT_walkie_contacts, MJWORLD_OT_export, MJWORLD_PT_main)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.mujoco_world = PointerProperty(type=MJWORLD_ObjectSettings)
    bpy.types.Scene.mujoco_world = PointerProperty(type=MJWORLD_SceneSettings)
    bpy.types.TOPBAR_MT_file_export.append(export_menu)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(export_menu)
    del bpy.types.Scene.mujoco_world
    del bpy.types.Object.mujoco_world
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
