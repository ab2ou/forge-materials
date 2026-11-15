# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators and sidebar panels (View3D > Sidebar > Forge Materials)."""

import json
import os
import subprocess
import tempfile

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       StringProperty)

from . import bake, library, looks, nodes, procedural

_PREVIEWS = {"pcoll": None}
_ENUM = {"key": None, "items": [("NONE", "No materials", "Scan the library first", 0, 0)]}


def prefs():
    return bpy.context.preferences.addons[__package__].preferences


def cache_path():
    try:
        base = bpy.utils.extension_path_user(__package__, create=True)
    except (AttributeError, ValueError, TypeError):
        base = os.path.join(tempfile.gettempdir(), "forge_materials")
    return os.path.join(base, "library_cache.json")


# --------------------------------------------------------------------------
# library enum with thumbnails
# --------------------------------------------------------------------------

def _icon(item):
    pcoll = _PREVIEWS["pcoll"]
    if pcoll is None:
        return 0
    name = item["name"]
    if name not in pcoll:
        try:
            pcoll.load(name, item["thumb"], "IMAGE")
        except (KeyError, RuntimeError):
            return 0
    return pcoll[name].icon_id


def _filtered(props):
    cat = props.category
    q = props.search.strip().lower()
    out = []
    for it in library.items():
        if cat == "FAV":
            if it["name"] not in looks.favourites():
                continue
        elif cat != "ALL" and it["category"] != cat:
            continue
        if q and q not in it["name"].lower():
            continue
        out.append(it)
    return out


def library_enum_items(self, context):
    key = (self.category, self.search, len(library.items()), library._LIBRARY["root"])
    if _ENUM["key"] == key:
        return _ENUM["items"]
    items = []
    for i, it in enumerate(_filtered(self)):
        items.append((it["name"], it["label"], f"{it['label']} ({library.CATEGORY_LABELS.get(it['category'], it['category'])})",
                      _icon(it), i))
    if not items:
        items = [("NONE", "No materials", "Scan the library or change the filter", 0, 0)]
    _ENUM.update(key=key, items=items)
    return items


CATEGORY_ITEMS = [("ALL", "All", ""), ("FAV", "Favourites", "")] + [
    (k, lbl, "") for k, lbl, *_ in library.CATEGORIES] + [("OTHER", "Other", "")]


_LOOK_ENUM = {"items": [("NONE", "No saved looks", "")]}


def _look_enum():
    items = [(lk["file"], lk.get("name", "look"), lk.get("source", ""))
             for lk in looks.list_looks()]
    _LOOK_ENUM["items"] = items or [("NONE", "No saved looks", "")]
    return _LOOK_ENUM["items"]


class FM_Props(bpy.types.PropertyGroup):
    category: EnumProperty(name="Category", items=CATEGORY_ITEMS, default="ALL")
    search: StringProperty(name="Search", default="", options={"TEXTEDIT_UPDATE"})
    library_item: EnumProperty(name="Material", items=library_enum_items)
    mapping: EnumProperty(name="Mapping", default="UV", items=[
        ("UV", "UV", "Use the mesh UV map (best for baking, uses the normal map)"),
        ("BOX", "Box", "Object-space triplanar projection, no UVs needed"),
    ])
    tex_scale: FloatProperty(name="Pattern Size", default=1.0, min=0.05, max=20.0,
                             description="1 = real-world size of the material")
    all_slots: BoolProperty(name="All Slots", default=False,
                            description="Replace every material slot instead of the active one")
    proc_preset: EnumProperty(name="Procedural", items=[
        (k, f"{v['label']}  ({library.CATEGORY_LABELS.get(v['category'], v['category'])})", "")
        for k, v in procedural.PRESETS.items()])

    look_name: StringProperty(name="Name", default="", description="Name for the saved look")
    pack_engine: EnumProperty(name="Engine", default="UNREAL", items=[
        ("UNREAL", "Unreal Engine", "T_Asset_BC / _N (DirectX) / _ORM + SM_Asset.fbx"),
        ("UNITY", "Unity", "Albedo / Normal (OpenGL) / MetallicSmoothness / AO"),
        ("GODOT", "Godot", "albedo / normal (OpenGL) / orm"),
        ("GENERIC", "Generic PBR", "BaseColor / Roughness / Metallic / Normal / AO / Height"),
    ])
    pack_asset: StringProperty(name="Asset Name", default="",
                               description="Base name for files (default: active object)")
    pack_fbx: BoolProperty(name="Export FBX", default=True,
                           description="Also export the selected meshes as one FBX next to the maps")
    look_item: EnumProperty(name="Look", items=lambda self, ctx: _look_enum())

    bake_res: EnumProperty(name="Resolution", default="2048", items=[
        (str(r), f"{r // 1024}K" if r >= 1024 else str(r), "")
        for r in (512, 1024, 2048, 4096, 8192)])
    bake_samples: IntProperty(name="Samples", default=32, min=1, max=4096,
                              description="Cycles samples for baking (masks need ~16+)")
    bake_dir: StringProperty(name="Output", default="//FM_Bake/", subtype="DIR_PATH")
    bake_background: BoolProperty(name="Bake in Background", default=True,
                                  description="Run the bake in a separate Blender process "
                                              "so you can keep working")
    bake_gpu: BoolProperty(name="Use GPU", default=True)
    bake_region_masks: BoolProperty(
        name="Smooth Region Borders", default=True,
        description="Objects split into regions: blend the region materials with per-texel masks, so "
                    "borders run through faces at texture precision (slower: one bake per region)")
    ch_color: BoolProperty(name="Base Color", default=True)
    ch_rough: BoolProperty(name="Roughness", default=True)
    ch_metal: BoolProperty(name="Metallic", default=True)
    ch_normal_gl: BoolProperty(name="Normal (OpenGL)", default=True,
                               description="Blender, Unity, Godot")
    ch_normal_dx: BoolProperty(name="Normal (DirectX)", default=True,
                               description="Unreal Engine")
    ch_ao: BoolProperty(name="AO", default=False)
    ch_height: BoolProperty(name="Height", default=False)
    ch_orm: BoolProperty(name="ORM (packed)", default=True,
                         description="R=AO, G=Roughness, B=Metallic (Unreal)")


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

class FM_OT_scan_library(bpy.types.Operator):
    bl_idname = "fm.scan_library"
    bl_label = "Scan Library"
    bl_description = "Read the texture library folder (read-only)"

    def execute(self, context):
        p = prefs()
        root = bpy.path.abspath(p.library_path)
        if not os.path.isdir(root):
            self.report({"ERROR"}, f"Library folder not found: {root}")
            return {"CANCELLED"}
        items = library.scan(root, p.preferred_res)
        library.save_cache(cache_path())
        _ENUM["key"] = None
        cats = {}
        for it in items:
            cats[it["category"]] = cats.get(it["category"], 0) + 1
        summary = ", ".join(f"{k.title()} {v}" for k, v in sorted(cats.items()))
        self.report({"INFO"}, f"{len(items)} materials found ({summary})")
        return {"FINISHED"}


def _targets(context):
    return [o for o in context.selected_objects if o.type == "MESH"]


class FM_OT_apply_library(bpy.types.Operator):
    bl_idname = "fm.apply_library"
    bl_label = "Apply Material"
    bl_description = ("Build the selected library material and assign it to the selected "
                      "meshes. Geometry is never modified; replaced materials are kept")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(_targets(context))

    def execute(self, context):
        props = context.scene.fm
        item = library.get(props.library_item)
        if not item:
            self.report({"ERROR"}, "Pick a material from the library first")
            return {"CANCELLED"}
        try:
            mat = nodes.build_texture_material(item, props.mapping, props.tex_scale)
        except RuntimeError as e:
            self.report({"ERROR"}, f"Could not load textures: {e}")
            return {"CANCELLED"}
        kept = []
        for obj in _targets(context):
            nodes.prepare_object(obj)
            kept += nodes.assign_material(obj, mat, props.all_slots)
        msg = f"Applied {mat.name}"
        if kept:
            msg += f" (previous kept: {', '.join(sorted(set(kept)))})"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class FM_OT_apply_procedural(bpy.types.Operator):
    bl_idname = "fm.apply_procedural"
    bl_label = "Apply Procedural"
    bl_description = "Create a texture-free procedural material and assign it"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(_targets(context))

    def execute(self, context):
        props = context.scene.fm
        mat = procedural.build(props.proc_preset)
        kept = []
        for obj in _targets(context):
            nodes.prepare_object(obj)
            kept += nodes.assign_material(obj, mat, props.all_slots)
        msg = f"Applied {mat.name}"
        if kept:
            msg += f" (previous kept: {', '.join(sorted(set(kept)))})"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


def _bake_cfg(context, objs, pack=False):
    props = context.scene.fm
    channels = [c for c in ("color", "rough", "metal", "normal_gl", "normal_dx", "ao",
                            "height", "orm") if getattr(props, "ch_" + c)]
    out_dir = props.bake_dir
    if out_dir.startswith("//") and not bpy.data.filepath:
        raise ValueError("Save the .blend first, or set an absolute output folder")
    cfg = {
        "objects": [o.name for o in objs],
        "res": int(props.bake_res),
        "samples": props.bake_samples,
        "out_dir": os.path.normpath(bpy.path.abspath(out_dir)),
        "channels": channels,
        "use_gpu": props.bake_gpu,
        "region_masks": props.bake_region_masks,
    }
    if pack:
        preset = bake.ENGINE_PRESETS[props.pack_engine]
        asset = (props.pack_asset.strip() or (context.active_object.name if context.active_object
                                               else objs[0].name))
        asset = "".join(c if c.isalnum() or c in "-_" else "_" for c in asset)
        cfg.update(channels=list(preset["channels"]), names=dict(preset["names"]), asset=asset,
                   engine=props.pack_engine, fbx=props.pack_fbx,
                   out_dir=os.path.join(cfg["out_dir"], asset),
                   part_suffix={o.name: ("" if len(objs) == 1 else "_" + o.name) for o in objs},
                   texel={o.name: round(bake.texel_density(o, cfg["res"]), 1) for o in objs})
    return cfg


class FM_OT_bake(bpy.types.Operator):
    bl_idname = "fm.bake"
    bl_label = "Bake Game Textures"
    bl_description = ("Bake the selected meshes to PBR texture maps (uses the active UV map; "
                      "mesh and UVs are not modified)")

    _timer = None
    _proc = None
    _tmp = None
    _cfg = None

    pack: BoolProperty(default=False, options={"SKIP_SAVE"},
                       description="Export a complete game-engine pack")

    @classmethod
    def poll(cls, context):
        return bool(_targets(context)) and not context.window_manager.get("fm_baking")

    def execute(self, context):
        objs = _targets(context)
        missing_uv = [o.name for o in objs if not o.data.uv_layers]
        if missing_uv:
            self.report({"ERROR"}, "No UV map on: " + ", ".join(missing_uv))
            return {"CANCELLED"}
        try:
            cfg = _bake_cfg(context, objs, self.pack)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if not cfg["channels"]:
            self.report({"ERROR"}, "Enable at least one map")
            return {"CANCELLED"}

        if not context.scene.fm.bake_background:
            rep = bake.bake_objects(cfg)
            return self._finish(rep)

        # background: save a copy of the current state, bake it in another process
        tmpdir = tempfile.mkdtemp(prefix="fm_bake_")
        blend = os.path.join(tmpdir, "bake_copy.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True, check_existing=False)
        cfg_path = os.path.join(tmpdir, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        runner = os.path.join(os.path.dirname(__file__), "bake_runner.py")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [bpy.app.binary_path, "-b", blend, "--python", runner, "--", cfg_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        self._tmp, self._cfg = tmpdir, cfg
        context.window_manager["fm_baking"] = 1
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, f"Baking {len(objs)} object(s) in the background...")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER" or self._proc.poll() is None:
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        context.window_manager.pop("fm_baking", None)
        rep_path = os.path.join(self._cfg["out_dir"], "fm_bake_report.json")
        rep = None
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
        except (OSError, ValueError):
            pass
        try:
            for fn in os.listdir(self._tmp):
                os.remove(os.path.join(self._tmp, fn))
            os.rmdir(self._tmp)
        except OSError:
            pass
        if rep is None:
            self.report({"ERROR"}, "Background bake failed; see fm_bake_log.txt in "
                        + self._cfg["out_dir"])
            return {"CANCELLED"}
        return self._finish(rep)

    def _finish(self, rep):
        n = sum(len(r["files"]) for r in rep["results"])
        for r in rep["results"]:
            for w in r["warnings"]:
                self.report({"WARNING"}, w)
        if rep["errors"]:
            self.report({"ERROR"}, "; ".join(rep["errors"]))
        msg = f"Baked {n} maps in {rep['seconds']}s -> {rep['out_dir']}"
        if rep.get("fbx"):
            msg += " + FBX"
        self.report({"INFO"}, msg)
        return {"FINISHED"} if n else {"CANCELLED"}


class FM_OT_save_look(bpy.types.Operator):
    bl_idname = "fm.save_look"
    bl_label = "Save Look"
    bl_description = "Save the active Forge material with all its settings as a reusable look"

    @classmethod
    def poll(cls, context):
        o = context.active_object
        m = o.active_material if o else None
        return bool(m and m.get("fm_source"))

    def execute(self, context):
        props = context.scene.fm
        _path, name = looks.save(context.active_object.active_material, props.look_name.strip() or None)
        props.look_name = ""
        self.report({"INFO"}, f"Saved look: {name}")
        return {"FINISHED"}


class FM_OT_apply_look(bpy.types.Operator):
    bl_idname = "fm.apply_look"
    bl_label = "Apply Look"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(_targets(context))

    def execute(self, context):
        sel = context.scene.fm.look_item
        lk = next((x for x in looks.list_looks() if x["file"] == sel), None)
        if lk is None:
            self.report({"ERROR"}, "Pick a saved look")
            return {"CANCELLED"}
        objs = _targets(context)
        mo = "BOX" if any(not o.data.uv_layers for o in objs) else None
        mat = looks.build(lk, mo)
        if mat is None:
            self.report({"ERROR"}, "The look's source material is not in the library")
            return {"CANCELLED"}
        for o in objs:
            nodes.prepare_object(o)
            nodes.assign_material(o, mat, context.scene.fm.all_slots)
        self.report({"INFO"}, f"Applied look {lk.get('name')}")
        return {"FINISHED"}


class FM_OT_delete_look(bpy.types.Operator):
    bl_idname = "fm.delete_look"
    bl_label = "Delete Look"
    bl_description = "Delete the selected saved look file"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        try:
            looks.delete(context.scene.fm.look_item)
        except OSError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class FM_OT_toggle_favourite(bpy.types.Operator):
    bl_idname = "fm.toggle_favourite"
    bl_label = "Favourite"
    bl_description = "Add or remove the selected library material from favourites"

    def execute(self, context):
        name = context.scene.fm.library_item
        if name and name != "NONE":
            on = looks.toggle_favourite(name)
            _ENUM["key"] = None
            self.report({"INFO"}, ("Added to" if on else "Removed from") + " favourites")
        return {"FINISHED"}


class FM_OT_open_bake_folder(bpy.types.Operator):
    bl_idname = "fm.open_bake_folder"
    bl_label = "Open Output Folder"

    def execute(self, context):
        path = bpy.path.abspath(context.scene.fm.bake_dir)
        if not os.path.isdir(path):
            self.report({"WARNING"}, "Nothing baked yet")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=path)
        return {"FINISHED"}


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Forge Materials"


class FM_PT_main(_Base, bpy.types.Panel):
    bl_label = "Forge Materials"
    bl_idname = "FM_PT_main"

    def draw(self, context):
        props = context.scene.fm
        self.layout.operator("fm.dial", text="Open Dial  (Ctrl+Alt+D)", icon="MESH_CIRCLE")
        col = self.layout.column(align=True)
        col.prop(props, "mapping", expand=True)
        row = col.row(align=True)
        row.prop(props, "tex_scale")
        row.prop(props, "all_slots", toggle=True)


class FM_PT_regions(_Base, bpy.types.Panel):
    bl_label = "Smart Regions"
    bl_parent_id = "FM_PT_main"

    def draw(self, context):
        lay = self.layout
        obj = context.active_object
        lay.operator("fm.regions", text="Smart Regions (click to refine)", icon="RESTRICT_SELECT_OFF")
        row = lay.row(align=True)
        op = row.operator("fm.regions_auto", text="Auto: By Shape", icon="MOD_LENGTH")
        op.method = "SHAPE"
        op = row.operator("fm.regions_auto", text="At Creases", icon="MOD_EDGESPLIT")
        op.method = "CREASES"
        op = lay.operator("fm.regions_auto", text="Auto + Blade Edge Strip", icon="SHARPCURVE")
        op.method = "SHAPE"
        op.edge = True
        lay.operator("fm.regions_ai", text="AI Hybrid Parts (background)", icon="SHADERFX")
        if not obj or obj.type != "MESH" or len(obj.material_slots) < 2:
            lay.label(text="Regions become material slots; geometry is not changed", icon="INFO")
            return
        names = list(obj.get("fm_region_names", []))
        col = lay.column(align=True)
        for i, slot in enumerate(obj.material_slots):
            nm = names[i] if i < len(names) and names[i] else (slot.material.name if slot.material else "empty")
            sub = col.row(align=True)
            op = sub.operator("fm.region_activate", text=f"{nm}", depress=(i == obj.active_material_index),
                              icon="MATERIAL")
            op.index = i
        lay.operator("fm.regions_suggest", icon="SHADING_TEXTURE")
        lay.label(text="The dial edits the active region (Tab in the dial)", icon="INFO")


class FM_PT_library(_Base, bpy.types.Panel):
    bl_label = "Texture Library"
    bl_parent_id = "FM_PT_main"

    def draw(self, context):
        lay = self.layout
        props = context.scene.fm
        n = len(library.items())
        row = lay.row(align=True)
        row.label(text=f"{n} materials" if n else "Library not scanned")
        row.operator("fm.scan_library", text="Rescan" if n else "Scan", icon="FILE_REFRESH")
        if not n:
            return
        lay.prop(props, "category", text="")
        lay.prop(props, "search", text="", icon="VIEWZOOM")
        lay.template_icon_view(props, "library_item", show_labels=True, scale=6.0,
                               scale_popup=5.0)
        it = library.get(props.library_item)
        if it:
            lay.label(text=f"{it['label']}  ·  maps: {', '.join(sorted(it['maps']))}")
        row = lay.row(align=True)
        row.operator("fm.apply_library", icon="MATERIAL")
        fav = it is not None and it["name"] in looks.favourites()
        row.operator("fm.toggle_favourite", text="", icon="SOLO_ON" if fav else "SOLO_OFF")


class FM_PT_procedural(_Base, bpy.types.Panel):
    bl_label = "Procedural Materials"
    bl_parent_id = "FM_PT_main"

    def draw(self, context):
        props = context.scene.fm
        row = self.layout.row(align=True)
        row.prop(props, "proc_preset", text="")
        row.operator("fm.apply_procedural", text="Apply", icon="MATERIAL")


class FM_PT_looks(_Base, bpy.types.Panel):
    bl_label = "Saved Looks"
    bl_parent_id = "FM_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        lay = self.layout
        props = context.scene.fm
        row = lay.row(align=True)
        row.prop(props, "look_name", text="")
        row.operator("fm.save_look", text="Save", icon="FILE_TICK")
        row = lay.row(align=True)
        row.prop(props, "look_item", text="")
        row.operator("fm.apply_look", text="", icon="CHECKMARK")
        row.operator("fm.delete_look", text="", icon="TRASH")


SURFACE_SECTIONS = [
    ("Adjust", "COLOR", ["Tint", "Brightness", "Roughness Scale", "Roughness Offset",
                         "Metallic Strength"]),
    ("Edge Wear", "MOD_BEVEL", ["Edge Wear", "Edge Width", "Edge Color", "Edge Roughness",
                                "Edge Metallic", "Curvature Boost"]),
    ("Dirt", "BRUSH_DATA", ["Dirt", "Dirt Spread", "Dirt Color", "Dirt Roughness"]),
    ("Rust", "MOD_NOISE", ["Rust", "Rust Scale", "Rust Color"]),
    ("Weather", "MOD_OCEAN", ["Snow", "Snow Color", "Moss", "Moss Scale", "Moss Color",
                              "Wetness", "Puddles"]),
    ("Paint", "BRUSH_DATA", ["Paint", "Paint Wear", "Paint Color"]),
    ("Variation", "TEXTURE", ["Breakup Scale"]),
]


class FM_PT_surface(_Base, bpy.types.Panel):
    bl_label = "Surface & Weathering"
    bl_parent_id = "FM_PT_main"

    def draw(self, context):
        lay = self.layout
        obj = context.active_object
        mat = obj.active_material if obj else None
        surf = nodes.find_surface(mat)
        if not surf:
            lay.label(text="Active material is not a Forge Materials material", icon="INFO")
            return
        lay.label(text=mat.name, icon="MATERIAL")
        nt = mat.node_tree
        sz = nt.nodes.get("FM_Size")
        if sz:
            lay.prop(sz.outputs[0], "default_value", text="Pattern Size \u00d7")
        mp = nt.nodes.get("FM_Mapping")
        if mp:
            lay.prop(mp.inputs["Rotation"], "default_value", index=2, text="Rotation")
        bump = nt.nodes.get("FM_Bump")
        if bump:
            lay.prop(bump.inputs["Strength"], "default_value", text="Detail Bump")
        nm = nt.nodes.get("FM_NormalMap")
        if nm:
            lay.prop(nm.inputs["Strength"], "default_value", text="Normal Strength")
        proc = nt.nodes.get("FM_Proc")
        if proc:
            box = lay.box()
            box.label(text=proc.label or "Pattern", icon="SHADING_RENDERED")
            for s in proc.inputs:
                if s.name in ("Vector", "Normal") or s.is_linked:
                    continue
                box.prop(s, "default_value", text=s.name)
        for title, icon, names in SURFACE_SECTIONS:
            box = lay.box()
            box.label(text=title, icon=icon)
            col = box.column(align=True)
            for name in names:
                s = surf.inputs.get(name)
                if s is not None and not s.is_linked:
                    col.prop(s, "default_value", text=name)
        if context.scene.render.engine != "CYCLES":
            lay.label(text="Edge wear preview needs Cycles (bake uses Cycles)", icon="INFO")


_TEXEL = {}


def _texel_cached(obj, res):
    me = obj.data
    key = (me.as_pointer(), len(me.polygons), len(me.loops), res, tuple(round(v, 4) for v in obj.matrix_world.to_scale()))
    if obj.name not in _TEXEL or _TEXEL[obj.name][0] != key:
        _TEXEL[obj.name] = (key, bake.texel_density(obj, res))
    return _TEXEL[obj.name][1]


class FM_PT_bake(_Base, bpy.types.Panel):
    bl_label = "Bake for Game Engines"
    bl_parent_id = "FM_PT_main"

    def draw(self, context):
        lay = self.layout
        props = context.scene.fm
        col = lay.column(align=True)
        col.prop(props, "bake_res")
        col.prop(props, "bake_samples")
        lay.prop(props, "bake_dir")
        grid = lay.grid_flow(columns=2, align=True)
        for c in ("color", "rough", "metal", "orm", "normal_gl", "normal_dx", "ao", "height"):
            grid.prop(props, "ch_" + c, toggle=True)
        row = lay.row(align=True)
        row.prop(props, "bake_background", toggle=True)
        row.prop(props, "bake_gpu", toggle=True)
        lay.prop(props, "bake_region_masks")
        objs = _targets(context)
        no_uv = [o.name for o in objs if not o.data.uv_layers]
        if no_uv:
            lay.label(text="No UV map: " + ", ".join(no_uv), icon="ERROR")
        if context.window_manager.get("fm_baking"):
            lay.label(text="Baking in background...", icon="SORTTIME")
        lay.operator("fm.bake", icon="RENDER_STILL")
        box = lay.box()
        box.label(text="Export Pack", icon="EXPORT")
        box.prop(props, "pack_engine", text="")
        box.prop(props, "pack_asset")
        box.prop(props, "pack_fbx")
        if objs:
            res = int(props.bake_res)
            dens = [(o.name, _texel_cached(o, res)) for o in objs[:6] if o.data.uv_layers]
            if dens:
                vals = [d for _, d in dens if d > 0]
                if vals:
                    spread = max(vals) / max(min(vals), 1e-6)
                    box.label(text=f"Texel density: {min(vals):.0f}\u2013{max(vals):.0f} px/m",
                              icon="ERROR" if spread > 1.25 else "CHECKMARK")
                    if spread > 1.25:
                        box.label(text="Parts differ by more than 25%: rescale UVs to match")
        op = box.operator("fm.bake", text="Export Game Pack", icon="PACKAGE")
        op.pack = True
        lay.operator("fm.open_bake_folder", icon="FILE_FOLDER")


classes = (
    FM_Props, FM_OT_scan_library, FM_OT_apply_library, FM_OT_apply_procedural,
    FM_OT_bake, FM_OT_open_bake_folder, FM_OT_save_look, FM_OT_apply_look, FM_OT_delete_look,
    FM_OT_toggle_favourite,
    FM_PT_main, FM_PT_regions, FM_PT_library, FM_PT_procedural, FM_PT_looks, FM_PT_surface, FM_PT_bake,
)


def register():
    import bpy.utils.previews
    _PREVIEWS["pcoll"] = bpy.utils.previews.new()
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.fm = bpy.props.PointerProperty(type=FM_Props)
    try:
        p = prefs()
        library.load_cache(cache_path(), bpy.path.abspath(p.library_path))
    except (KeyError, AttributeError):
        pass


def unregister():
    del bpy.types.Scene.fm
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    if _PREVIEWS["pcoll"] is not None:
        bpy.utils.previews.remove(_PREVIEWS["pcoll"])
        _PREVIEWS["pcoll"] = None
