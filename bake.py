# SPDX-License-Identifier: GPL-3.0-or-later
"""Game-ready texture baking (Cycles).

Self-contained (imports only bpy/numpy/os/json) so the background runner can
load it from a separate Blender process without registering the add-on.

Bakes per object: Base Color, Roughness, Metallic, Normal (OpenGL + DirectX),
AO, Height and a packed ORM (R=AO, G=Roughness, B=Metallic). All materials on
the object bake into the same image set, using the object's active UV map.
The mesh is never modified; UVs are never touched.
"""

import json
import os
import time

import bpy

try:
    import numpy as np
except ImportError:  # numpy ships with Blender; guard anyway
    np = None

CHANNELS = {
    # key: (file suffix, float buffer, non-color, bit depth)
    "color": ("BaseColor", False, False, "8"),
    "rough": ("Roughness", False, True, "8"),
    "metal": ("Metallic", False, True, "8"),
    "ao": ("AO", False, True, "8"),
    "height": ("Height", True, True, "16"),
    "normal_gl": ("Normal_OpenGL", True, True, "16"),
    "normal_dx": ("Normal_DirectX", True, True, "16"),
    "orm": ("ORM", False, True, "8"),
    "msmooth": ("MetallicSmoothness", False, True, "8"),   # Unity: R=metal, A=1-rough
}
RGBA_CHANNELS = {"msmooth"}

# file naming per engine: channel -> pattern ({asset}, {part} = object suffix or "")
ENGINE_PRESETS = {
    "UNREAL": {"channels": ["color", "normal_dx", "orm"],
               "names": {"color": "T_{asset}{part}_BC", "normal_dx": "T_{asset}{part}_N",
                         "orm": "T_{asset}{part}_ORM"}},
    "UNITY": {"channels": ["color", "normal_gl", "msmooth", "ao"],
              "names": {"color": "{asset}{part}_Albedo", "normal_gl": "{asset}{part}_Normal",
                        "msmooth": "{asset}{part}_MetallicSmoothness", "ao": "{asset}{part}_AO"}},
    "GODOT": {"channels": ["color", "normal_gl", "orm"],
              "names": {"color": "{asset}{part}_albedo", "normal_gl": "{asset}{part}_normal",
                        "orm": "{asset}{part}_orm"}},
    "GENERIC": {"channels": ["color", "rough", "metal", "normal_gl", "ao", "height"],
                "names": {"color": "{asset}{part}_BaseColor", "rough": "{asset}{part}_Roughness",
                          "metal": "{asset}{part}_Metallic", "normal_gl": "{asset}{part}_Normal",
                          "ao": "{asset}{part}_AO", "height": "{asset}{part}_Height"}},
}
EMIT_SOCKET = {"color": "Base Color", "rough": "Roughness", "metal": "Metallic"}


class BakeError(Exception):
    pass


# --------------------------------------------------------------------------
# scene state
# --------------------------------------------------------------------------

def _snapshot(scene):
    r = scene.render
    vs = scene.view_settings
    snap = {
        "engine": r.engine,
        "samples": scene.cycles.samples,
        "device": scene.cycles.device,
        "margin": r.bake.margin,
        "use_clear": r.bake.use_clear,
        "use_selected_to_active": r.bake.use_selected_to_active,
        "view_transform": vs.view_transform,
        "look": vs.look,
        "exposure": vs.exposure,
        "gamma": vs.gamma,
        "fmt": (r.image_settings.file_format, r.image_settings.color_mode,
                r.image_settings.color_depth),
        "selected": [o.name for o in bpy.context.view_layer.objects if o.select_get()],
        "active": bpy.context.view_layer.objects.active.name
        if bpy.context.view_layer.objects.active else None,
    }
    try:
        snap["margin_type"] = r.bake.margin_type
    except AttributeError:
        pass
    return snap


def _restore(scene, snap):
    r = scene.render
    r.engine = snap["engine"]
    scene.cycles.samples = snap["samples"]
    scene.cycles.device = snap["device"]
    r.bake.margin = snap["margin"]
    r.bake.use_clear = snap["use_clear"]
    r.bake.use_selected_to_active = snap["use_selected_to_active"]
    if "margin_type" in snap:
        r.bake.margin_type = snap["margin_type"]
    vs = scene.view_settings
    vs.view_transform = snap["view_transform"]
    vs.look = snap["look"]
    vs.exposure = snap["exposure"]
    vs.gamma = snap["gamma"]
    fmt = r.image_settings
    fmt.file_format, fmt.color_mode = snap["fmt"][0], snap["fmt"][1]
    try:
        fmt.color_depth = snap["fmt"][2]
    except TypeError:
        pass
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(o.name in snap["selected"])
    if snap["active"] and snap["active"] in vl.objects:
        vl.objects.active = vl.objects[snap["active"]]


def _setup_scene(scene, cfg):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(cfg.get("samples", 32))
    if cfg.get("use_gpu", True):
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.get_devices()
            if prefs.compute_device_type != "NONE":
                scene.cycles.device = "GPU"
        except (KeyError, AttributeError, TypeError):
            pass
    b = scene.render.bake
    b.margin = int(cfg.get("margin", max(4, int(cfg["res"]) // 128)))
    try:
        b.margin_type = "EXTEND"
    except (AttributeError, TypeError):
        pass
    b.use_clear = True
    b.use_selected_to_active = False
    try:
        b.target = "IMAGE_TEXTURES"
    except (AttributeError, TypeError):
        pass
    # plain output transform so saved maps contain the baked values
    vs = scene.view_settings
    vs.view_transform = "Standard"
    try:
        vs.look = "None"
    except TypeError:
        pass
    vs.exposure = 0.0
    vs.gamma = 1.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _materials(obj):
    seen = []
    for slot in obj.material_slots:
        m = slot.material
        if m and m.node_tree and m not in seen:
            seen.append(m)
    return seen


def _principled(nt):
    out = nt.get_output_node("CYCLES")
    if out and out.inputs["Surface"].is_linked:
        n = out.inputs["Surface"].links[0].from_node
        if n.type == "BSDF_PRINCIPLED":
            return out, n
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return out, n
    return out, None


def _new_image(name, res, is_float, non_color):
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, res, res, alpha=False, float_buffer=is_float)
    if non_color:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            img.colorspace_settings.is_data = True
    return img


def _set_targets(mats, img):
    nodes = []
    for m in mats:
        nt = m.node_tree
        n = nt.nodes.new("ShaderNodeTexImage")
        n.name = "FM_BAKE_TARGET"
        n.image = img
        n.location = (1200, 400)
        for other in nt.nodes:
            other.select = False
        n.select = True
        nt.nodes.active = n
        nodes.append((nt, n))
    return nodes


def _clear_targets(nodes):
    for nt, n in nodes:
        nt.nodes.remove(n)


def _patch_emit(mats, channel, warnings):
    """Route one channel into an Emission shader; return undo records."""
    undo = []
    for m in mats:
        nt = m.node_tree
        out, bsdf = _principled(nt)
        if out is None:
            warnings.append(f"{m.name}: no material output")
            continue
        surf = out.inputs["Surface"]
        orig = surf.links[0].from_socket if surf.is_linked else None
        em = nt.nodes.new("ShaderNodeEmission")
        em.name = "FM_BAKE_EMIT"
        em.inputs["Strength"].default_value = 1.0
        em.location = (out.location.x, out.location.y + 300)
        src = None
        default = 0.0
        if channel == "height":
            hn = nt.nodes.get("FM_HEIGHT")
            if hn:
                src = hn.outputs[0]
            else:
                default = 0.5
        elif bsdf is None:
            warnings.append(f"{m.name}: no Principled BSDF, '{channel}' baked as default")
            default = 0.5 if channel == "rough" else 0.0
        else:
            sock = bsdf.inputs[EMIT_SOCKET[channel]]
            if sock.is_linked:
                src = sock.links[0].from_socket
            else:
                default = sock.default_value
        if src is not None:
            nt.links.new(src, em.inputs["Color"])
        else:
            if isinstance(default, float):
                em.inputs["Color"].default_value = (default, default, default, 1.0)
            else:
                em.inputs["Color"].default_value = tuple(default)
        nt.links.new(em.outputs["Emission"], surf)
        undo.append((nt, out, orig, em))
    return undo


def _unpatch_emit(undo):
    for nt, out, orig, em in undo:
        nt.nodes.remove(em)
        if orig is not None:
            nt.links.new(orig, out.inputs["Surface"])


def _pixels(img):
    w, h = img.size
    arr = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    return arr


def _save(scene, img, path, depth, rgba=False):
    fmt = scene.render.image_settings
    fmt.file_format = "PNG"
    fmt.color_mode = "RGBA" if rgba else "RGB"
    try:
        fmt.color_depth = depth
    except TypeError:
        fmt.color_depth = "8"
    fmt.compression = 15
    img.save_render(path, scene=scene)


def _mesh_fingerprint(obj):
    me = obj.data
    n = len(me.vertices)
    if np is None:
        return (n, len(me.polygons))
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    return (n, len(me.polygons), len(me.loops), float(np.round(co.sum(), 6)))


# --------------------------------------------------------------------------
# main entry points
# --------------------------------------------------------------------------

def bake_object(obj, cfg, log=print):
    """Bake one mesh object. Returns dict with written files and warnings."""
    if np is None:
        raise BakeError("numpy is not available")
    scene = bpy.context.scene
    if obj.type != "MESH":
        raise BakeError(f"{obj.name}: not a mesh")
    if not obj.data.uv_layers or not obj.data.uv_layers.active:
        raise BakeError(f"{obj.name}: has no UV map. Unwrap it first (the add-on never "
                        "changes UVs on its own).")
    mats = _materials(obj)
    if not mats:
        raise BakeError(f"{obj.name}: has no node materials")
    if obj.hide_get() or obj.hide_viewport or obj.hide_render:
        raise BakeError(f"{obj.name}: is hidden; unhide it to bake")

    res = int(cfg["res"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    wanted = list(dict.fromkeys(cfg["channels"]))
    need = set(wanted)
    if "orm" in need:
        need |= {"ao", "rough", "metal"}
    if "msmooth" in need:
        need |= {"rough", "metal"}
    if "normal_dx" in need:
        need.add("normal_gl")

    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(False)
    obj.select_set(True)
    vl.objects.active = obj

    base = cfg.get("prefix") or obj.name
    base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    fp_before = _mesh_fingerprint(obj)
    warnings, files, images = [], {}, {}

    order = [c for c in ("color", "rough", "metal", "height", "ao", "normal_gl") if c in need]
    masked = bool(cfg.get("region_masks", True)) and _region_count(obj) > 1
    if masked:
        try:
            images = _bake_masked(obj, cfg, order, base, res, warnings, log)
            order = []
        except Exception as e:  # never lose the bake: fall back to per-face regions
            warnings.append(f"{obj.name}: smooth region borders unavailable ({e}); baked per face")
            for img in images.values():
                bpy.data.images.remove(img)
            images = {}
            for o in vl.objects:
                o.select_set(False)
            obj.select_set(True)
            vl.objects.active = obj
    for ch in order:
        suffix, is_float, non_color, _ = CHANNELS[ch]
        img = _new_image(f"FMB_{base}_{suffix}", res, is_float, non_color)
        images[ch] = img
        targets = _set_targets(mats, img)
        t0 = time.time()
        try:
            if ch in ("color", "rough", "metal", "height"):
                undo = _patch_emit(mats, ch, warnings)
                try:
                    bpy.ops.object.bake(type="EMIT")
                finally:
                    _unpatch_emit(undo)
            elif ch == "ao":
                bpy.ops.object.bake(type="AO")
            elif ch == "normal_gl":
                bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT",
                                    normal_r="POS_X", normal_g="POS_Y", normal_b="POS_Z")
        finally:
            _clear_targets(targets)
        log(f"  {obj.name}: {suffix} baked in {time.time() - t0:.1f}s")

    if "normal_dx" in need:
        src = images["normal_gl"]
        arr = _pixels(src)
        arr[1::4] = 1.0 - arr[1::4]
        dx = _new_image(f"FMB_{base}_Normal_DirectX", res, True, True)
        dx.pixels.foreach_set(arr)
        images["normal_dx"] = dx

    if "orm" in need:
        ao, ro, me = _pixels(images["ao"]), _pixels(images["rough"]), _pixels(images["metal"])
        orm = np.ones_like(ao)
        orm[0::4], orm[1::4], orm[2::4] = ao[0::4], ro[0::4], me[0::4]
        img = _new_image(f"FMB_{base}_ORM", res, False, True)
        img.pixels.foreach_set(orm)
        images["orm"] = img

    if "msmooth" in need:
        ro, me = _pixels(images["rough"]), _pixels(images["metal"])
        ms = np.zeros_like(ro)
        ms[0::4] = me[0::4]
        ms[3::4] = 1.0 - ro[0::4]
        img = _new_image(f"FMB_{base}_MetallicSmoothness", res, False, True)
        img.alpha_mode = "STRAIGHT"
        img.pixels.foreach_set(ms)
        images["msmooth"] = img

    names = cfg.get("names") or {}
    for ch in wanted:
        suffix, _, _, depth = CHANNELS[ch]
        if ch in names:
            stem = names[ch].format(asset=cfg.get("asset") or base, part=cfg.get("part_suffix", {}).get(obj.name, ""))
            stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
        else:
            stem = f"{base}_{suffix}"
        path = os.path.join(out_dir, stem + ".png")
        _save(scene, images[ch], path, depth, rgba=ch in RGBA_CHANNELS)
        files[ch] = path

    if cfg.get("cleanup_images", True):
        for img in images.values():
            bpy.data.images.remove(img)

    if _mesh_fingerprint(obj) != fp_before:  # should never happen
        warnings.append(f"{obj.name}: mesh fingerprint changed during bake!")
    return {"object": obj.name, "files": files, "warnings": warnings}


# --------------------------------------------------------------------------
# smooth region borders (texture-space masks)
# --------------------------------------------------------------------------

def _sibling(name):
    """Load segment.py / masks.py next to this file (works in the add-on and
    in the background runner, which loads bake.py on its own)."""
    import importlib.util
    import sys
    key = "fm_bake_" + name
    if key not in sys.modules:
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(key, os.path.join(here, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def _region_count(obj):
    me = obj.data
    if len(obj.material_slots) < 2:
        return 1
    mi = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", mi)
    return len(np.unique(mi))


def _info_material(kind):
    """Helper material whose emission is a per-texel fact: the face's cell + 1,
    the object-space position (as bounding-box coordinates) or the normal."""
    m = bpy.data.materials.new(f"FM_BAKE_INFO_{kind}")
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    if kind == "cell":
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "GEOMETRY"
        at.attribute_name = "fm_cell"
        add = nt.nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = 1.0
        nt.links.new(at.outputs["Fac"], add.inputs[0])
        nt.links.new(add.outputs[0], em.inputs["Color"])
    else:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        if kind == "pos":
            nt.links.new(tc.outputs["Generated"], em.inputs["Color"])
        else:
            vm = nt.nodes.new("ShaderNodeVectorMath")
            vm.operation = "MULTIPLY_ADD"
            vm.inputs[1].default_value = (0.5, 0.5, 0.5)
            vm.inputs[2].default_value = (0.5, 0.5, 0.5)
            nt.links.new(tc.outputs["Normal"], vm.inputs[0])
            nt.links.new(vm.outputs[0], em.inputs["Color"])
    return m


def _bake_into(tmp, mats, img):
    targets = _set_targets(mats, img)
    try:
        bpy.ops.object.bake(type="EMIT")
    finally:
        _clear_targets(targets)


def _bake_masked(obj, cfg, order, base, res, warnings, log):
    """Bake every region's material over the whole object (on a temporary copy)
    and blend the results with per-texel region weights, so borders run
    through faces at texel precision, anti-aliased."""
    seg = _sibling("segment")
    M = _sibling("masks")
    scene = bpy.context.scene
    vl = bpy.context.view_layer
    me = obj.data
    nslots = len(obj.material_slots)
    nf = len(me.polygons)
    mi = np.empty(nf, np.int32)
    me.polygons.foreach_get("material_index", mi)
    t0 = time.time()
    C = seg.Cells(me, target_cells=int(np.clip(nf / 25, 4000, 32000)))
    labels, used = M.slot_labels(C, mi, nslots)
    names = (list(obj.get("fm_region_names", [])) + [""] * nslots)[:nslots]
    rules = M.parse_rules(obj.get("fm_region_rules", "{}"), names)
    regions = [int(r) for r in used if obj.material_slots[int(r)].material is not None]
    log(f"  {obj.name}: {len(regions)} regions, analysis {time.time() - t0:.1f}s")

    tmp_me = me.copy()
    tmp = bpy.data.objects.new(obj.name + "__fm_bake", tmp_me)
    tmp.matrix_world = obj.matrix_world.copy()
    scene.collection.objects.link(tmp)
    attr = tmp_me.attributes.new("fm_cell", "FLOAT", "FACE")
    attr.data.foreach_set("value", C.face_cell.astype(np.float32))
    zero = np.zeros(nf, np.int32)
    tmp_me.polygons.foreach_set("material_index", zero)
    helpers, temp_imgs = [], []
    was_hidden = obj.hide_render
    samples = scene.cycles.samples
    acc = {}
    try:
        obj.hide_render = True                  # the copy stands in for it
        for o in vl.objects:
            o.select_set(False)
        tmp.select_set(True)
        vl.objects.active = tmp

        def bake_info(kind):
            m = _info_material(kind)
            helpers.append(m)
            tmp_me.materials.clear()
            tmp_me.materials.append(m)
            img = _new_image(f"FMB_info_{kind}", res, True, True)
            temp_imgs.append(img)
            _bake_into(tmp, [m], img)
            return _pixels(img).reshape(-1, 4)[:, :3].copy()

        scene.cycles.samples = 1                # one sample: exact values per texel
        cellv = bake_info("cell")[:, 0]
        gen = bake_info("pos")
        nrm = bake_info("nor") * 2.0 - 1.0
        scene.cycles.samples = samples
        cov = cellv > 0.5
        bb = np.array([v[:] for v in obj.bound_box])
        lo_b, hi_b = bb.min(0), bb.max(0)
        pos = lo_b + gen[cov] * (hi_b - lo_b)
        nor = nrm[cov]
        nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
        tcell = M.texel_cells(C, cellv[cov] - 1.0, pos)
        # world size of one texel, in the mesh's own units
        uv_area, area3d = _uv_and_area(me)
        texel = (1.0 / (res * np.sqrt(uv_area / max(area3d, 1e-12)))) if uv_area > 0 else C.cell
        t1 = time.time()
        W = M.region_weights(C, labels, regions, rules, pos, nor, tcell, texel)
        W = M.despeckle(W, cov, res)
        log(f"  {obj.name}: region masks {time.time() - t1:.1f}s ({cov.sum()} texels)")

        idx = np.nonzero(cov)[0]
        npx = res * res
        bake_ch = [c for c in order if c in ("color", "rough", "metal", "height", "normal_gl")]
        for ch in bake_ch:
            acc[ch] = np.zeros((npx, 3), np.float32)
        for k, r in enumerate(regions):
            wr = W[:, k]
            if not (wr > 1e-4).any():
                continue
            mat = obj.material_slots[r].material
            tmp_me.materials.clear()
            tmp_me.materials.append(mat)
            for ch in bake_ch:
                suffix, is_float, non_color, _ = CHANNELS[ch]
                img = _new_image(f"FMB_r{r}_{suffix}", res, True, non_color)
                tb = time.time()
                if ch == "normal_gl":
                    targets = _set_targets([mat], img)
                    try:
                        bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT",
                                            normal_r="POS_X", normal_g="POS_Y", normal_b="POS_Z")
                    finally:
                        _clear_targets(targets)
                else:
                    undo = _patch_emit([mat], ch, warnings)
                    try:
                        _bake_into(tmp, [mat], img)
                    finally:
                        _unpatch_emit(undo)
                px = _pixels(img).reshape(-1, 4)[:, :3]
                if ch == "normal_gl":
                    px = px * 2.0 - 1.0
                acc[ch][idx] += wr[:, None] * px[idx]
                bpy.data.images.remove(img)
                log(f"  {obj.name}: region {names[r] or r} {suffix} {time.time() - tb:.1f}s")
        images = {}
        for ch in bake_ch:
            suffix, is_float, non_color, _ = CHANNELS[ch]
            a = acc[ch]
            if ch == "normal_gl":
                n = a[idx]
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
                a = np.zeros_like(a)
                a[:, 2] = 1.0
                a[idx] = n
                a = a * 0.5 + 0.5
            img = _new_image(f"FMB_{base}_{suffix}", res, is_float, non_color)
            full = np.ones((npx, 4), np.float32)
            full[:, :3] = a
            img.pixels.foreach_set(full.ravel())
            images[ch] = img
        if "ao" in order:
            img = _new_image(f"FMB_{base}_AO", res, False, True)
            mat = obj.material_slots[regions[0]].material
            tmp_me.materials.clear()
            tmp_me.materials.append(mat)
            targets = _set_targets([mat], img)
            try:
                bpy.ops.object.bake(type="AO")
            finally:
                _clear_targets(targets)
            images["ao"] = img
        return images
    finally:
        scene.cycles.samples = samples
        obj.hide_render = was_hidden
        bpy.data.objects.remove(tmp)
        bpy.data.meshes.remove(tmp_me)
        for m in helpers:
            bpy.data.materials.remove(m)
        for im in temp_imgs:
            bpy.data.images.remove(im)
        for o in vl.objects:
            o.select_set(False)
        obj.select_set(True)
        vl.objects.active = obj


def _uv_and_area(me):
    """UV area and 3D area of a mesh (its own units)."""
    n = len(me.polygons)
    uv = me.uv_layers.active
    areas = np.empty(n, np.float64)
    me.polygons.foreach_get("area", areas)
    nl = len(me.loops)
    co = np.empty(nl * 2, np.float64)
    uv.data.foreach_get("uv", co)
    co = co.reshape(-1, 2)
    starts = np.empty(n, np.int64)
    counts = np.empty(n, np.int64)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", counts)
    order = np.argsort(starts)
    starts, counts = starts[order], counts[order]
    nxt = np.arange(nl) + 1
    nxt[starts + counts - 1] = starts
    x, y = co[:, 0], co[:, 1]
    uv_area = float(np.abs(np.add.reduceat(x * y[nxt] - y * x[nxt], starts)).sum() * 0.5)
    return uv_area, float(areas.sum())


def bake_objects(cfg, log=print):
    scene = bpy.context.scene
    snap = _snapshot(scene)
    results, errors = [], []
    t0 = time.time()
    try:
        _setup_scene(scene, cfg)
        for name in cfg["objects"]:
            obj = bpy.data.objects.get(name)
            if obj is None:
                errors.append(f"{name}: object not found")
                continue
            try:
                results.append(bake_object(obj, cfg, log))
            except BakeError as e:
                errors.append(str(e))
            except RuntimeError as e:
                errors.append(f"{name}: {e}")
    finally:
        _restore(scene, snap)
    fbx = None
    if cfg.get("fbx") and results:
        try:
            fbx = export_fbx(cfg, [r["object"] for r in results], results)
        except (RuntimeError, OSError) as e:
            errors.append(f"FBX export failed: {e}")
        except (KeyError, StopIteration) as e:
            errors.append(f"FBX export failed: {e}")
    report = {"results": results, "errors": errors, "seconds": round(time.time() - t0, 1),
              "out_dir": cfg["out_dir"], "fbx": fbx, "texel": cfg.get("texel")}
    try:
        with open(os.path.join(cfg["out_dir"], "fm_bake_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except OSError:
        pass
    return report


def _export_material(name, files):
    """One plain material that points at the baked maps, so the engine imports a
    single material with its textures already wired (whatever regions the
    object had in Blender)."""
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except AttributeError:
        pass
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    images = []

    def tex(path, non_color, x, y):
        img = bpy.data.images.load(path, check_existing=False)
        if non_color:
            try:
                img.colorspace_settings.name = "Non-Color"
            except TypeError:
                img.colorspace_settings.is_data = True
        images.append(img)
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        n.location = (x, y)
        return n

    if "color" in files:
        nt.links.new(tex(files["color"], False, -600, 300).outputs["Color"], bsdf.inputs["Base Color"])
    nrm = files.get("normal_gl") or files.get("normal_dx")
    if nrm:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-250, -300)
        nt.links.new(tex(nrm, True, -600, -300).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if "rough" in files:
        nt.links.new(tex(files["rough"], True, -600, 0).outputs["Color"], bsdf.inputs["Roughness"])
    if "metal" in files:
        nt.links.new(tex(files["metal"], True, -600, -150).outputs["Color"], bsdf.inputs["Metallic"])
    return m, images


def export_fbx(cfg, object_names, results=None):
    """Export the baked objects as one FBX next to the textures.

    Exported from temporary copies that carry a single material (the baked
    maps), so several material regions in Blender still give one material and
    one draw call in the engine. The scene, the objects and their materials are
    left exactly as they were."""
    vl = bpy.context.view_layer
    scn = bpy.context.scene
    prev_sel = [o.name for o in vl.objects if o.select_get()]
    prev_act = vl.objects.active.name if vl.objects.active else None
    files_of = {r["object"]: r["files"] for r in (results or [])}
    name = cfg.get("asset") or object_names[0]
    prefix = "SM_" if cfg.get("engine") == "UNREAL" else ""
    path = os.path.join(cfg["out_dir"], f"{prefix}{name}.fbx")
    temps, mats, imgs, renamed = [], [], [], []
    try:
        for o in vl.objects:
            o.select_set(False)
        for on in object_names:
            src = bpy.data.objects[on]
            me = src.data.copy()
            mi = np.zeros(len(me.polygons), np.int32)
            me.polygons.foreach_set("material_index", mi)
            me.materials.clear()
            mat_name = ("M_" if prefix else "") + name + cfg.get("part_suffix", {}).get(on, "")
            m, im = _export_material(mat_name, files_of.get(on, {}))
            mats.append(m)
            imgs += im
            me.materials.append(m)
            # the copy takes the original's name in the FBX; the original is
            # renamed for the duration of the export and renamed back after
            orig = src.name
            src.name = orig + "__fm_export_src"
            renamed.append((src, orig))
            t = bpy.data.objects.new(orig, me)
            t.matrix_world = src.matrix_world.copy()
            scn.collection.objects.link(t)
            t.select_set(True)
            temps.append(t)
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True, apply_unit_scale=True,
                                 apply_scale_options="FBX_SCALE_ALL", mesh_smooth_type="FACE",
                                 use_tspace=True, add_leaf_bones=False, path_mode="RELATIVE",
                                 bake_anim=False, object_types={"MESH"})
    finally:
        for t in temps:
            me = t.data
            bpy.data.objects.remove(t)
            bpy.data.meshes.remove(me)
        for src, orig in renamed:
            src.name = orig
        for m in mats:
            bpy.data.materials.remove(m)
        for im in imgs:
            bpy.data.images.remove(im)
        for o in vl.objects:
            o.select_set(o.name in prev_sel)
        if prev_act and prev_act in vl.objects:
            vl.objects.active = vl.objects[prev_act]
    return path


def texel_density(obj, res):
    """Pixels per metre for this object at the given texture resolution (read-only)."""
    me = obj.data
    uv = me.uv_layers.active
    if uv is None or not me.polygons or np is None:
        return 0.0
    n = len(me.polygons)
    areas = np.empty(n, np.float64)
    me.polygons.foreach_get("area", areas)
    s = obj.matrix_world.to_scale()
    area3d = float(areas.sum()) * abs(s.x * s.y * s.z) ** (2.0 / 3.0)
    nl = len(me.loops)
    co = np.empty(nl * 2, np.float64)
    uv.data.foreach_get("uv", co)
    co = co.reshape(-1, 2)
    starts = np.empty(n, np.int64)
    counts = np.empty(n, np.int64)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", counts)
    order = np.argsort(starts)
    starts, counts = starts[order], counts[order]
    nxt = np.arange(nl) + 1
    nxt[starts + counts - 1] = starts
    x, y = co[:, 0], co[:, 1]
    uv_area = float(np.abs(np.add.reduceat(x * y[nxt] - y * x[nxt], starts)).sum() * 0.5)
    if area3d <= 0 or uv_area <= 0:
        return 0.0
    return float(res * (uv_area / area3d) ** 0.5)
