# SPDX-License-Identifier: GPL-3.0-or-later
# Runs inside Blender (background, on a copy):
#   blender -b <file.blend> --python render_views.py -- <out_dir> <object> [res] [nviews] [parts.npy]
# Renders the object from many directions: a face-ID image (EXR, exact) plus
# images for the segmenter (normal matcap and/or shaded with cavity). Never saves.
# With parts.npy (part id per face, -1 = skip) every part is rendered on its
# own, so no other part hides any of its faces.
import json, math, os, sys
import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT, OBJ = argv[0], argv[1]
RES = int(argv[2]) if len(argv) > 2 else 1024
NV = int(argv[3]) if len(argv) > 3 else 14
PARTS = argv[4] if len(argv) > 4 else None
MODES = os.environ.get("FM_AI_MODES", "normal,shaded").split(",")
MAXT = int(os.environ.get("FM_AI_MAXTILES", "6"))
ASPECT = float(os.environ.get("FM_AI_TILE_ASPECT", "1.1"))
os.makedirs(OUT, exist_ok=True)
scn = bpy.context.scene
ob = bpy.data.objects.get(OBJ)
if ob is None or ob.type != "MESH":
    ob = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.polygons))
for o in scn.objects:
    if o is not ob:
        o.hide_render = True
me = ob.data
nf = len(me.polygons)
# face id + 1 in base 256 over RGB, on the face corners (Workbench shows corner colours)
ids = np.arange(1, nf + 1)
lt = np.empty(nf, np.int64); me.polygons.foreach_get("loop_total", lt)
ls = np.empty(nf, np.int64); me.polygons.foreach_get("loop_start", ls)
order = np.argsort(ls)
per_loop = np.repeat(ids[order], lt[order])
rgb = np.stack([(per_loop & 255), (per_loop >> 8) & 255, (per_loop >> 16) & 255], 1) / 255.0
col = np.ones((len(per_loop), 4), np.float32); col[:, :3] = rgb
attr = me.color_attributes.new("fm_faceid", "FLOAT_COLOR", "CORNER")
attr.data.foreach_set("color", col.ravel())
me.color_attributes.active_color = attr
try:
    me.color_attributes.render_color_index = me.color_attributes.active_color_index
except Exception:
    pass

cd = bpy.data.cameras.new("FMV"); cd.type = "ORTHO"
cam = bpy.data.objects.new("FMV", cd); scn.collection.objects.link(cam); scn.camera = cam
scn.render.engine = "BLENDER_WORKBENCH"
scn.render.resolution_x = scn.render.resolution_y = RES
scn.render.film_transparent = False
sh = scn.display.shading
scn.world = bpy.data.worlds.new("W")
scn.world.color = (0, 0, 0)


def directions(n):
    dirs = [Vector(d) for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    dirs += [Vector((x, y, z)).normalized() for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    if n > len(dirs):
        g = (1 + 5 ** 0.5) / 2
        extra = [(0, 1, g), (0, -1, g), (0, 1, -g), (0, -1, -g), (1, g, 0), (-1, g, 0), (1, -g, 0), (-1, -g, 0),
                 (g, 0, 1), (-g, 0, 1), (g, 0, -1), (-g, 0, -1)]
        dirs += [Vector(e).normalized() for e in extra]
    if n == 10:                       # axes + a tetrahedron of diagonals
        dirs = dirs[:6] + [Vector(v).normalized() for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    return dirs[:n]


meta = {"object": ob.name, "faces": nf, "res": RES, "views": [], "modes": MODES}


def part_object(fmask):
    """a new object with only the faces in fmask (keeps fm_faceid)"""
    nl = len(me.loops)
    lt = np.empty(nf, np.int64); me.polygons.foreach_get("loop_total", lt)
    ls = np.empty(nf, np.int64); me.polygons.foreach_get("loop_start", ls)
    vi = np.empty(nl, np.int64); me.loops.foreach_get("vertex_index", vi)
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co)
    F = np.nonzero(fmask)[0]
    t = lt[F]
    starts = np.repeat(ls[F] - np.concatenate([[0], np.cumsum(t)[:-1]]), t)
    loops = starts + np.arange(t.sum())
    used, inv = np.unique(vi[loops], return_inverse=True)
    m = bpy.data.meshes.new("FMpart")
    m.vertices.add(len(used)); m.vertices.foreach_set("co", co.reshape(-1, 3)[used].ravel())
    m.loops.add(len(loops)); m.loops.foreach_set("vertex_index", inv.astype(np.int32))
    m.polygons.add(len(F))
    m.polygons.foreach_set("loop_start", np.concatenate([[0], np.cumsum(t)[:-1]]).astype(np.int32))
    m.polygons.foreach_set("loop_total", t.astype(np.int32))
    m.update(calc_edges=True)
    a = m.color_attributes.new("fm_faceid", "FLOAT_COLOR", "CORNER")
    a.data.foreach_set("color", col[loops].ravel())
    m.color_attributes.active_color = a
    try:
        m.color_attributes.render_color_index = m.color_attributes.active_color_index
    except Exception:
        pass
    o = bpy.data.objects.new("FMpart", m)
    o.matrix_world = ob.matrix_world
    scn.collection.objects.link(o)
    return o

def set_mode(mode):
    sh.light = "FLAT" if mode == "id" else ("MATCAP" if mode == "normal" else "STUDIO")
    sh.color_type = "VERTEX" if mode == "id" else "SINGLE"
    sh.single_color = (0.8, 0.8, 0.8)
    sh.show_cavity = mode == "shaded"
    sh.show_shadows = False
    sh.show_specular_highlight = mode == "shaded"
    sh.show_object_outline = False
    if mode == "normal":
        try:
            sh.studio_light = "check_normal+y.exr"
        except TypeError:
            pass
    scn.display.render_aa = "OFF" if mode == "id" else "8"
    scn.render.dither_intensity = 0.0
    fmt = scn.render.image_settings
    if mode == "id":
        fmt.file_format = "OPEN_EXR"
        fmt.color_depth = "32"
        fmt.exr_codec = "ZIP"
        scn.view_settings.view_transform = "Standard"
    else:
        fmt.file_format = "PNG"
        fmt.color_mode = "RGB"
        fmt.color_depth = "8"
        scn.view_settings.view_transform = "Standard"


def render_target(tob, dirs, prefix, part):
    cs = [tob.matrix_world @ Vector(c) for c in tob.bound_box]
    ctr = sum(cs, Vector()) / 8
    rad = max((c - ctr).length for c in cs)
    cd.clip_start = rad * 0.01; cd.clip_end = rad * 10
    for vi, d in enumerate(dirs):
        rot = (-d).to_track_quat("-Z", "Y") if abs(d.z) < 0.95 else (-d).to_track_quat("-Z", "X")
        right = rot @ Vector((1, 0, 0))
        upv = rot @ Vector((0, 1, 0))
        # tight framing: tiles along the longer image axis so thin parts fill the frame
        xs = [(c - ctr).dot(right) for c in cs]
        ys = [(c - ctr).dot(upv) for c in cs]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        long_ax, long_len, short_len = (right, w, h) if w >= h else (upv, h, w)
        short_len = max(short_len, long_len / 6)
        n = int(min(MAXT, max(1, math.ceil(long_len / (short_len * ASPECT)))))
        tile = max(short_len, long_len / n) * 1.25
        centre = ctr + right * cx + upv * cy
        for ti in range(n):
            off = (ti + 0.5) / n - 0.5
            c = centre + long_ax * (off * long_len)
            cam.location = c + d * rad * 3
            cam.rotation_mode = "QUATERNION"
            cam.rotation_quaternion = rot
            cd.ortho_scale = tile
            name = f"{prefix}v{vi:02d}t{ti}"
            for mode in ["id"] + MODES:
                set_mode(mode)
                ext = "exr" if mode == "id" else "png"
                scn.render.filepath = os.path.join(OUT, f"{name}_{mode}.{ext}")
                bpy.ops.render.render(write_still=True)
            meta["views"].append({"name": name, "dir": list(d), "part": part})


if PARTS:
    plab = np.load(PARTS)[:nf]
    farea = np.empty(nf, np.float32); me.polygons.foreach_get("area", farea)
    np.save(os.path.join(OUT, "parts.npy"), plab.astype(np.int32))
    meta["parts"] = "parts.npy"
    ob.hide_render = True
    for p in np.unique(plab[plab >= 0]).tolist():
        fm = plab == p
        if farea[fm].sum() < 0.01 * farea.sum():
            continue
        po = part_object(fm)
        render_target(po, directions(NV), f"p{p:02d}", int(p))
        po.hide_render = True
        print("PART", p, int(fm.sum()), "faces", flush=True)
else:
    render_target(ob, directions(NV), "", -1)
json.dump(meta, open(os.path.join(OUT, "views.json"), "w"), indent=1)
print("VIEWS DONE", len(meta["views"]))
