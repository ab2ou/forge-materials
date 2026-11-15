# SPDX-License-Identifier: GPL-3.0-or-later
"""The Forge Materials dial: a circular material picker / editor drawn in the 3D viewport.

Layout (outer to inner):
  top bar     breadcrumb (category · style), drag to move, "-" to minimise
  outer ring  categories / styles (thumbnails) or the page's parameter knobs
  page ring   Material, Style, Scale, Colour, Surface, Edges, Dirt, Rust, Detail
  centre      big knob for the selected parameter (drag / scroll)
  bottom      Random, Reset, Advanced, Close, resize grip

Only materials are touched: mesh data is never modified.
"""

import colorsys
import math
import os
import random

import blf
import bpy
import gpu
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

from . import library, looks, nodes, procedural

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------

ORANGE = (0.98, 0.52, 0.13, 1.0)
WHITE = (0.95, 0.95, 0.95, 1.0)
DIM = (0.70, 0.70, 0.70, 1.0)
OUTER = (0.36, 0.36, 0.37, 0.93)
INNER = (0.11, 0.11, 0.12, 0.97)
SECTOR = (0.16, 0.16, 0.17, 1.0)
SECTOR_HOVER = (0.24, 0.24, 0.25, 1.0)
TRACK = (0.27, 0.27, 0.29, 1.0)
PILL = (0.12, 0.12, 0.13, 0.95)
PILL_HOVER = (0.22, 0.22, 0.24, 0.95)
SHADOW = (0.0, 0.0, 0.0, 0.25)

ARC_START = math.radians(225)   # knob sweep: 225 deg -> -45 deg (270 deg)
ARC_SWEEP = math.radians(270)

# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


class Param:
    def __init__(self, label, kind, name, lo, hi, fmt="pct", hint=""):
        self.label, self.kind, self.name = label, kind, name
        self.lo, self.hi, self.fmt, self.hint = lo, hi, fmt, hint

    # -- material access ------------------------------------------------------
    def _socket(self, mat):
        nt = mat.node_tree
        if self.kind == "surf":
            n = nt.nodes.get("FM_Surface")
            return n.inputs.get(self.name) if n else None
        if self.kind == "proc":
            n = nt.nodes.get("FM_Proc")
            return n.inputs.get(self.name) if n else None
        if self.kind == "bump":
            n = nt.nodes.get("FM_Bump")
            return n.inputs.get("Strength") if n else None
        if self.kind == "nmap":
            n = nt.nodes.get("FM_NormalMap")
            return n.inputs.get("Strength") if n else None
        if self.kind == "size":
            n = nt.nodes.get("FM_Size")
            return n.outputs[0] if n else None
        if self.kind == "rot":
            n = nt.nodes.get("FM_Mapping")
            return n.inputs.get("Rotation") if n else None
        return None

    def available(self, mat):
        if mat is None or not mat.node_tree:
            return False
        if self.kind in ("hue", "sat"):
            return mat.node_tree.nodes.get("FM_Surface") is not None
        s = self._socket(mat)
        return s is not None and not s.is_linked

    def get(self, mat):
        if self.kind == "hue":
            return float(mat.get("fm_hue", 0.08))
        if self.kind == "sat":
            return float(mat.get("fm_sat", 0.0))
        s = self._socket(mat)
        v = s.default_value
        return float(v[2]) if self.kind == "rot" else float(v)

    def set(self, mat, v):
        v = max(self.lo, min(self.hi, v))
        if self.kind in ("hue", "sat"):
            mat["fm_hue" if self.kind == "hue" else "fm_sat"] = v
            h, s = float(mat.get("fm_hue", 0.08)), float(mat.get("fm_sat", 0.0))
            r, g, b = colorsys.hsv_to_rgb(h, s, 1.0)
            mat.node_tree.nodes["FM_Surface"].inputs["Tint"].default_value = (r, g, b, 1.0)
            return
        s = self._socket(mat)
        if self.kind == "rot":
            s.default_value = (0.0, 0.0, v)
        else:
            s.default_value = v

    def norm(self, mat):
        return (self.get(mat) - self.lo) / (self.hi - self.lo)

    def text(self, mat):
        v = self.get(mat)
        f = self.fmt
        if f == "pct":
            return f"{v * 100:.0f}%"
        if f == "mul":
            return f"×{v:.2f}"
        if f == "deg":
            return f"{v * 360:.0f}°"
        if f == "rad":
            return f"{math.degrees(v):.0f}\u00b0"
        if f == "mm":
            return f"{v * 1000:.1f} mm"
        if f == "cm":
            return f"{v * 100:.1f} cm"
        if f == "int":
            return f"{v:.0f}"
        return f"{v:.2f}"


PAGES = [
    ("MATERIAL", "Material", []),
    ("STYLE", "Style", []),
    ("PATTERN", "Pattern", []),   # filled per procedural family
    ("SCALE", "Scale", [
        Param("Size", "size", "", 0.1, 10.0, "mul", "\u00d71 = real-world size"),
        Param("Rotation", "rot", "", 0.0, 6.2832, "rad", "Turn the pattern"),
    ]),
    ("COLOUR", "Colour", [
        Param("Tint", "sat", "", 0.0, 1.0, "pct", "How strongly the hue tints the material"),
        Param("Hue", "hue", "", 0.0, 1.0, "deg", "Tint colour"),
        Param("Brightness", "surf", "Brightness", 0.0, 2.0, "mul", "Overall brightness"),
    ]),
    ("SURFACE", "Surface", [
        Param("Roughness", "surf", "Roughness Offset", -0.5, 0.5, "num", "+ rougher / - glossier"),
        Param("Gloss Range", "surf", "Roughness Scale", 0.0, 2.0, "mul", "Roughness contrast"),
        Param("Metal", "surf", "Metallic Strength", 0.0, 1.0, "pct", "Metallic strength"),
    ]),
    ("EDGES", "Edges", [
        Param("Wear", "surf", "Edge Wear", 0.0, 1.0, "pct", "Polished, chipped edges"),
        Param("Width", "surf", "Edge Width", 0.0, 0.02, "mm", "Keep below half the blade thickness"),
        Param("Edge Gloss", "surf", "Edge Roughness", 0.0, 1.0, "num", "Roughness of worn edges"),
        Param("Curvature", "surf", "Curvature Boost", 0.0, 1.0, "pct", "Dense / rounded meshes only"),
    ]),
    ("DIRT", "Dirt", [
        Param("Dirt", "surf", "Dirt", 0.0, 1.0, "pct", "Grime in cavities and on surfaces"),
        Param("Spread", "surf", "Dirt Spread", 0.002, 0.1, "cm", "How far grime creeps from corners"),
        Param("Dirt Gloss", "surf", "Dirt Roughness", 0.0, 1.0, "num", "Roughness of grime"),
    ]),
    ("RUST", "Rust", [
        Param("Rust", "surf", "Rust", 0.0, 1.0, "pct", "Corrosion patches"),
        Param("Rust Size", "surf", "Rust Scale", 2.0, 150.0, "int", "Patch frequency"),
    ]),
    ("WEATHER", "Weather", [
        Param("Snow", "surf", "Snow", 0.0, 1.0, "pct", "Snow on upward faces"),
        Param("Moss", "surf", "Moss", 0.0, 1.0, "pct", "Moss on tops and in cavities"),
        Param("Wetness", "surf", "Wetness", 0.0, 1.0, "pct", "Darker, glossier when wet"),
        Param("Puddles", "surf", "Puddles", 0.0, 1.0, "pct", "Water pooling on flat ground"),
        Param("Paint", "surf", "Paint", 0.0, 1.0, "pct", "Paint coat (colour in the panel)"),
        Param("Paint Wear", "surf", "Paint Wear", 0.0, 1.0, "pct", "Chipped / peeling paint"),
        Param("Moss Size", "surf", "Moss Scale", 0.5, 100.0, "int", "Moss patch frequency"),
    ]),
    ("DETAIL", "Detail", [
        Param("Bump", "bump", "", 0.0, 2.0, "mul", "Height detail strength"),
        Param("Normal", "nmap", "", 0.0, 3.0, "mul", "Normal map strength"),
        Param("Breakup", "surf", "Breakup Scale", 2.0, 200.0, "int", "Size of wear / dirt breakup"),
    ]),
]
PAGE_KEYS = [p[0] for p in PAGES]

CATEGORY_ORDER = [
    (k, lbl) for k, lbl, *_ in library.CATEGORIES] + [("OTHER", "Other")]
RING_SLOTS = 14

# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------

_SH = {}


def _uniform():
    if "u" not in _SH:
        _SH["u"] = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _SH["u"]


def _thumb_shader():
    if "t" in _SH:
        return _SH["t"]
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "ModelViewProjectionMatrix")
    info.push_constant("FLOAT", "alpha")
    info.sampler(0, "FLOAT_2D", "image")
    iface = gpu.types.GPUStageInterfaceInfo("fm_thumb_iface")
    iface.smooth("VEC2", "uv")
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_in(1, "VEC2", "texCoord")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "FragColor")
    info.vertex_source(
        "void main(){ uv = texCoord; gl_Position = ModelViewProjectionMatrix * vec4(pos, 0.0, 1.0); }")
    info.fragment_source(
        "void main(){ vec2 d = uv - vec2(0.5); float r = length(d) * 2.0;"
        " float a = 1.0 - smoothstep(0.93, 1.0, r);"
        " vec4 c = texture(image, uv);"
        " float m = a * alpha * c.a;"
        " FragColor = vec4(c.rgb * m, m); }")
    _SH["t"] = gpu.shader.create_from_info(info)
    return _SH["t"]


def _tris(verts, color):
    sh = _uniform()
    batch = batch_for_shader(sh, "TRIS", {"pos": verts})
    sh.bind()
    sh.uniform_float("color", color)
    batch.draw(sh)


def disc(cx, cy, r, color, seg=64):
    v = []
    for i in range(seg):
        a0, a1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
        v += [(cx, cy), (cx + r * math.cos(a0), cy + r * math.sin(a0)),
              (cx + r * math.cos(a1), cy + r * math.sin(a1))]
    _tris(v, color)


def ring_sector(cx, cy, r0, r1, a0, a1, color, seg=None):
    seg = seg or max(4, int(abs(a1 - a0) / (2 * math.pi) * 96))
    v = []
    for i in range(seg):
        t0 = a0 + (a1 - a0) * i / seg
        t1 = a0 + (a1 - a0) * (i + 1) / seg
        p0 = (cx + r0 * math.cos(t0), cy + r0 * math.sin(t0))
        p1 = (cx + r1 * math.cos(t0), cy + r1 * math.sin(t0))
        p2 = (cx + r1 * math.cos(t1), cy + r1 * math.sin(t1))
        p3 = (cx + r0 * math.cos(t1), cy + r0 * math.sin(t1))
        v += [p0, p1, p2, p0, p2, p3]
    _tris(v, color)


def ring(cx, cy, r, w, color):
    ring_sector(cx, cy, r - w / 2, r + w / 2, 0, 2 * math.pi, color, 64)


def arc(cx, cy, r, w, a0, a1, color):
    if abs(a1 - a0) < 1e-4:
        return
    ring_sector(cx, cy, r - w / 2, r + w / 2, a0, a1, color)


def rect(x0, y0, x1, y1, color):
    _tris([(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)], color)


def pill(x0, y0, x1, y1, color):
    r = (y1 - y0) / 2
    rect(x0 + r, y0, x1 - r, y1, color)
    disc(x0 + r, y0 + r, r, color, 24)
    disc(x1 - r, y0 + r, r, color, 24)


def text(s, x, y, size, color=WHITE, align="CENTER"):
    fid = 0
    blf.size(fid, size)
    w, h = blf.dimensions(fid, s)
    if align == "CENTER":
        x -= w / 2
    elif align == "RIGHT":
        x -= w
    blf.position(fid, x, y - h / 2 + 1, 0)
    blf.color(fid, *color)
    blf.draw(fid, s)
    return w


def text_width(s, size):
    blf.size(0, size)
    return blf.dimensions(0, s)[0]


def thumb(tex, cx, cy, r, alpha=1.0):
    sh = _thumb_shader()
    batch = batch_for_shader(sh, "TRI_FAN", {
        "pos": [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)],
        "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]})
    sh.bind()
    sh.uniform_float("alpha", alpha)
    sh.uniform_sampler("image", tex)
    gpu.state.blend_set("ALPHA_PREMULT")  # shader outputs premultiplied colour
    batch.draw(sh)
    gpu.state.blend_set("ALPHA")


# --------------------------------------------------------------------------
# thumbnails (downscaled once, cached on disk, uploaded to GPU lazily)
# --------------------------------------------------------------------------

_THUMBS = {"tex": {}, "img": {}, "queue": [], "failed": set()}


def _thumb_dir():
    from .ui import cache_path
    d = os.path.join(os.path.dirname(cache_path()), "thumbs")
    os.makedirs(d, exist_ok=True)
    return d


def request_thumb(name, path, direct=False):
    """direct=True: the file is already a finished swatch (no cropping)."""
    if (name in _THUMBS["img"] or name in _THUMBS["failed"] or not path
            or name in _THUMBS["queue"]):
        return
    _THUMBS["queue"].append(name)
    _THUMBS.setdefault("paths", {})[name] = path
    if direct:
        _THUMBS.setdefault("direct", set()).add(name)


def _load_pixels(path):
    import numpy as np
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        a = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(a)
        return a.reshape(h, w, 4), img.colorspace_settings.name
    finally:
        bpy.data.images.remove(img)


def _make_swatch(src, dst, normal=None, frac=0.6):
    """Cut a square swatch so the circle is filled edge to edge, shrink to 128 px.

    With `normal` (a tangent normal map matching `src`) the colour map is lit by
    the normal map, so plain-coloured materials still show their relief.
    """
    import numpy as np
    a, cs = _load_pixels(src)
    h, w = a.shape[:2]
    x0, x1, y0, y1 = 0, w - 1, 0, h - 1
    opaque = a[..., 3] > 0.5
    if normal is None and (~opaque).any() and opaque.sum() > 0.05 * w * h:
        ys, xs = np.nonzero(opaque)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(8, int(min(x1 - x0, y1 - y0) * frac))
    sx, sy = int(cx - side / 2), int(cy - side / 2)
    crop = np.ascontiguousarray(a[sy:sy + side, sx:sx + side])
    if normal is not None:
        try:
            n, _ = _load_pixels(normal)
            if n.shape[:2] != a.shape[:2]:
                ry, rx = n.shape[0] / a.shape[0], n.shape[1] / a.shape[1]
                n = n[(np.arange(h) * ry).astype(int)][:, (np.arange(w) * rx).astype(int)]
            nc = n[sy:sy + side, sx:sx + side, :3] * 2.0 - 1.0
            light = np.array([-0.45, 0.45, 0.77], np.float32)
            light /= np.linalg.norm(light)
            shade = np.clip(0.35 + 0.85 * (nc @ light), 0.2, 1.35)
            crop[..., :3] *= shade[..., None]
        except (RuntimeError, ValueError, IndexError):
            pass
    crop[..., 3] = 1.0
    out = bpy.data.images.new(".fm_swatch_tmp", side, side, alpha=True)
    try:
        out.colorspace_settings.name = cs
    except TypeError:
        pass
    out.pixels.foreach_set(np.clip(crop, 0.0, 1.0).ravel())
    out.scale(128, 128)
    out.filepath_raw = dst
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


def process_thumb_queue(budget=2):
    """Load up to `budget` thumbnails (call from the modal timer, not from draw)."""
    done = 0
    while _THUMBS["queue"] and done < budget:
        name = _THUMBS["queue"].pop(0)
        src = _THUMBS["paths"].get(name)
        try:
            if name in _THUMBS.get("direct", ()):
                small = src
            else:
                small = os.path.join(_thumb_dir(), name + "_swatch.png")
                if not os.path.exists(small):
                    nm = _THUMBS.get("normals", {}).get(name)
                    if nm:
                        _make_swatch(src, small, normal=nm, frac=0.3)
                    else:
                        _make_swatch(src, small)
            img = bpy.data.images.load(small, check_existing=True)
            img.name = ".fm_thumb_" + name
            _THUMBS["img"][name] = img
        except (RuntimeError, OSError, ValueError):
            _THUMBS["failed"].add(name)
        done += 1
    return done


# Poly Haven previews of cloth are dark draped renders: show the colour map lit by
# the normal map instead (relief stays visible on plain-coloured fabrics)
ALBEDO_SWATCH_CATEGORIES = {"FABRIC", "LEATHER"}


def _thumb_source(it):
    """(cache key, source image, is it already a finished swatch)."""
    if it.get("thumb_direct"):
        return it.get("thumb_key", it["name"]), it.get("thumb"), True
    maps = it.get("maps") or {}
    if it.get("category") in ALBEDO_SWATCH_CATEGORIES and maps.get("color"):
        _THUMBS.setdefault("normals", {})[it["name"] + "__lit"] = maps.get("normal_gl")
        return it["name"] + "__lit", maps["color"], False
    return it.get("thumb_key", it["name"]), it.get("thumb"), False


def get_tex(name):
    tex = _THUMBS["tex"].get(name)
    if tex is not None:
        return tex
    img = _THUMBS["img"].get(name)
    if img is None:
        return None
    try:
        tex = gpu.texture.from_image(img)
    except (ReferenceError, RuntimeError):
        _THUMBS["img"].pop(name, None)
        return None
    _THUMBS["tex"][name] = tex
    return tex


# --------------------------------------------------------------------------
# helpers on scene state
# --------------------------------------------------------------------------

def _targets(context):
    return [o for o in context.selected_objects if o.type == "MESH"]


def _active_sf_material(context):
    obj = context.active_object
    if obj and obj.type == "MESH" and obj.active_material:
        m = obj.active_material
        if m.node_tree and m.node_tree.nodes.get("FM_Surface"):
            return m
    for o in _targets(context):
        m = o.active_material
        if m and m.node_tree and m.node_tree.nodes.get("FM_Surface"):
            return m
    return None


def _category_items(cat):
    """Procedural presets of the category first, then the library textures."""
    if cat == "FAV":
        return looks.favourite_items()
    if cat == "LOOKS":
        return looks.look_items()
    return procedural.presets_in(cat) + [it for it in library.items() if it["category"] == cat]


def _categories():
    out = []
    if looks.favourites():
        out.append(("FAV", "Favourites"))
    if looks.list_looks():
        out.append(("LOOKS", "My Looks"))
    return out + CATEGORY_ORDER


_SEARCH = {"q": None, "items": []}


def _search_items(q):
    q = q.lower().strip()
    if _SEARCH["q"] == q:
        return _SEARCH["items"]
    pool = [it for k, _ in CATEGORY_ORDER for it in _category_items(k)] + looks.look_items()
    words = q.split()
    res = [it for it in pool
           if all(w in (it["label"] + " " + it["name"]).lower() for w in words)]
    _SEARCH.update(q=q, items=res)
    return res


_FAMILY_PARAM_CACHE = {}


def _family_params(mat):
    """Pattern-page knobs for the material's procedural family."""
    if mat is None:
        return []
    fam = mat.get("fm_family", "")
    if fam not in _FAMILY_PARAM_CACHE:
        _FAMILY_PARAM_CACHE[fam] = [Param(lbl, "proc", name, lo, hi, fmt, hint)
                                    for name, lbl, lo, hi, fmt, hint
                                    in procedural.FAMILY_PARAMS.get(fam, [])]
    return _FAMILY_PARAM_CACHE[fam]


def _copy_controls(src, dst):
    """Carry the user's adjustments over when switching style."""
    if src is None:
        return
    s, d = src.node_tree.nodes.get("FM_Surface"), dst.node_tree.nodes.get("FM_Surface")
    if not s or not d:
        return
    for name in nodes.SURFACE_CONTROLS:
        if name in ("Edge Color", "Edge Metallic", "Edge Roughness", "Rust Color"):
            continue  # these follow the material type
        a, b = s.inputs.get(name), d.inputs.get(name)
        if a is not None and b is not None and not b.is_linked:
            b.default_value = a.default_value
    for k in ("fm_hue", "fm_sat"):
        if k in src:
            dst[k] = src[k]


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------

_STATE = {"cx": None, "cy": None, "R": None, "page": "MATERIAL", "category": "METAL",
          "ring_page": 0, "param": {}, "advanced": False, "minimised": False, "search": ""}


class FM_OT_dial(bpy.types.Operator):
    bl_idname = "fm.dial"
    bl_label = "Forge Materials Dial"
    bl_description = ("Open the material dial in the viewport: pick a material, turn knobs, "
                      "see it change live")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    # ---------------------------------------------------------------- setup
    def invoke(self, context, event):
        _STATE["kill"] = False
        if not library.items():
            try:
                from .ui import prefs, cache_path
                p = prefs()
                root = bpy.path.abspath(p.library_path)
                if not library.load_cache(cache_path(), root) and os.path.isdir(root):
                    library.scan(root, p.preferred_res)
                    library.save_cache(cache_path())
            except (KeyError, AttributeError, OSError):
                pass
        self.area = context.area
        self.region = next((r for r in context.area.regions if r.type == "WINDOW"), context.region)
        st = _STATE
        scale = context.preferences.system.ui_scale
        if st["R"] is None:
            st["R"] = 230 * scale
        if st["cx"] is None or st["cx"] > self.region.width or st["cy"] > self.region.height:
            # right-hand side of the viewport so the model stays visible
            st["R"] = min(st["R"], self.region.height * 0.36)
            st["cx"] = max(st["R"] + 60 * scale, self.region.width - st["R"] - 140 * scale)
            st["cy"] = self.region.height * 0.5
        self.created = set()
        self.drag = None          # ("knob"|"move"|"resize", data)
        self.hover = None
        self.message = ""
        self.close_requested = False
        m = _active_sf_material(context)
        if m is not None:
            src = m.get("fm_source", "")
            kind = m.get("fm_kind", "")
            if kind == "procedural" and src in procedural.PRESETS:
                st["category"] = procedural.PRESETS[src]["category"]
            else:
                it = library.get(src)
                if it:
                    st["category"] = it["category"]
            if st["page"] == "MATERIAL":
                st["page"] = "STYLE"
        self._queue_visible_thumbs()
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw, (), "WINDOW", "POST_PIXEL")
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
        context.window_manager.event_timer_remove(self._timer)
        self.area.tag_redraw()
        return {"FINISHED"}

    # ------------------------------------------------------------ geometry
    @property
    def R(self):
        return _STATE["R"]

    def _ui(self):
        return self.R / 230.0

    def _layout(self):
        st, R = _STATE, self.R
        cx, cy = st["cx"], st["cy"]
        L = {
            "cx": cx, "cy": cy, "R": R,
            "r_items": R * 0.815, "r_thumb": R * 0.082,
            "r_page_out": R * 0.615, "r_page_in": R * 0.365,
            "r_knob": R * 0.27, "knob_w": R * 0.045,
        }
        u = self._ui()
        bar_h = 26 * u
        crumb = self._crumb()
        bw = max(text_width(crumb, 13 * u) + 90 * u, 200 * u)
        L["bar"] = (cx - bw / 2, cy + R + 12 * u, cx + bw / 2, cy + R + 12 * u + bar_h)
        bx1, by0, by1 = L["bar"][2], L["bar"][1], L["bar"][3]
        L["minbtn"] = (bx1 - bar_h, by0, bx1, by1)
        names = ["Random", "Reset", "Save", "Advanced" if not st["advanced"] else "Simple", "Close"]
        bwid, gap = 82 * u, 8 * u
        total = len(names) * bwid + (len(names) - 1) * gap + gap + bar_h
        x = cx - total / 2
        y0, y1 = cy - R - 12 * u - bar_h, cy - R - 12 * u
        L["buttons"] = []
        for n in names:
            L["buttons"].append((n, (x, y0, x + bwid, y1)))
            x += bwid + gap
        L["grip"] = (x, y0, x + bar_h, y1)
        return L

    def _crumb(self):
        st = _STATE
        if st.get("search"):
            return f"Forge Materials  \u00b7  search: {st['search']}"
        cat = dict(_categories()).get(st["category"], st["category"].title())
        m = self._mat
        style = ""
        if m is not None:
            src = m.get("fm_source", "")
            if m.get("fm_kind") == "procedural" and src in procedural.PRESETS:
                style = procedural.PRESETS[src]["label"]
            else:
                it = library.get(src)
                style = it["label"] if it else src
        parts = ["Forge Materials", cat] + ([style] if style else [])
        obj = bpy.context.active_object
        if obj and obj.type == "MESH" and len(obj.material_slots) > 1:
            i = obj.active_material_index
            sm = obj.material_slots[i].material
            rn = list(obj.get("fm_region_names", []))
            nm = rn[i] if i < len(rn) and rn[i] else (sm.name if sm else "empty")
            parts.insert(1, f"[Tab] {i + 1}/{len(obj.material_slots)} {nm}")
        return "  ·  ".join(parts)

    @property
    def _mat(self):
        return _active_sf_material(bpy.context)

    def _ring_entries(self):
        """[(kind, payload)] for the outer ring of the current page."""
        st = _STATE
        page = st["page"]
        if page == "MATERIAL":
            out = []
            for key, label in _categories():
                its = _category_items(key)
                if its:
                    out.append(("cat", {"key": key, "label": f"{label} ({len(its)})",
                                        "sample": its[0]}))
            return out, 1
        if page == "STYLE":
            its = _search_items(st["search"]) if st.get("search") else _category_items(st["category"])
            pages = max(1, math.ceil(len(its) / RING_SLOTS))
            st["ring_page"] = min(st["ring_page"], pages - 1)
            chunk = its[st["ring_page"] * RING_SLOTS:(st["ring_page"] + 1) * RING_SLOTS]
            return [("item", it) for it in chunk], pages
        return [("param", p) for p in self._params()], 1

    def _params(self):
        st = _STATE
        mat = self._mat
        if st["page"] == "PATTERN":
            page = _family_params(mat)
        else:
            page = dict((k, ps) for k, _, ps in PAGES)[st["page"]]
        ps = [p for p in page if p.available(mat)] if mat else []
        if not st["advanced"]:
            ps = ps[:2]
        return ps

    def _current_param(self):
        ps = self._params()
        if not ps:
            return None
        idx = _STATE["param"].get(_STATE["page"], 0)
        return ps[min(idx, len(ps) - 1)]

    def _slots(self, entries, pages):
        # a single page spreads its items around the whole ring
        if _STATE["page"] == "STYLE" and pages > 1:
            return RING_SLOTS
        return max(len(entries), 1)

    def _item_pos(self, L, i, n):
        # start at the top, go clockwise
        a = math.pi / 2 - 2 * math.pi * i / max(n, 1)
        return (L["cx"] + L["r_items"] * math.cos(a), L["cy"] + L["r_items"] * math.sin(a), a)

    def _page_angles(self, i):
        n = len(PAGES)
        span = 2 * math.pi / n
        mid = math.pi / 2 - span * i
        return mid - span / 2, mid + span / 2, mid

    # ---------------------------------------------------------------- thumbs
    def _queue_visible_thumbs(self):
        entries, _ = self._ring_entries()
        for kind, p in entries:
            it = p["sample"] if kind == "cat" else (p if kind == "item" else None)
            if it:
                key, path, direct = _thumb_source(it)
                if path:
                    request_thumb(key, path, direct=direct)

    # -------------------------------------------------------------- hit test
    def _hit(self, x, y):
        L = self._layout()
        st = _STATE

        def inside(r):
            return r[0] <= x <= r[2] and r[1] <= y <= r[3]

        if inside(L["minbtn"]):
            return ("min", None)
        if inside(L["bar"]):
            return ("bar", None)
        if st["minimised"]:
            return None
        for name, r in L["buttons"]:
            if inside(r):
                return ("button", name)
        if inside(L["grip"]):
            return ("grip", None)
        dx, dy = x - L["cx"], y - L["cy"]
        d = math.hypot(dx, dy)
        if d > L["R"] * 1.02:
            return None
        if d <= L["r_knob"] + L["knob_w"]:
            return ("knob", None)
        if L["r_page_in"] <= d <= L["r_page_out"]:
            a = math.atan2(dy, dx)
            for i in range(len(PAGES)):
                a0, a1, _ = self._page_angles(i)
                aa = a
                while aa < a0:
                    aa += 2 * math.pi
                while aa > a0 + 2 * math.pi:
                    aa -= 2 * math.pi
                if a0 <= aa <= a1:
                    return ("page", PAGES[i][0])
        entries, pages = self._ring_entries()
        for i, (kind, p) in enumerate(entries):
            ix, iy, _ = self._item_pos(L, i, self._slots(entries, pages))
            if math.hypot(x - ix, y - iy) <= L["r_thumb"] * 1.35:
                return (kind, p if kind != "param" else i)
        return ("dial", None)

    # ----------------------------------------------------------------- modal
    def modal(self, context, event):
        if event.type == "TIMER":
            if _STATE.get("kill"):
                return self._finish(context)
            if process_thumb_queue(2):
                self.area.tag_redraw()
            return {"PASS_THROUGH"}

        x = event.mouse_x - self.region.x
        y = event.mouse_y - self.region.y
        in_region = 0 <= x <= self.region.width and 0 <= y <= self.region.height

        if event.type in {"ESC"} and event.value == "PRESS":
            if _STATE.get("search"):
                _STATE["search"] = ""
                self._queue_visible_thumbs()
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            return self._finish(context)

        # ---- dragging --------------------------------------------------
        if self.drag:
            if event.type == "MOUSEMOVE":
                self._drag_move(context, x, y, event.shift)
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self.drag = None
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}

        # Tab: next material slot (a region made by Smart Regions); the dial
        # always edits the active slot
        if event.type == "TAB" and event.value == "PRESS" and in_region:
            obj = context.active_object
            if obj and obj.type == "MESH" and len(obj.material_slots) > 1:
                step = -1 if event.shift else 1
                obj.active_material_index = (obj.active_material_index + step) % len(obj.material_slots)
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}

        hit = self._hit(x, y) if in_region else None
        if event.type == "MOUSEMOVE":
            if hit != self.hover:
                self.hover = hit
                self.area.tag_redraw()
            return {"PASS_THROUGH"}

        # ctrl+click outside the dial pastes the material onto the clicked object
        if (hit is None and in_region and event.type == "LEFTMOUSE" and event.value == "PRESS"
                and event.ctrl):
            self._paste(context, x, y)
            return {"RUNNING_MODAL"}

        if hit is None:
            return {"PASS_THROUGH"}

        kind, data = hit
        if event.value == "PRESS" and not (event.ctrl or event.alt or event.oskey):
            if event.type == "BACK_SPACE" and _STATE.get("search"):
                _STATE["search"] = _STATE["search"][:-1]
                _STATE["ring_page"] = 0
                self._queue_visible_thumbs()
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            ch = event.unicode
            if ch and ch.isprintable() and (ch.isalnum() or ch in " -_"):
                _STATE["search"] = (_STATE.get("search", "") + ch)[:24]
                _STATE["page"] = "STYLE"
                _STATE["ring_page"] = 0
                self._queue_visible_thumbs()
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            step = 1 if event.type == "WHEELUPMOUSE" else -1
            self._wheel(context, kind, data, step, event.shift)
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self._click(context, kind, data, x, y, event)
            self.area.tag_redraw()
            if self.close_requested:
                return self._finish(context)
            return {"RUNNING_MODAL"}
        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            if kind == "knob":
                self._reset_param(context)
            elif kind == "item" and not data.get("look"):
                on = looks.toggle_favourite(data["name"])
                self.message = ("Added to" if on else "Removed from") + f" favourites: {data['label']}"
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type in {"LEFTMOUSE", "RIGHTMOUSE", "MIDDLEMOUSE"}:
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    # ----------------------------------------------------------- actions
    def _click(self, context, kind, data, x, y, event):
        st = _STATE
        if kind == "min":
            st["minimised"] = not st["minimised"]
        elif kind == "bar":
            self.drag = ("move", (x - st["cx"], y - st["cy"]))
        elif kind == "grip":
            self.drag = ("resize", (x, y, st["R"]))
        elif kind == "button":
            self._button(context, data)
        elif kind == "page":
            st["page"] = data
            st["ring_page"] = 0 if data != "STYLE" else st["ring_page"]
            self._queue_visible_thumbs()
            self.message = ""
        elif kind == "cat":
            st["category"] = data["key"]
            st["page"] = "STYLE"
            st["ring_page"] = 0
            self._queue_visible_thumbs()
        elif kind == "item":
            self._apply_style(context, data)
        elif kind == "param":
            st["param"][st["page"]] = data
        elif kind == "knob":
            p = self._current_param()
            mat = self._mat
            if p and mat:
                self.drag = ("knob", (x, y, p.norm(mat)))

    def _button(self, context, name):
        st = _STATE
        if name == "Close":
            self.close_requested = True
            return
        if name in ("Advanced", "Simple"):
            st["advanced"] = not st["advanced"]
        elif name == "Random":
            self._random(context)
        elif name == "Save":
            mat = self._mat
            if mat is None:
                self.message = "Nothing to save: pick a material first"
            else:
                try:
                    _path, nm = looks.save(mat)
                    _SEARCH["q"] = None
                    self.message = f"Saved look \u201c{nm}\u201d (rename it in the panel)"
                except OSError as e:
                    self.message = f"Could not save: {e}"
        elif name == "Reset":
            mat = self._mat
            if mat:
                item = self._item_for(mat)
                if item:
                    self._apply_style(context, item, keep=False)

    def _drag_move(self, context, x, y, fine):
        st = _STATE
        kind, data = self.drag
        if kind == "move":
            st["cx"], st["cy"] = x - data[0], y - data[1]
        elif kind == "resize":
            x0, y0, r0 = data
            d = ((x - x0) - (y - y0)) * 0.5
            st["R"] = max(150 * context.preferences.system.ui_scale, min(600, r0 + d))
        elif kind == "knob":
            x0, y0, n0 = data
            p, mat = self._current_param(), self._mat
            if p and mat:
                k = 0.0012 if fine else 0.004
                n = max(0.0, min(1.0, n0 + ((x - x0) + (y - y0)) * k))
                p.set(mat, p.lo + n * (p.hi - p.lo))

    def _wheel(self, context, kind, data, step, fine):
        st = _STATE
        if kind in ("item", "dial", "cat") and st["page"] == "STYLE":
            _, pages = self._ring_entries()
            st["ring_page"] = (st["ring_page"] - step) % pages
            self._queue_visible_thumbs()
            return
        if kind == "page":
            i = PAGE_KEYS.index(st["page"])
            st["page"] = PAGE_KEYS[(i - step) % len(PAGE_KEYS)]
            self._queue_visible_thumbs()
            return
        if kind == "param":
            st["param"][st["page"]] = data
        p, mat = self._current_param(), self._mat
        if p and mat:
            n = p.norm(mat) + step * (0.005 if fine else 0.02)
            p.set(mat, p.lo + max(0.0, min(1.0, n)) * (p.hi - p.lo))

    def _reset_param(self, context):
        p, mat = self._current_param(), self._mat
        if not (p and mat):
            return
        if p.kind in ("hue", "sat"):
            p.set(mat, 0.08 if p.kind == "hue" else 0.0)
            return
        defaults = {n: d for n, _, d, *_ in nodes.SURFACE_INPUTS}
        if p.kind == "proc":
            # preset value first, else the family group's own default
            pre = procedural.PRESETS.get(mat.get("fm_source", ""), {}).get("values", {})
            if p.name in pre:
                p.set(mat, pre[p.name])
                return
            n = mat.node_tree.nodes.get("FM_Proc")
            if n and n.node_tree:
                for it in n.node_tree.interface.items_tree:
                    if getattr(it, "in_out", "") == "INPUT" and it.name == p.name:
                        p.set(mat, it.default_value)
                        return
        elif p.kind == "surf" and p.name in defaults and not isinstance(defaults[p.name], tuple):
            p.set(mat, defaults[p.name])
        elif p.kind == "size":
            p.set(mat, 1.0)
        elif p.kind == "rot":
            p.set(mat, 0.0)
        elif p.kind == "bump":
            p.set(mat, 0.6 if mat.node_tree.nodes.get("FM_NormalMap") is None else 0.0)
        elif p.kind == "nmap":
            p.set(mat, 1.0)

    def _item_for(self, mat):
        src = mat.get("fm_source", "")
        if mat.get("fm_kind") == "procedural":
            p = procedural.PRESETS.get(src)
            if p is None:
                return None
            return procedural.preset_item(src)
        return library.get(src)

    def _apply_style(self, context, item, keep=True):
        objs = _targets(context)
        if not objs:
            self.message = "Select a mesh first"
            return
        old = self._mat
        props = context.scene.fm
        try:
            if item.get("look"):
                mo = "BOX" if any(not o.data.uv_layers for o in objs) else None
                mat = looks.build(item["look"], mo)
                keep = False
                if mat is None:
                    self.message = "The look's source material is not in the library any more"
                    return
            elif item.get("proc"):
                mat = nodes.build_procedural_material(item["name"])
            else:
                mapping = props.mapping
                if any(not o.data.uv_layers for o in objs):
                    mapping = "BOX"  # no UVs: project instead (UVs are never created)
                mat = nodes.build_texture_material(item, mapping, props.tex_scale)
            for o in objs:
                nodes.prepare_object(o)
        except RuntimeError as e:
            self.message = f"Could not load: {e}"
            return
        if keep:
            _copy_controls(old, mat)
            if old is not None and not item.get("proc") and old.get("fm_kind") == "texture":
                for nm in ("FM_Size",):
                    a, b = old.node_tree.nodes.get(nm), mat.node_tree.nodes.get(nm)
                    if a and b:
                        b.outputs[0].default_value = a.outputs[0].default_value
                a, b = old.node_tree.nodes.get("FM_Mapping"), mat.node_tree.nodes.get("FM_Mapping")
                if a and b:
                    b.inputs["Rotation"].default_value = a.inputs["Rotation"].default_value
        self.created.add(mat.name)
        mat["fm_dial"] = 1  # transient dial pick: removed when replaced and unused
        for obj in objs:
            if not obj.material_slots:
                obj.data.materials.append(mat)
                continue
            slots = obj.material_slots if props.all_slots else [obj.material_slots[obj.active_material_index]]
            for slot in slots:
                prev = slot.material
                slot.material = mat
                if prev and prev != mat:
                    if (prev.name in self.created or prev.get("fm_dial")) and prev.users == 0:
                        bpy.data.materials.remove(prev)
                    elif prev.users == 0 and not prev.use_fake_user:
                        prev.use_fake_user = True  # never lose the user's own material
        self.message = ""

    def _random(self, context):
        st = _STATE
        cat = st["category"]
        its = _category_items(cat)
        if not its:
            return
        self._apply_style(context, random.choice(its), keep=False)
        mat = self._mat
        if not mat:
            return
        s = mat.node_tree.nodes["FM_Surface"]
        s.inputs["Edge Wear"].default_value = random.uniform(0.15, 0.7)
        s.inputs["Dirt"].default_value = random.uniform(0.0, 0.5)
        if cat == "METAL":
            s.inputs["Rust"].default_value = random.choice([0.0, 0.0, random.uniform(0.1, 0.5)])
        s.inputs["Roughness Offset"].default_value = random.uniform(-0.1, 0.15)

    def _paste(self, context, x, y):
        mat = self._mat
        if mat is None:
            return
        region, rv3d = self.region, None
        for sp in self.area.spaces:
            if sp.type == "VIEW_3D":
                rv3d = sp.region_3d
        if rv3d is None:
            return
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y))
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))
        dg = context.evaluated_depsgraph_get()
        hit, _loc, _n, _i, obj, _m = context.scene.ray_cast(dg, origin, direction)
        if not hit or obj is None or obj.type != "MESH":
            return
        obj = bpy.data.objects.get(obj.name)
        nodes.prepare_object(obj)
        if not obj.material_slots:
            obj.data.materials.append(mat)
        else:
            slot = obj.material_slots[obj.active_material_index]
            prev = slot.material
            slot.material = mat
            if prev and prev != mat and prev.users == 0 and not prev.use_fake_user:
                if prev.name in self.created or prev.get("fm_dial"):
                    bpy.data.materials.remove(prev)
                else:
                    prev.use_fake_user = True
        self.message = f"Pasted onto {obj.name}"

    # ------------------------------------------------------------------ draw
    def _draw(self):
        ctx = bpy.context
        if ctx.region is None or ctx.region.as_pointer() != self.region.as_pointer():
            return
        try:
            self._draw_impl(ctx)
        except ReferenceError:
            pass

    def _draw_impl(self, ctx):
        gpu.state.blend_set("ALPHA")
        st = _STATE
        L = self._layout()
        u = self._ui()
        cx, cy, R = L["cx"], L["cy"], L["R"]
        hk = self.hover[0] if self.hover else None
        hd = self.hover[1] if self.hover else None
        mat = self._mat

        # ---- top bar --------------------------------------------------
        bx0, by0, bx1, by1 = L["bar"]
        pill(bx0, by0, bx1, by1, PILL)
        for i in range(3):
            for j in range(2):
                disc(bx0 + 13 * u + j * 5 * u, by0 + 8 * u + i * 5 * u, 1.4 * u, DIM, 8)
        text(self._crumb(), (bx0 + bx1) / 2 - 8 * u, (by0 + by1) / 2, 13 * u)
        mx0, my0, mx1, my1 = L["minbtn"]
        rect(mx0 + 7 * u, (my0 + my1) / 2 - 1 * u, mx1 - 9 * u, (my0 + my1) / 2 + 1 * u,
             ORANGE if hk == "min" else WHITE)
        if st["minimised"]:
            gpu.state.blend_set("NONE")
            return

        # ---- hint line ------------------------------------------------
        p = self._current_param()
        hint = self.message
        if not hint:
            if mat is None:
                if not _targets(bpy.context):
                    hint = "Select a mesh, then pick a material"
                elif st["page"] in ("MATERIAL", "STYLE"):
                    hint = "Pick a category, then a style"
                else:
                    hint = "Pick a material first (Material / Style page)"
            elif p is not None and st["page"] not in ("MATERIAL", "STYLE"):
                hint = f"{p.label} {p.text(mat)}: {p.hint}"
            elif st["page"] == "STYLE":
                hint = "Click a style · type to search · right-click = favourite · Ctrl+click an object to paste"
        if hint:
            text(hint, cx, by1 + 14 * u, 12 * u, WHITE)

        # ---- discs ----------------------------------------------------
        disc(cx + 3 * u, cy - 5 * u, R, SHADOW, 96)
        disc(cx, cy, R, OUTER, 96)
        disc(cx, cy, L["r_page_out"] + 6 * u, INNER, 96)

        # ---- page ring ------------------------------------------------
        gap = 0.012
        for i, (key, label, _ps) in enumerate(PAGES):
            a0, a1, mid = self._page_angles(i)
            active = st["page"] == key
            col = ORANGE if active else (SECTOR_HOVER if hk == "page" and hd == key else SECTOR)
            ring_sector(cx, cy, L["r_page_in"], L["r_page_out"], a0 + gap, a1 - gap, col)
            rm = (L["r_page_in"] + L["r_page_out"]) / 2
            disabled = key not in ("MATERIAL", "STYLE") and mat is None
            tc = (0.1, 0.1, 0.1, 1) if active else ((0.45, 0.45, 0.45, 1) if disabled else WHITE)
            text(label, cx + rm * math.cos(mid), cy + rm * math.sin(mid), 12 * u, tc)

        # ---- centre knob ----------------------------------------------
        disc(cx, cy, L["r_page_in"] - 4 * u, INNER, 64)
        rk, kw = L["r_knob"], L["knob_w"]
        arc(cx, cy, rk, kw, ARC_START - ARC_SWEEP, ARC_START, TRACK)
        if p is not None and mat is not None:
            n = p.norm(mat)
            a_end = ARC_START - ARC_SWEEP * n
            arc(cx, cy, rk, kw, a_end, ARC_START, ORANGE)
            hx, hy = cx + rk * math.cos(a_end), cy + rk * math.sin(a_end)
            disc(hx, hy, kw * 0.75, WHITE, 24)
            text(p.label, cx, cy + 14 * u, 14 * u, WHITE)
            text(p.text(mat), cx, cy - 12 * u, 24 * u, ORANGE)
        else:
            label = {"MATERIAL": "Material", "STYLE": "Style"}.get(st["page"], "")
            sub = ""
            if st["page"] == "STYLE" and st.get("search"):
                label = f"\u201c{st['search']}\u201d"
                n_found = len(_search_items(st["search"]))
                sub = f"{n_found} found" if n_found else "nothing found"
            elif st["page"] == "STYLE":
                _, pages = self._ring_entries()
                sub = f"{st['ring_page'] + 1} / {pages}"
            elif st["page"] == "MATERIAL":
                sub = f"{len(library.items())} textures"
            text(label, cx, cy + 12 * u, 15 * u, WHITE)
            if sub:
                text(sub, cx, cy - 12 * u, 13 * u, ORANGE)

        # ---- outer ring ---------------------------------------------------
        entries, pages = self._ring_entries()
        n_slots = self._slots(entries, pages)
        rt = L["r_thumb"]
        cur_src = mat.get("fm_source", "") if mat else ""
        for i, (kind, payload) in enumerate(entries):
            ix, iy, a = self._item_pos(L, i, n_slots)
            hovered = hk == kind and (
                (kind == "cat" and hd["key"] == payload["key"]) or
                (kind == "item" and hd["name"] == payload["name"]) or
                (kind == "param" and hd == i))
            if kind in ("cat", "item"):
                it = payload["sample"] if kind == "cat" else payload
                selected = (kind == "cat" and payload["key"] == st["category"]) or \
                           (kind == "item" and it["name"] == cur_src)
                if selected:
                    ring(ix, iy, rt + 4 * u, 3 * u, ORANGE)
                elif hovered:
                    ring(ix, iy, rt + 4 * u, 2 * u, WHITE)
                if kind == "item" and it["name"] in looks.favourites():
                    disc(ix + rt * 0.75, iy + rt * 0.75, rt * 0.22, ORANGE, 16)
                ptex = get_tex(it["thumb_key"]) if it.get("proc") and it.get("thumb_key") else None
                if ptex is not None:
                    thumb(ptex, ix, iy, rt)
                elif it.get("proc"):
                    c = it["color"]
                    disc(ix, iy, rt, (c[0], c[1], c[2], 1.0), 32)
                    disc(ix - rt * 0.3, iy + rt * 0.3, rt * 0.35, (1, 1, 1, 0.35), 24)
                else:
                    tex = get_tex(_thumb_source(it)[0])
                    if tex is not None:
                        thumb(tex, ix, iy, rt)
                    else:  # still loading
                        disc(ix, iy, rt, (0.30, 0.30, 0.31, 1), 32)
                label = payload["label"] if kind == "cat" else it["label"]
            else:
                pp = payload
                sel = _STATE["param"].get(st["page"], 0) == i
                disc(ix, iy, rt, INNER, 32)
                arc(ix, iy, rt * 0.72, rt * 0.22, ARC_START - ARC_SWEEP, ARC_START, TRACK)
                if mat:
                    nn = pp.norm(mat)
                    arc(ix, iy, rt * 0.72, rt * 0.22, ARC_START - ARC_SWEEP * nn, ARC_START, ORANGE)
                if sel:
                    ring(ix, iy, rt + 4 * u, 3 * u, ORANGE)
                elif hovered:
                    ring(ix, iy, rt + 4 * u, 2 * u, WHITE)
                label = f"{pp.label}  {pp.text(mat)}" if mat else pp.label
            # label outside the thumbnail, pushed radially
            lx = ix + math.cos(a) * (rt + 12 * u)
            ly = iy + math.sin(a) * (rt + 12 * u)
            align = "CENTER"
            if math.cos(a) > 0.35:
                align = "LEFT"
                lx = ix + rt + 8 * u
                ly = iy
            elif math.cos(a) < -0.35:
                align = "RIGHT"
                lx = ix - rt - 8 * u
                ly = iy
            text(label, lx, ly, 11 * u, WHITE, align)

        # ---- buttons --------------------------------------------------------
        for name, (x0, y0, x1, y1) in L["buttons"]:
            col = PILL_HOVER if hk == "button" and hd == name else PILL
            pill(x0, y0, x1, y1, col)
            text(name, (x0 + x1) / 2, (y0 + y1) / 2, 12 * u, WHITE)
        gx0, gy0, gx1, gy1 = L["grip"]
        disc((gx0 + gx1) / 2, (gy0 + gy1) / 2, (gy1 - gy0) / 2, PILL_HOVER if hk == "grip" else PILL, 24)
        for k in range(3):
            o = (k + 1) * 4 * u
            x_a, y_a = gx1 - 7 * u - o, gy0 + 7 * u
            x_b, y_b = gx1 - 7 * u, gy0 + 7 * u + o
            # short diagonal hatch
            _tris([(x_a, y_a), (x_b, y_b), (x_b - 1.2 * u, y_b + 1.2 * u),
                   (x_a, y_a), (x_b - 1.2 * u, y_b + 1.2 * u), (x_a - 1.2 * u, y_a + 1.2 * u)], DIM)
        gpu.state.blend_set("NONE")


classes = (FM_OT_dial,)
_KEYMAPS = []


def register():
    for c in classes:
        bpy.utils.register_class(c)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        kmi = km.keymap_items.new("fm.dial", "D", "PRESS", ctrl=True, alt=True)
        _KEYMAPS.append((km, kmi))


def unregister():
    _STATE["kill"] = True  # running dials close themselves on their next timer tick
    for km, kmi in _KEYMAPS:
        km.keymap_items.remove(kmi)
    _KEYMAPS.clear()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
