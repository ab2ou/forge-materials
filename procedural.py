# SPDX-License-Identifier: GPL-3.0-or-later
"""Texture-free procedural material families: wood, brick, tiles, concrete,
fabric and metal. Every family is one node group ("FM_Proc" in the material)
whose inputs are the knobs shown in the panel and on the dial's Pattern page.

2D patterns (planks, bricks, tiles, weave) are projected tri-planar so they work
on any mesh without UVs. All coordinates are in metres (see nodes.coords_group).
"""

import os

import bpy

from .nodes import (_fresh_group, _iface, _math, _maprange, _mix, _sock, _new_material,
                    _surface_node, _coords_node, _sized_mapping, _height_node,
                    _apply_category_defaults, proc_group, PROC_PRESETS, PROC_EDGE)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _vmath(nt, op, a, b=None, loc=(0, 0)):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = op
    n.location = loc
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (tuple, list)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs["Vector"]


def _scale(nt, vec, fac, loc=(0, 0)):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = "SCALE"
    n.location = loc
    nt.links.new(vec, n.inputs[0])
    s = _sock(n.inputs, "Scale")
    if isinstance(fac, (int, float)):
        s.default_value = fac
    else:
        nt.links.new(fac, s)
    return n.outputs["Vector"]


def _noise(nt, vec, scale, detail=6.0, rough=0.55, loc=(0, 0), distortion=0.0):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.location = loc
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    n.inputs["Distortion"].default_value = distortion
    nt.links.new(vec, n.inputs["Vector"])
    if isinstance(scale, (int, float)):
        n.inputs["Scale"].default_value = scale
    else:
        nt.links.new(scale, n.inputs["Scale"])
    return n.outputs["Fac"]


def _new_family(name, inputs, outputs=(("Base Color", "NodeSocketColor"),
                                       ("Roughness", "NodeSocketFloat"),
                                       ("Height", "NodeSocketFloat"))):
    ng, build = _fresh_group(name)
    if not build:
        return ng, None, None, False
    for nm, st, d, lo, hi in inputs:
        _iface(ng, nm, "INPUT", st, d, lo, hi)
    _iface(ng, "Vector", "INPUT", "NodeSocketVector")
    _iface(ng, "Normal", "INPUT", "NodeSocketVector")
    for nm, st in outputs:
        _iface(ng, nm, "OUTPUT", st)
    gi = ng.nodes.new("NodeGroupInput")
    gi.location = (-1600, 0)
    go = ng.nodes.new("NodeGroupOutput")
    go.location = (1600, 0)
    I = {s.name: s for s in gi.outputs if s.name}
    O = {s.name: s for s in go.inputs if s.name}
    return ng, I, O, True


def _blend_floats(nt, values, weights, loc):
    acc = _math(nt, "MULTIPLY", values[0], weights[0], loc=loc)
    for i in (1, 2):
        nxt = _math(nt, "MULTIPLY_ADD", values[i], weights[i], loc=(loc[0] + 150 * i, loc[1]))
        nt.links.new(acc, nxt.node.inputs[2])
        acc = nxt
    return acc


def _triplanar2(nt, vec, normal, planes, make, loc=(-1000, 0)):
    """Evaluate the 2D pattern `make(nt, uv, y)` once per projection plane and blend
    the results by the normal (|n|^4 weights). planes: three (u_index, v_index)
    pairs for the planes facing X, Y and Z. Returns {output name: blended socket}."""
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = loc
    nt.links.new(vec, sep.inputs[0])
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nsep.location = (loc[0], loc[1] - 600)
    nt.links.new(normal, nsep.inputs[0])
    ws = []
    for i in range(3):
        a = _math(nt, "ABSOLUTE", nsep.outputs[i], loc=(loc[0] + 200, loc[1] - 600 - i * 120))
        ws.append(_math(nt, "POWER", a, 4.0, loc=(loc[0] + 350, loc[1] - 600 - i * 120)))
    tot = _math(nt, "ADD", ws[0], ws[1], loc=(loc[0] + 500, loc[1] - 650))
    tot = _math(nt, "ADD", tot, ws[2], loc=(loc[0] + 650, loc[1] - 650))
    tot = _math(nt, "MAXIMUM", tot, 1e-5, loc=(loc[0] + 800, loc[1] - 650))
    ws = [_math(nt, "DIVIDE", w, tot, loc=(loc[0] + 950, loc[1] - 600 - i * 120))
          for i, w in enumerate(ws)]
    results = []
    for i, (u, v) in enumerate(planes):
        c = nt.nodes.new("ShaderNodeCombineXYZ")
        c.location = (loc[0] + 200, loc[1] + i * 400)
        nt.links.new(sep.outputs[u], c.inputs[0])
        nt.links.new(sep.outputs[v], c.inputs[1])
        results.append(make(nt, c.outputs["Vector"], loc[1] + i * 400))
    out = {}
    for j, key in enumerate(results[0]):
        first = results[0][key]
        y = loc[1] + 1400 + j * 150
        if first.type in ("RGBA", "VECTOR"):
            acc = _scale(nt, first, ws[0], loc=(loc[0] + 1200, y))
            for i in (1, 2):
                acc = _vmath(nt, "ADD", acc, _scale(nt, results[i][key], ws[i], loc=(loc[0] + 1300, y)),
                             loc=(loc[0] + 1400 + 100 * i, y))
        else:
            acc = _blend_floats(nt, [r[key] for r in results], ws, (loc[0] + 1200, y))
        out[key] = acc
    return out


PLANES_RAW = ((1, 2), (0, 2), (0, 1))     # walls keep Z up: X-facing uses (y,z) ...
PLANES_AXIS = ((2, 1), (2, 0), (0, 1))    # long axis (z) runs along the pattern's u


def _brick_maker(offset, smooth_in, I):
    """Returns make(nt, uv, y) -> {"mortar", "rand"} using a Brick Texture."""
    def make(nt, uv, y):
        b = nt.nodes.new("ShaderNodeTexBrick")
        b.offset = offset
        b.offset_frequency = 2
        b.squash = 1.0
        b.location = (-500, y)
        b.inputs["Color1"].default_value = (0, 0, 0, 1)
        b.inputs["Color2"].default_value = (1, 1, 1, 1)
        b.inputs["Mortar"].default_value = (1, 1, 1, 1)
        b.inputs["Scale"].default_value = 1.0
        b.inputs["Bias"].default_value = 0.0
        nt.links.new(uv, b.inputs["Vector"])
        nt.links.new(I["_mortar"], b.inputs["Mortar Size"])
        nt.links.new(I["_width"], b.inputs["Brick Width"])
        nt.links.new(I["_height"], b.inputs["Row Height"])
        if smooth_in is not None:
            nt.links.new(smooth_in, b.inputs["Mortar Smooth"])
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-300, y)
        nt.links.new(b.outputs["Color"], sep.inputs["Color"])
        return {"mortar": b.outputs["Fac"], "rand": sep.outputs[0]}
    return make


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------

def brick_family(name="FM_ProcBrick", offset=0.5):
    inputs = [
        ("Color", "NodeSocketColor", (0.42, 0.13, 0.07, 1), None, None),
        ("Color Variation", "NodeSocketFloat", 0.4, 0.0, 1.0),
        ("Mortar Color", "NodeSocketColor", (0.55, 0.53, 0.50, 1), None, None),
        ("Brick Width", "NodeSocketFloat", 0.215, 0.01, 5.0),
        ("Brick Height", "NodeSocketFloat", 0.065, 0.01, 5.0),
        ("Mortar Size", "NodeSocketFloat", 0.010, 0.0, 0.1),
        ("Edge Softness", "NodeSocketFloat", 0.1, 0.0, 1.0),
        ("Roughness", "NodeSocketFloat", 0.85, 0.0, 1.0),
        ("Mortar Roughness", "NodeSocketFloat", 0.95, 0.0, 1.0),
        ("Surface Noise", "NodeSocketFloat", 0.5, 0.0, 1.0),
    ]
    ng, I, O, build = _new_family(name, inputs)
    if not build:
        return ng
    nt = ng
    I = dict(I, _mortar=I["Mortar Size"], _width=I["Brick Width"], _height=I["Brick Height"])
    pat = _triplanar2(nt, I["Vector"], I["Normal"], PLANES_RAW,
                      _brick_maker(offset, I["Edge Softness"], I))
    mortar, rand = pat["mortar"], pat["rand"]
    # per-brick brightness variation + fine speckle
    var = _math(nt, "SUBTRACT", rand, 0.5, loc=(600, 400))
    var = _math(nt, "MULTIPLY", var, I["Color Variation"], loc=(750, 400))
    var = _math(nt, "MULTIPLY_ADD", var, 1.2, loc=(900, 400))
    var.node.inputs[2].default_value = 1.0
    speck = _noise(nt, I["Vector"], 90.0, 8.0, 0.6, loc=(600, 650))
    speck = _maprange(nt, speck, 0.3, 0.7, 0.8, 1.15, loc=(750, 650), smooth=False)
    k = _math(nt, "MULTIPLY", var, speck, loc=(1050, 500))
    col = _scale(nt, I["Color"], k, loc=(1150, 400))
    col = _mix(nt, "RGBA", mortar, col, I["Mortar Color"], loc=(1300, 300))
    noise = _noise(nt, I["Vector"], 25.0, 8.0, 0.6, loc=(600, -200))
    nn = _math(nt, "MULTIPLY", noise, I["Surface Noise"], loc=(800, -200))
    h = _math(nt, "SUBTRACT", 1.0, mortar, loc=(800, -50))
    h = _math(nt, "MULTIPLY_ADD", h, 0.5, loc=(950, -50))
    nt.links.new(_math(nt, "MULTIPLY", nn, 0.15, loc=(950, -200)), h.node.inputs[2])
    r = _math(nt, "MULTIPLY_ADD", speck, 0.2, loc=(1000, 100))
    nt.links.new(I["Roughness"], r.node.inputs[2])
    r = _math(nt, "SUBTRACT", r, 0.2, clamp=True, loc=(1100, 100))
    r = _mix(nt, "FLOAT", mortar, r, I["Mortar Roughness"], loc=(1300, 100))
    nt.links.new(col, O["Base Color"])
    nt.links.new(r, O["Roughness"])
    nt.links.new(h, O["Height"])
    return ng


def wood_family():
    inputs = [
        ("Light Color", "NodeSocketColor", (0.60, 0.42, 0.25, 1), None, None),
        ("Dark Color", "NodeSocketColor", (0.36, 0.22, 0.11, 1), None, None),
        ("Grain Scale", "NodeSocketFloat", 18.0, 1.0, 200.0),
        ("Grain Distortion", "NodeSocketFloat", 4.0, 0.0, 30.0),
        ("Fibers", "NodeSocketFloat", 0.5, 0.0, 1.0),
        ("Planks", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Plank Width", "NodeSocketFloat", 0.14, 0.02, 2.0),
        ("Plank Length", "NodeSocketFloat", 1.2, 0.1, 10.0),
        ("Gap", "NodeSocketFloat", 0.0025, 0.0, 0.02),
        ("Plank Variation", "NodeSocketFloat", 0.35, 0.0, 1.0),
        ("Roughness", "NodeSocketFloat", 0.45, 0.0, 1.0),
    ]
    ng, I, O, build = _new_family("FM_ProcWood", inputs)
    if not build:
        return ng
    nt = ng
    gap = _math(nt, "MULTIPLY", I["Gap"], I["Planks"], loc=(-1400, 900))
    I2 = dict(I, _mortar=gap, _width=I["Plank Length"], _height=I["Plank Width"])
    pat = _triplanar2(nt, I["Vector"], I["Normal"], PLANES_AXIS, _brick_maker(0.5, None, I2))
    mortar, rand = pat["mortar"], pat["rand"]
    rp = _math(nt, "MULTIPLY", rand, I["Planks"], loc=(600, 700))
    # per-plank ring centre offset so every board shows a different cut
    off = nt.nodes.new("ShaderNodeCombineXYZ")
    off.location = (750, 700)
    nt.links.new(_math(nt, "MULTIPLY_ADD", rp, 0.9, loc=(650, 800)), off.inputs[0])
    nt.links.new(_math(nt, "MULTIPLY_ADD", rp, 1.7, loc=(650, 650)), off.inputs[1])
    off.inputs[0].links[0].from_node.inputs[2].default_value = 0.35
    off.inputs[1].links[0].from_node.inputs[2].default_value = 0.25
    pv = _vmath(nt, "ADD", I["Vector"], off.outputs["Vector"], loc=(850, 700))
    # stretch along the board so the grain's waviness has a long wavelength
    pv = _vmath(nt, "MULTIPLY", pv, (1.0, 1.0, 0.08), loc=(950, 700))
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "RINGS"
    wave.rings_direction = "Z"
    wave.wave_profile = "SAW"
    wave.location = (1050, 700)
    wave.inputs["Detail"].default_value = 4.0
    wave.inputs["Detail Scale"].default_value = 1.2
    nt.links.new(pv, wave.inputs["Vector"])
    nt.links.new(I["Grain Scale"], wave.inputs["Scale"])
    nt.links.new(I["Grain Distortion"], wave.inputs["Distortion"])
    ring = _maprange(nt, wave.outputs["Fac"], 0.15, 0.95, loc=(1200, 700))
    # fibres: noise stretched along the long axis (z)
    fv = _vmath(nt, "MULTIPLY", I["Vector"], (300.0, 300.0, 6.0), loc=(900, 400))
    fib = _noise(nt, fv, 1.0, 4.0, 0.5, loc=(1050, 400))
    fib = _maprange(nt, fib, 0.35, 0.65, loc=(1200, 400))
    fk = _math(nt, "MULTIPLY", fib, I["Fibers"], loc=(1300, 400))
    fk = _math(nt, "MULTIPLY_ADD", fk, -0.25, loc=(1400, 400))
    fk.node.inputs[2].default_value = 1.0
    col = _mix(nt, "RGBA", ring, I["Light Color"], I["Dark Color"], loc=(1300, 650))
    pvv = _math(nt, "SUBTRACT", rp, 0.5, loc=(1100, 550))
    pvv = _math(nt, "MULTIPLY", pvv, I["Plank Variation"], loc=(1200, 550))
    pvv = _math(nt, "MULTIPLY_ADD", pvv, 0.8, loc=(1300, 550))
    pvv.node.inputs[2].default_value = 1.0
    k = _math(nt, "MULTIPLY", fk, pvv, loc=(1400, 550))
    col = _scale(nt, col, k, loc=(1450, 650))
    col = _mix(nt, "RGBA", mortar, col, (0.03, 0.02, 0.015, 1.0), loc=(1500, 600))
    h = _math(nt, "SUBTRACT", 1.0, mortar, loc=(1300, 0))
    h = _math(nt, "MULTIPLY_ADD", h, 0.4, loc=(1400, 0))
    nt.links.new(_math(nt, "MULTIPLY", ring, 0.06, loc=(1300, -100)), h.node.inputs[2])
    r = _math(nt, "MULTIPLY_ADD", ring, 0.12, loc=(1300, 200))
    nt.links.new(I["Roughness"], r.node.inputs[2])
    r = _mix(nt, "FLOAT", mortar, r, 1.0, loc=(1450, 200))
    nt.links.new(col, O["Base Color"])
    nt.links.new(r, O["Roughness"])
    nt.links.new(h, O["Height"])
    return ng


def concrete_family():
    inputs = [
        ("Color", "NodeSocketColor", (0.48, 0.47, 0.45, 1), None, None),
        ("Variation", "NodeSocketFloat", 0.5, 0.0, 1.0),
        ("Pores", "NodeSocketFloat", 0.4, 0.0, 1.0),
        ("Stains", "NodeSocketFloat", 0.3, 0.0, 1.0),
        ("Roughness", "NodeSocketFloat", 0.85, 0.0, 1.0),
    ]
    ng, I, O, build = _new_family("FM_ProcConcrete", inputs)
    if not build:
        return ng
    nt = ng
    v = I["Vector"]
    big = _noise(nt, v, 2.0, 8.0, 0.6, loc=(-800, 400))
    big = _maprange(nt, big, 0.35, 0.65, loc=(-600, 400), smooth=False)
    bv = _math(nt, "SUBTRACT", big, 0.5, loc=(-450, 400))
    bv = _math(nt, "MULTIPLY", bv, I["Variation"], loc=(-300, 400))
    bv = _math(nt, "MULTIPLY_ADD", bv, 0.6, loc=(-150, 400))
    bv.node.inputs[2].default_value = 1.0
    st = _noise(nt, v, 0.8, 3.0, 0.5, loc=(-800, 150))
    st = _maprange(nt, st, 0.55, 0.7, loc=(-600, 150))
    st = _math(nt, "MULTIPLY", st, I["Stains"], loc=(-450, 150))
    st = _math(nt, "MULTIPLY_ADD", st, -0.45, loc=(-300, 150))
    st.node.inputs[2].default_value = 1.0
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.location = (-800, -150)
    vo.inputs["Scale"].default_value = 350.0
    nt.links.new(v, vo.inputs["Vector"])
    pr = _math(nt, "MULTIPLY", I["Pores"], 0.25, loc=(-600, -300))
    pit = _math(nt, "LESS_THAN", _sock(vo.outputs, "Distance"), pr, loc=(-450, -150))
    fine = _noise(nt, v, 150.0, 6.0, 0.6, loc=(-800, -450))
    k = _math(nt, "MULTIPLY", bv, st, loc=(0, 300))
    pk = _math(nt, "MULTIPLY_ADD", pit, -0.35, loc=(0, 100))
    pk.node.inputs[2].default_value = 1.0
    k = _math(nt, "MULTIPLY", k, pk, loc=(150, 250))
    col = _scale(nt, I["Color"], k, loc=(300, 300))
    h2 = _math(nt, "SUBTRACT", _math(nt, "MULTIPLY_ADD", fine, 0.15, loc=(0, -450)),
               _math(nt, "MULTIPLY", pit, 0.25, loc=(0, -600)), loc=(200, -500))
    h2 = _math(nt, "ADD", h2, 0.45, loc=(350, -500))
    r = _math(nt, "MULTIPLY_ADD", fine, 0.15, loc=(200, -100))
    nt.links.new(I["Roughness"], r.node.inputs[2])
    r = _math(nt, "SUBTRACT", r, 0.07, clamp=True, loc=(350, -100))
    nt.links.new(col, O["Base Color"])
    nt.links.new(r, O["Roughness"])
    nt.links.new(h2, O["Height"])
    return ng


def fabric_family():
    inputs = [
        ("Warp Color", "NodeSocketColor", (0.78, 0.72, 0.62, 1), None, None),
        ("Weft Color", "NodeSocketColor", (0.72, 0.66, 0.56, 1), None, None),
        ("Thread Density", "NodeSocketFloat", 300.0, 20.0, 3000.0),
        ("Fuzz", "NodeSocketFloat", 0.4, 0.0, 1.0),
        ("Roughness", "NodeSocketFloat", 0.9, 0.0, 1.0),
    ]
    ng, I, O, build = _new_family("FM_ProcFabric", inputs)
    if not build:
        return ng
    nt = ng
    wscale = _math(nt, "DIVIDE", I["Thread Density"], 3.1831, loc=(-1400, 900))

    def make(nt, uv, y):
        ch = nt.nodes.new("ShaderNodeTexChecker")
        ch.location = (-500, y)
        ch.inputs["Color1"].default_value = (1, 1, 1, 1)
        ch.inputs["Color2"].default_value = (0, 0, 0, 1)
        nt.links.new(uv, ch.inputs["Vector"])
        nt.links.new(I["Thread Density"], ch.inputs["Scale"])
        wu = nt.nodes.new("ShaderNodeTexWave")
        wu.wave_type = "BANDS"
        wu.bands_direction = "X"
        wu.location = (-500, y - 150)
        nt.links.new(uv, wu.inputs["Vector"])
        nt.links.new(wscale, wu.inputs["Scale"])
        wv = nt.nodes.new("ShaderNodeTexWave")
        wv.wave_type = "BANDS"
        wv.bands_direction = "Y"
        wv.location = (-500, y - 300)
        nt.links.new(uv, wv.inputs["Vector"])
        nt.links.new(wscale, wv.inputs["Scale"])
        sel = ch.outputs["Fac"]
        h = _mix(nt, "FLOAT", sel, wu.outputs["Fac"], wv.outputs["Fac"], loc=(-300, y))
        return {"sel": sel, "h": h}

    pat = _triplanar2(nt, I["Vector"], I["Normal"], PLANES_AXIS, make)
    col = _mix(nt, "RGBA", pat["sel"], I["Weft Color"], I["Warp Color"], loc=(800, 400))
    fz = _noise(nt, _vmath(nt, "MULTIPLY", I["Vector"], (400.0, 400.0, 400.0), loc=(600, 650)),
                1.0, 4.0, 0.6, loc=(750, 650))
    fz = _maprange(nt, fz, 0.3, 0.7, 0.75, 1.15, loc=(900, 650), smooth=False)
    one = _math(nt, "SUBTRACT", fz, 1.0, loc=(1000, 650))
    fzk = _math(nt, "MULTIPLY_ADD", one, I["Fuzz"], loc=(1100, 650))
    fzk.node.inputs[2].default_value = 1.0
    shade = _maprange(nt, pat["h"], 0.0, 1.0, 0.75, 1.05, loc=(900, 500), smooth=False)
    k = _math(nt, "MULTIPLY", fzk, shade, loc=(1200, 550))
    col = _scale(nt, col, k, loc=(1300, 450))
    h = _math(nt, "MULTIPLY_ADD", pat["h"], 0.5, loc=(1000, 0))
    h.node.inputs[2].default_value = 0.25
    nt.links.new(col, O["Base Color"])
    nt.links.new(I["Roughness"], O["Roughness"])
    nt.links.new(h, O["Height"])
    return ng


def metal_family():
    return proc_group()


# family key: (builder, coordinates, uses normal)
FAMILIES = {
    "WOOD": (wood_family, "axis", True),
    "BRICK": (lambda: brick_family("FM_ProcBrick", 0.5), "raw", True),
    "TILES": (lambda: brick_family("FM_ProcTiles", 0.0), "raw", True),
    "CONCRETE": (concrete_family, "raw", False),
    "FABRIC": (fabric_family, "axis", True),
    "METAL": (metal_family, "axis", False),
}

# knobs shown on the dial's Pattern page: (input, label, lo, hi, fmt, hint)
FAMILY_PARAMS = {
    "WOOD": [("Grain Scale", "Grain", 1, 80, "int", "Growth ring density"),
             ("Grain Distortion", "Figure", 0, 20, "num", "How wavy the grain is"),
             ("Fibers", "Fibres", 0, 1, "pct", "Fine fibre streaks"),
             ("Planks", "Planks", 0, 1, "pct", "0 = solid wood, 1 = boards"),
             ("Plank Width", "Board Width", 0.03, 0.5, "cm", "Width of each board"),
             ("Plank Length", "Board Length", 0.2, 5, "num", "Length of each board (m)"),
             ("Gap", "Gap", 0, 0.01, "mm", "Gap between boards"),
             ("Plank Variation", "Variation", 0, 1, "pct", "Colour change per board")],
    "BRICK": [("Color Variation", "Variation", 0, 1, "pct", "Colour change per brick"),
              ("Brick Width", "Brick Width", 0.05, 1.0, "cm", "Brick length"),
              ("Brick Height", "Brick Height", 0.02, 0.5, "cm", "Course height"),
              ("Mortar Size", "Mortar", 0, 0.04, "mm", "Joint width"),
              ("Edge Softness", "Softness", 0, 1, "pct", "Rounded brick edges"),
              ("Surface Noise", "Roughness Noise", 0, 1, "pct", "Surface unevenness")],
    "TILES": [("Color Variation", "Variation", 0, 1, "pct", "Colour change per tile"),
              ("Brick Width", "Tile Width", 0.02, 1.5, "cm", "Tile width"),
              ("Brick Height", "Tile Height", 0.02, 1.5, "cm", "Tile height"),
              ("Mortar Size", "Grout", 0, 0.02, "mm", "Grout width"),
              ("Edge Softness", "Bevel", 0, 1, "pct", "Rounded tile edges")],
    "CONCRETE": [("Variation", "Variation", 0, 1, "pct", "Cloudy colour variation"),
                 ("Pores", "Pores", 0, 1, "pct", "Air holes"),
                 ("Stains", "Stains", 0, 1, "pct", "Water stains")],
    "FABRIC": [("Thread Density", "Threads", 20, 2000, "int", "Threads per metre"),
               ("Fuzz", "Fuzz", 0, 1, "pct", "Fibre fuzz")],
    "METAL": [("Roughness Variation", "Variation", 0, 1, "pct", "Brushed variation"),
              ("Damascus", "Damascus", 0, 1, "pct", "Folded-steel pattern"),
              ("Damascus Scale", "Pattern Size", 10, 300, "int", "Damascus line density"),
              ("Hammered", "Hammered", 0, 1, "pct", "Hammer marks")],
}

# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

def _metal_presets():
    out = {}
    for key, (label, values) in PROC_PRESETS.items():
        out[key] = {"label": label, "family": "METAL", "category": "METAL",
                    "values": values, "metallic": 1.0, "swatch": values["Color A"],
                    "edge": PROC_EDGE[key],
                    "rust": (0.16, 0.36, 0.29, 1.0) if key in ("BRONZE", "COPPER") else None}
    return out


PRESETS = {
    "OAK_PLANKS": {"label": "Oak Planks", "family": "WOOD", "category": "WOOD", "values": {},
                   "swatch": (0.55, 0.38, 0.21)},
    "PINE_PLANKS": {"label": "Pine Planks", "family": "WOOD", "category": "WOOD",
                    "values": {"Light Color": (0.82, 0.66, 0.45, 1), "Dark Color": (0.62, 0.43, 0.24, 1),
                               "Plank Width": 0.18, "Grain Scale": 14.0},
                    "swatch": (0.78, 0.6, 0.38)},
    "WALNUT": {"label": "Walnut (solid)", "family": "WOOD", "category": "WOOD",
               "values": {"Light Color": (0.38, 0.24, 0.14, 1), "Dark Color": (0.17, 0.09, 0.05, 1),
                          "Planks": 0.0, "Grain Scale": 22.0, "Roughness": 0.35},
               "swatch": (0.3, 0.18, 0.1)},
    "WEATHERED_PLANKS": {"label": "Weathered Planks", "family": "WOOD", "category": "WOOD",
                         "values": {"Light Color": (0.52, 0.50, 0.46, 1),
                                    "Dark Color": (0.30, 0.29, 0.27, 1), "Roughness": 0.75,
                                    "Plank Variation": 0.5, "Gap": 0.004},
                         "swatch": (0.45, 0.43, 0.4)},
    "RED_BRICK": {"label": "Red Brick", "family": "BRICK", "category": "BRICK", "values": {},
                  "swatch": (0.42, 0.13, 0.07)},
    "OLD_BRICK": {"label": "Old Brick", "family": "BRICK", "category": "BRICK",
                  "values": {"Color": (0.35, 0.17, 0.10, 1), "Color Variation": 0.7,
                             "Mortar Color": (0.40, 0.38, 0.35, 1), "Edge Softness": 0.3},
                  "swatch": (0.35, 0.17, 0.1)},
    "WHITE_BRICK": {"label": "White Brick", "family": "BRICK", "category": "BRICK",
                    "values": {"Color": (0.82, 0.80, 0.76, 1), "Color Variation": 0.15,
                               "Mortar Color": (0.60, 0.60, 0.58, 1)},
                    "swatch": (0.82, 0.8, 0.76)},
    "STONE_BLOCKS": {"label": "Stone Blocks", "family": "BRICK", "category": "STONE",
                     "values": {"Color": (0.52, 0.50, 0.46, 1), "Color Variation": 0.35,
                                "Brick Width": 0.6, "Brick Height": 0.3, "Mortar Size": 0.012,
                                "Mortar Color": (0.35, 0.34, 0.32, 1), "Roughness": 0.9,
                                "Edge Softness": 0.25},
                     "swatch": (0.52, 0.5, 0.46)},
    "WHITE_TILES": {"label": "White Tiles", "family": "TILES", "category": "TILES",
                    "values": {"Color": (0.86, 0.86, 0.84, 1), "Color Variation": 0.05,
                               "Mortar Color": (0.45, 0.45, 0.43, 1), "Brick Width": 0.15,
                               "Brick Height": 0.15, "Mortar Size": 0.003, "Roughness": 0.15,
                               "Mortar Roughness": 0.8, "Surface Noise": 0.1, "Edge Softness": 0.2},
                    "swatch": (0.86, 0.86, 0.84)},
    "TERRACOTTA": {"label": "Terracotta Tiles", "family": "TILES", "category": "TILES",
                   "values": {"Color": (0.55, 0.24, 0.12, 1), "Color Variation": 0.3,
                              "Mortar Color": (0.5, 0.47, 0.42, 1), "Brick Width": 0.3,
                              "Brick Height": 0.3, "Mortar Size": 0.008, "Roughness": 0.6},
                   "swatch": (0.55, 0.24, 0.12)},
    "BLACK_TILES": {"label": "Black Tiles", "family": "TILES", "category": "TILES",
                    "values": {"Color": (0.03, 0.03, 0.03, 1), "Color Variation": 0.05,
                               "Mortar Color": (0.25, 0.25, 0.25, 1), "Brick Width": 0.3,
                               "Brick Height": 0.3, "Mortar Size": 0.003, "Roughness": 0.1,
                               "Surface Noise": 0.1},
                    "swatch": (0.05, 0.05, 0.05)},
    "CONCRETE": {"label": "Concrete", "family": "CONCRETE", "category": "CONCRETE", "values": {},
                 "swatch": (0.48, 0.47, 0.45)},
    "DARK_CONCRETE": {"label": "Dark Concrete", "family": "CONCRETE", "category": "CONCRETE",
                      "values": {"Color": (0.24, 0.24, 0.23, 1), "Stains": 0.5},
                      "swatch": (0.24, 0.24, 0.23)},
    "WHITE_PLASTER": {"label": "White Plaster", "family": "CONCRETE", "category": "PLASTER",
                      "values": {"Color": (0.85, 0.84, 0.80, 1), "Variation": 0.2, "Pores": 0.05,
                                 "Stains": 0.15, "Roughness": 0.9},
                      "swatch": (0.85, 0.84, 0.8)},
    "LINEN": {"label": "Linen", "family": "FABRIC", "category": "FABRIC", "values": {},
              "swatch": (0.76, 0.7, 0.6)},
    "DENIM": {"label": "Denim", "family": "FABRIC", "category": "FABRIC",
              "values": {"Warp Color": (0.10, 0.17, 0.32, 1), "Weft Color": (0.30, 0.34, 0.42, 1),
                         "Thread Density": 500.0, "Fuzz": 0.3},
              "swatch": (0.15, 0.22, 0.38)},
    "RED_CLOTH": {"label": "Red Cloth", "family": "FABRIC", "category": "FABRIC",
                  "values": {"Warp Color": (0.45, 0.05, 0.05, 1), "Weft Color": (0.38, 0.04, 0.04, 1),
                             "Thread Density": 400.0},
                  "swatch": (0.42, 0.05, 0.05)},
}
PRESETS.update(_metal_presets())


THUMB_DIR = os.path.join(os.path.dirname(__file__), "thumbs")


def preset_item(k):
    p = PRESETS[k]
    it = {"name": k, "label": p["label"], "proc": True, "color": p["swatch"],
          "category": p["category"]}
    path = os.path.join(THUMB_DIR, k + ".png")
    if os.path.exists(path):
        it.update(thumb=path, thumb_key="proc_" + k, thumb_direct=True)
    return it


def presets_in(category):
    return [preset_item(k) for k, p in PRESETS.items() if p["category"] == category]


def build(key):
    p = PRESETS[key]
    fam = p["family"]
    group_fn, coord_kind, uses_normal = FAMILIES[fam]
    mat, nt, bsdf = _new_material("FM_" + p["label"].replace(" ", "_").replace("(", "").replace(")", ""))
    mat["fm_source"] = key
    mat["fm_kind"] = "procedural"
    mat["fm_family"] = fam

    crd = _coords_node(nt, (-1900, 300))
    src = crd.outputs["Object (m)" if coord_kind == "axis" else "Object raw (m)"]
    mp = _sized_mapping(nt, src, 1.0, 1.0)
    pg = nt.nodes.new("ShaderNodeGroup")
    pg.node_tree = group_fn()
    pg.name = "FM_Proc"
    pg.label = p["label"]
    pg.location = (-150, 0)
    pg.width = 230
    for k, v in p["values"].items():
        if k in pg.inputs:
            pg.inputs[k].default_value = v
    nt.links.new(mp.outputs["Vector"], pg.inputs["Vector"])
    if uses_normal and "Normal" in pg.inputs:
        nt.links.new(crd.outputs["Normal (axis)" if coord_kind == "axis" else "Normal"],
                     pg.inputs["Normal"])

    surf = _surface_node(nt)
    nt.links.new(src, surf.inputs["Vector"])
    nt.links.new(pg.outputs["Base Color"], surf.inputs["Base Color"])
    nt.links.new(pg.outputs["Roughness"], surf.inputs["Roughness"])
    nt.links.new(pg.outputs["Height"], surf.inputs["Height"])
    surf.inputs["Metallic"].default_value = p.get("metallic", 0.0)
    _apply_category_defaults(surf, "METAL" if fam == "METAL" else p["category"])
    if p.get("edge"):
        surf.inputs["Edge Color"].default_value = p["edge"]
    if p.get("rust"):
        surf.inputs["Rust Color"].default_value = p["rust"]

    bump = nt.nodes.new("ShaderNodeBump")
    bump.name = "FM_Bump"
    bump.label = "Detail Bump"
    bump.location = (300, -400)
    bump.inputs["Strength"].default_value = 0.35 if fam == "METAL" else 0.6
    bump.inputs["Distance"].default_value = 0.002 if fam == "METAL" else 0.004
    nt.links.new(surf.outputs["Height"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(surf.outputs["Base Color"], bsdf.inputs["Base Color"])
    nt.links.new(surf.outputs["Roughness"], bsdf.inputs["Roughness"])
    nt.links.new(surf.outputs["Metallic"], bsdf.inputs["Metallic"])
    _height_node(nt, surf.outputs["Height"])
    return mat
