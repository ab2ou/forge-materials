# SPDX-License-Identifier: GPL-3.0-or-later
"""Shader node groups and material construction.

Nothing in this module touches mesh data: it only creates node groups,
materials and image datablocks.
"""

import os

import bpy

SURFACE_GROUP = "FM_Surface"
PROC_GROUP = "FM_ProcMetal"
GROUP_VERSION = 13

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _sock(sockets, name, kind=None):
    """First enabled socket with this name (and optional type)."""
    for s in sockets:
        if s.name == name and getattr(s, "enabled", True):
            if kind is None or s.type == kind:
                return s
    raise KeyError(name)


def _math(nt, op, a=None, b=None, clamp=False, loc=(0, 0), label=""):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    n.location = loc
    if label:
        n.label = label
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _maprange(nt, value, fmin, fmax, tmin=0.0, tmax=1.0, loc=(0, 0), smooth=True):
    n = nt.nodes.new("ShaderNodeMapRange")
    n.data_type = "FLOAT"
    n.interpolation_type = "SMOOTHSTEP" if smooth else "LINEAR"
    n.clamp = True
    n.location = loc
    vals = {"Value": value, "From Min": fmin, "From Max": fmax, "To Min": tmin, "To Max": tmax}
    for key, v in vals.items():
        s = _sock(n.inputs, key, "VALUE")
        if isinstance(v, (int, float)):
            s.default_value = v
        else:
            nt.links.new(v, s)
    return _sock(n.outputs, "Result", "VALUE")


def _mix(nt, kind, fac, a, b, loc=(0, 0), blend="MIX"):
    """kind: 'RGBA' or 'FLOAT'."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = kind
    if kind == "RGBA":
        n.blend_type = blend
    n.clamp_factor = True
    n.location = loc
    stype = "RGBA" if kind == "RGBA" else "VALUE"
    for s, v in ((_sock(n.inputs, "Factor", "VALUE"), fac),
                 (_sock(n.inputs, "A", stype), a),
                 (_sock(n.inputs, "B", stype), b)):
        if isinstance(v, (int, float, tuple)):
            s.default_value = v
        else:
            nt.links.new(v, s)
    return _sock(n.outputs, "Result", stype)


def _iface(ng, name, in_out, stype, default=None, vmin=None, vmax=None):
    s = ng.interface.new_socket(name=name, in_out=in_out, socket_type=stype)
    if name == "Vector" and in_out == "INPUT":
        s.hide_value = True
    if default is not None:
        s.default_value = default
    if vmin is not None:
        s.min_value = vmin
    if vmax is not None:
        s.max_value = vmax
    return s


def _fresh_group(name):
    ng = bpy.data.node_groups.get(name)
    if ng and ng.get("fm_version") == GROUP_VERSION:
        return ng, False
    if ng is None:
        ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    else:
        ng.nodes.clear()
        ng.interface.clear()
    ng["fm_version"] = GROUP_VERSION
    return ng, True


# --------------------------------------------------------------------------
# FM_Surface: adjustments + procedural weathering
# --------------------------------------------------------------------------

SURFACE_INPUTS = [
    # name, socket type, default, min, max
    ("Base Color", "NodeSocketColor", (0.8, 0.8, 0.8, 1.0), None, None),
    ("Roughness", "NodeSocketFloat", 0.5, 0.0, 1.0),
    ("Metallic", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Height", "NodeSocketFloat", 0.5, 0.0, 1.0),
    ("Tint", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0), None, None),
    ("Brightness", "NodeSocketFloat", 1.0, 0.0, 4.0),
    ("Roughness Scale", "NodeSocketFloat", 1.0, 0.0, 4.0),
    ("Roughness Offset", "NodeSocketFloat", 0.0, -1.0, 1.0),
    ("Metallic Strength", "NodeSocketFloat", 1.0, 0.0, 1.0),
    ("Edge Wear", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Edge Width", "NodeSocketFloat", 0.006, 0.0, 0.1),
    ("Edge Color", "NodeSocketColor", (0.85, 0.85, 0.87, 1.0), None, None),
    ("Edge Roughness", "NodeSocketFloat", 0.2, 0.0, 1.0),
    ("Edge Metallic", "NodeSocketFloat", 1.0, 0.0, 1.0),
    ("Curvature Boost", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Dirt", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Dirt Spread", "NodeSocketFloat", 0.02, 0.001, 0.5),
    ("Dirt Color", "NodeSocketColor", (0.035, 0.026, 0.018, 1.0), None, None),
    ("Dirt Roughness", "NodeSocketFloat", 0.75, 0.0, 1.0),
    ("Rust", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Rust Scale", "NodeSocketFloat", 30.0, 0.1, 500.0),
    ("Rust Color", "NodeSocketColor", (0.22, 0.07, 0.025, 1.0), None, None),
    ("Breakup Scale", "NodeSocketFloat", 40.0, 0.1, 500.0),
    # weather layers (world-up aware), applied on top of everything above
    ("Paint", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Paint Wear", "NodeSocketFloat", 0.3, 0.0, 1.0),
    ("Paint Color", "NodeSocketColor", (0.12, 0.25, 0.35, 1.0), None, None),
    ("Moss", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Moss Scale", "NodeSocketFloat", 12.0, 0.5, 200.0),
    ("Moss Color", "NodeSocketColor", (0.10, 0.17, 0.03, 1.0), None, None),
    ("Wetness", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Puddles", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Snow", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Snow Color", "NodeSocketColor", (0.90, 0.92, 0.95, 1.0), None, None),
    ("Vector", "NodeSocketVector", (0.0, 0.0, 0.0), None, None),  # metres, from FM_Coords
]

# inputs the UI shows as user controls (the first four are wired from textures)
SURFACE_CONTROLS = [n for n, *_ in SURFACE_INPUTS[4:] if n != "Vector"]


def surface_group():
    ng, build = _fresh_group(SURFACE_GROUP)
    if not build:
        return ng
    for name, stype, d, lo, hi in SURFACE_INPUTS:
        _iface(ng, name, "INPUT", stype, d, lo, hi)
    for name, stype in (("Base Color", "NodeSocketColor"), ("Roughness", "NodeSocketFloat"),
                        ("Metallic", "NodeSocketFloat"), ("Height", "NodeSocketFloat"),
                        ("Edge Mask", "NodeSocketFloat"), ("Dirt Mask", "NodeSocketFloat"),
                        ("Rust Mask", "NodeSocketFloat")):
        _iface(ng, name, "OUTPUT", stype)

    nt = ng
    gi = nt.nodes.new("NodeGroupInput")
    gi.location = (-1400, 0)
    go = nt.nodes.new("NodeGroupOutput")
    go.location = (1400, 0)
    I = {s.name: s for s in gi.outputs if s.name}

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-1200, 500)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-1200, -500)

    # ---- edge mask: bevel normal vs true normal, convex only (AO) -------
    bev = nt.nodes.new("ShaderNodeBevel")
    bev.samples = 8
    bev.location = (-1000, 500)
    nt.links.new(I["Edge Width"], bev.inputs["Radius"])
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.location = (-800, 500)
    nt.links.new(bev.outputs["Normal"], dot.inputs[0])
    nt.links.new(geo.outputs["Normal"], dot.inputs[1])
    curv = _math(nt, "SUBTRACT", 1.0, _sock(dot.outputs, "Value"), loc=(-600, 500))
    curv = _maprange(nt, curv, 0.0, 0.06, loc=(-400, 500))

    # optional Cycles pointiness for dense / subdivided meshes whose rounded
    # bevels the Bevel node cannot see (0 = off; low-poly meshes should keep 0)
    pnt = _maprange(nt, geo.outputs["Pointiness"], 0.505, 0.55, loc=(-600, 900))
    pnt = _math(nt, "MULTIPLY", pnt, I["Curvature Boost"], loc=(-300, 850))
    curv = _math(nt, "MAXIMUM", curv, pnt, loc=(-250, 650))

    ao_edge = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao_edge.samples = 8
    ao_edge.location = (-1000, 250)
    edge_dist = _math(nt, "MULTIPLY", I["Edge Width"], 4.0, loc=(-1200, 250))
    nt.links.new(edge_dist, ao_edge.inputs["Distance"])
    convex = _maprange(nt, ao_edge.outputs["AO"], 0.75, 0.98, loc=(-800, 250))
    edge = _math(nt, "MULTIPLY", curv, convex, loc=(-200, 500))

    # breakup noise
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.location = (-1000, -300)
    nz.inputs["Detail"].default_value = 8.0
    nz.inputs["Roughness"].default_value = 0.6
    nt.links.new(I["Vector"], nz.inputs["Vector"])
    nt.links.new(I["Breakup Scale"], nz.inputs["Scale"])
    noise = nz.outputs["Fac"]

    # Edge Wear widens the band (lower curvature threshold) and fills in the
    # chips along it (noise threshold): low = sparse chips, high = solid worn edge
    noise = _maprange(nt, noise, 0.3, 0.7, smooth=False, loc=(-800, -300))
    inv_w = _math(nt, "SUBTRACT", 1.0, I["Edge Wear"], loc=(-200, 300))
    c_thr = _math(nt, "MULTIPLY", inv_w, 0.6, loc=(0, 300))
    c_hi = _math(nt, "ADD", c_thr, 0.15, loc=(200, 300))
    e_band = _maprange(nt, edge, c_thr, c_hi, loc=(400, 500))
    n_thr = _math(nt, "MULTIPLY_ADD", inv_w, 0.9, loc=(0, 150))
    n_thr.node.inputs[2].default_value = -0.05
    n_hi = _math(nt, "ADD", n_thr, 0.1, loc=(200, 150))
    chips = _maprange(nt, noise, n_thr, n_hi, loc=(400, 150))
    e_mask = _math(nt, "MULTIPLY", e_band, chips, loc=(500, 400))
    gate_e = _math(nt, "MULTIPLY", I["Edge Wear"], 25.0, clamp=True, loc=(400, 300))
    e_mask = _math(nt, "MULTIPLY", e_mask, gate_e, clamp=True, loc=(600, 500), label="Edge Mask")

    # ---- cavity dirt + surface grime ---------------------------------------
    ao_cav = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao_cav.samples = 8
    ao_cav.location = (-1000, 0)
    nt.links.new(I["Dirt Spread"], ao_cav.inputs["Distance"])
    cavity = _math(nt, "SUBTRACT", 1.0, ao_cav.outputs["AO"], loc=(-800, 0))
    gz = nt.nodes.new("ShaderNodeTexNoise")
    gz.location = (-1000, -150)
    gz.inputs["Detail"].default_value = 6.0
    gz.inputs["Roughness"].default_value = 0.55
    nt.links.new(I["Vector"], gz.inputs["Vector"])
    g_scale = _math(nt, "MULTIPLY", I["Breakup Scale"], 0.5, loc=(-1200, -150))
    nt.links.new(g_scale, gz.inputs["Scale"])
    # noise Fac clusters around 0.5; stretch it to ~0..1 so the slider is linear
    grime = _maprange(nt, gz.outputs["Fac"], 0.35, 0.65, smooth=False, loc=(-800, -150))
    # cavity grime (main) + light surface grime at half strength
    cav_n = _math(nt, "MULTIPLY_ADD", grime, 0.8, loc=(-600, 100))
    cav_n.node.inputs[2].default_value = 0.6
    cav_v = _math(nt, "MULTIPLY", cavity, cav_n, loc=(-400, 100))
    inv_d = _math(nt, "SUBTRACT", 1.0, I["Dirt"], loc=(-600, -150))
    cthr = _math(nt, "MULTIPLY", inv_d, 0.5, loc=(-400, -150))
    cthr_hi = _math(nt, "ADD", cthr, 0.2, loc=(-200, -150))
    cav_m = _maprange(nt, cav_v, cthr, cthr_hi, loc=(0, 100))
    gthr = _math(nt, "MULTIPLY_ADD", I["Dirt"], -0.6, loc=(-400, -300))
    gthr.node.inputs[2].default_value = 1.0
    gthr_hi = _math(nt, "ADD", gthr, 0.35, loc=(-200, -300))
    gm = _maprange(nt, grime, gthr, gthr_hi, loc=(0, -150))
    gm = _math(nt, "MULTIPLY", gm, 0.5, loc=(100, -150))
    d_mask = _math(nt, "MAXIMUM", cav_m, gm, loc=(200, 0))
    gate_d = _math(nt, "MULTIPLY", I["Dirt"], 25.0, clamp=True, loc=(200, -150))
    d_mask = _math(nt, "MULTIPLY", d_mask, gate_d, clamp=True, loc=(400, 0))
    d_mask = _math(nt, "MULTIPLY", d_mask, 0.9, loc=(500, 0), label="Dirt Mask")

    # ---- rust patches ----------------------------------------------------
    rz = nt.nodes.new("ShaderNodeTexNoise")
    rz.location = (-1000, -600)
    rz.inputs["Detail"].default_value = 10.0
    rz.inputs["Roughness"].default_value = 0.65
    rz.inputs["Distortion"].default_value = 0.4
    nt.links.new(I["Vector"], rz.inputs["Vector"])
    nt.links.new(I["Rust Scale"], rz.inputs["Scale"])
    rust_n = _maprange(nt, rz.outputs["Fac"], 0.4, 0.6, smooth=False, loc=(-800, -600))
    r_val = _math(nt, "MULTIPLY_ADD", cavity, 0.5, loc=(-600, -600))
    nt.links.new(rust_n, r_val.node.inputs[2])
    rthr = _math(nt, "MULTIPLY_ADD", I["Rust"], -0.9, loc=(-400, -750))
    rthr.node.inputs[2].default_value = 1.0
    rthr_hi = _math(nt, "ADD", rthr, 0.1, loc=(0, -750))
    r_mask = _maprange(nt, r_val, rthr, rthr_hi, loc=(200, -600))
    gate_r = _math(nt, "MULTIPLY", I["Rust"], 25.0, clamp=True, loc=(200, -750))
    r_mask = _math(nt, "MULTIPLY", r_mask, gate_r, clamp=True, loc=(400, -600), label="Rust Mask")
    # rust colour variation
    rust_dark = _mix(nt, "RGBA", rz.outputs["Fac"], (0.08, 0.03, 0.015, 1.0), I["Rust Color"],
                     loc=(400, -850))

    # ---- base adjustments --------------------------------------------------
    col = _mix(nt, "RGBA", 1.0, I["Base Color"], I["Tint"], loc=(-400, 900), blend="MULTIPLY")
    bright = nt.nodes.new("ShaderNodeVectorMath")
    bright.operation = "SCALE"
    bright.location = (-200, 900)
    nt.links.new(col, bright.inputs[0])
    nt.links.new(I["Brightness"], _sock(bright.inputs, "Scale"))
    col = bright.outputs["Vector"]

    rough = _math(nt, "MULTIPLY_ADD", I["Roughness"], I["Roughness Scale"], loc=(-400, 750))
    nt.links.new(I["Roughness Offset"], rough.node.inputs[2])
    rough.node.use_clamp = True
    metal = _math(nt, "MULTIPLY", I["Metallic"], I["Metallic Strength"], clamp=True,
                  loc=(-400, 650))

    # ---- layer stack: base -> rust -> dirt -> edge wear --------------------
    c1 = _mix(nt, "RGBA", r_mask, col, rust_dark, loc=(700, 900))
    c2 = _mix(nt, "RGBA", d_mask, c1, I["Dirt Color"], loc=(900, 900))
    c3 = _mix(nt, "RGBA", e_mask, c2, I["Edge Color"], loc=(1100, 900))

    r1 = _mix(nt, "FLOAT", r_mask, rough, 0.85, loc=(700, 700))
    r2 = _mix(nt, "FLOAT", d_mask, r1, I["Dirt Roughness"], loc=(900, 700))
    r3 = _mix(nt, "FLOAT", e_mask, r2, I["Edge Roughness"], loc=(1100, 700))

    m1 = _mix(nt, "FLOAT", r_mask, metal, 0.0, loc=(700, 550))
    m2 = _mix(nt, "FLOAT", d_mask, m1, 0.0, loc=(900, 550))
    m3 = _mix(nt, "FLOAT", e_mask, m2, I["Edge Metallic"], loc=(1100, 550))

    # rust sits slightly proud of the surface in the height channel
    h = _math(nt, "MULTIPLY_ADD", r_mask, 0.04, loc=(1100, 350))
    nt.links.new(I["Height"], h.node.inputs[2])

    # ---- weather -----------------------------------------------------------
    c, r, m, h = _weather(nt, I, geo, cavity, c3, r3, m3, h)

    O = {s.name: s for s in go.inputs if s.name}
    nt.links.new(c, O["Base Color"])
    nt.links.new(r, O["Roughness"])
    nt.links.new(m, O["Metallic"])
    nt.links.new(h, O["Height"])
    nt.links.new(e_mask, O["Edge Mask"])
    nt.links.new(d_mask, O["Dirt Mask"])
    nt.links.new(r_mask, O["Rust Mask"])
    return ng


def _gate(nt, amount, loc):
    return _math(nt, "MULTIPLY", amount, 25.0, clamp=True, loc=loc)


def _weather(nt, I, geo, cavity, col, rough, metal, height):
    """Paint -> moss -> wetness/puddles -> snow, each off at 0."""
    x0 = 1400
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nsep.location = (x0, -1000)
    nt.links.new(geo.outputs["Normal"], nsep.inputs[0])
    up = nsep.outputs[2]

    def noise(scale, loc, detail=8.0):
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.location = loc
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = 0.6
        nt.links.new(I["Vector"], n.inputs["Vector"])
        if isinstance(scale, (int, float)):
            n.inputs["Scale"].default_value = scale
        else:
            nt.links.new(scale, n.inputs["Scale"])
        return _maprange(nt, n.outputs["Fac"], 0.35, 0.65, smooth=False,
                         loc=(loc[0] + 150, loc[1]))

    # paint: a coat over everything, chipped by Paint Wear
    pn = noise(_math(nt, "MULTIPLY", I["Breakup Scale"], 0.6, loc=(x0, -1200)), (x0 + 150, -1200))
    chip_lo = _math(nt, "SUBTRACT", 1.0, I["Paint Wear"], loc=(x0 + 300, -1300))
    chip = _maprange(nt, pn, chip_lo, _math(nt, "ADD", chip_lo, 0.06, loc=(x0 + 450, -1300)),
                     loc=(x0 + 600, -1200))
    cover = _math(nt, "SUBTRACT", 1.0, chip, loc=(x0 + 750, -1200))
    cover = _math(nt, "MULTIPLY", cover, _gate(nt, I["Paint"], (x0 + 750, -1350)), loc=(x0 + 900, -1200))
    cover = _math(nt, "MULTIPLY", cover, _math(nt, "MINIMUM", I["Paint"], 1.0, loc=(x0 + 900, -1350)),
                  loc=(x0 + 1050, -1200))
    col = _mix(nt, "RGBA", cover, col, I["Paint Color"], loc=(x0 + 1200, -1100))
    rough = _mix(nt, "FLOAT", cover, rough, 0.45, loc=(x0 + 1200, -1250))
    metal = _mix(nt, "FLOAT", cover, metal, 0.0, loc=(x0 + 1200, -1400))
    h0 = height
    height = _math(nt, "MULTIPLY_ADD", cover, 0.03, loc=(x0 + 1200, -1550))
    nt.links.new(h0, height.node.inputs[2])

    # moss: upward faces and cavities, patchy
    mn = noise(I["Moss Scale"], (x0 + 150, -1700))
    upm = _maprange(nt, up, -0.2, 0.7, loc=(x0 + 300, -1850))
    mm = _math(nt, "MULTIPLY_ADD", cavity, 1.5, loc=(x0 + 450, -1850))
    nt.links.new(upm, mm.node.inputs[2])
    mv = _math(nt, "MULTIPLY", mn, mm, loc=(x0 + 600, -1750))
    m_lo = _math(nt, "SUBTRACT", 1.0, I["Moss"], loc=(x0 + 600, -1900))
    mmask = _maprange(nt, mv, m_lo, _math(nt, "ADD", m_lo, 0.2, loc=(x0 + 750, -1900)),
                      loc=(x0 + 900, -1750))
    mmask = _math(nt, "MULTIPLY", mmask, _gate(nt, I["Moss"], (x0 + 900, -1900)), loc=(x0 + 1050, -1750))
    mcol = _mix(nt, "RGBA", mn, _vec_scale(nt, I["Moss Color"], 0.6, (x0 + 1050, -1600)),
                I["Moss Color"], loc=(x0 + 1150, -1650))
    col = _mix(nt, "RGBA", mmask, col, mcol, loc=(x0 + 1300, -1100))
    rough = _mix(nt, "FLOAT", mmask, rough, 0.95, loc=(x0 + 1300, -1250))
    metal = _mix(nt, "FLOAT", mmask, metal, 0.0, loc=(x0 + 1300, -1400))
    h0 = height
    height = _math(nt, "MULTIPLY_ADD", mmask, 0.08, loc=(x0 + 1300, -1550))
    nt.links.new(h0, height.node.inputs[2])

    # wetness: darker (non-metals) and glossier; puddles pool on flat up-facing spots
    nonmetal = _math(nt, "SUBTRACT", 1.0, metal, loc=(x0 + 1450, -1000))
    dark = _math(nt, "MULTIPLY", I["Wetness"], 0.4, loc=(x0 + 1450, -1150))
    dark = _math(nt, "MULTIPLY", dark, nonmetal, loc=(x0 + 1600, -1150))
    dark = _math(nt, "SUBTRACT", 1.0, dark, loc=(x0 + 1750, -1150))
    col = _vec_scale(nt, col, dark, (x0 + 1900, -1100))
    wr = _math(nt, "MULTIPLY", I["Wetness"], 0.75, loc=(x0 + 1450, -1300))
    wr = _math(nt, "SUBTRACT", 1.0, wr, loc=(x0 + 1600, -1300))
    rough = _math(nt, "MULTIPLY", rough, wr, loc=(x0 + 1750, -1300))
    flat = _maprange(nt, up, 0.85, 0.97, loc=(x0 + 1450, -1500))
    pd = noise(_math(nt, "MULTIPLY", I["Breakup Scale"], 0.08, loc=(x0 + 1300, -1700)), (x0 + 1450, -1700))
    p_lo = _math(nt, "SUBTRACT", 1.0, I["Puddles"], loc=(x0 + 1600, -1800))
    pmask = _maprange(nt, pd, p_lo, _math(nt, "ADD", p_lo, 0.05, loc=(x0 + 1750, -1800)),
                      loc=(x0 + 1900, -1700))
    pmask = _math(nt, "MULTIPLY", pmask, flat, loc=(x0 + 2050, -1650))
    pmask = _math(nt, "MULTIPLY", pmask, _gate(nt, I["Puddles"], (x0 + 2050, -1800)), loc=(x0 + 2200, -1650))
    col = _mix(nt, "RGBA", pmask, col, _vec_scale(nt, col, 0.55, (x0 + 2200, -1500)), loc=(x0 + 2350, -1100))
    rough = _mix(nt, "FLOAT", pmask, rough, 0.02, loc=(x0 + 2350, -1300))
    height = _mix(nt, "FLOAT", pmask, height, 0.5, loc=(x0 + 2350, -1550))

    # snow: settles on upward faces, broken up by noise
    sn = noise(_math(nt, "MULTIPLY", I["Breakup Scale"], 0.25, loc=(x0 + 2200, -1900)), (x0 + 2350, -1900))
    ups = _maprange(nt, up, 0.15, 0.6, loc=(x0 + 2500, -2000))
    sv = _math(nt, "MULTIPLY_ADD", sn, 0.6, loc=(x0 + 2650, -1900))
    sv.node.inputs[2].default_value = 0.4
    sv = _math(nt, "MULTIPLY", sv, ups, loc=(x0 + 2800, -1900))
    s_lo = _math(nt, "SUBTRACT", 1.0, I["Snow"], loc=(x0 + 2800, -2050))
    smask = _maprange(nt, sv, s_lo, _math(nt, "ADD", s_lo, 0.12, loc=(x0 + 2950, -2050)),
                      loc=(x0 + 3100, -1900))
    smask = _math(nt, "MULTIPLY", smask, _gate(nt, I["Snow"], (x0 + 3100, -2050)), loc=(x0 + 3250, -1900))
    col = _mix(nt, "RGBA", smask, col, I["Snow Color"], loc=(x0 + 3400, -1100))
    rough = _mix(nt, "FLOAT", smask, rough, 0.65, loc=(x0 + 3400, -1300))
    metal = _mix(nt, "FLOAT", smask, metal, 0.0, loc=(x0 + 3400, -1450))
    h0 = height
    height = _math(nt, "MULTIPLY_ADD", smask, 0.25, loc=(x0 + 3400, -1600))
    nt.links.new(h0, height.node.inputs[2])
    return col, rough, metal, height


def _vec_scale(nt, vec, fac, loc):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = "SCALE"
    n.location = loc
    if isinstance(vec, (tuple, list)):
        n.inputs[0].default_value = vec[:3]
    else:
        nt.links.new(vec, n.inputs[0])
    s = _sock(n.inputs, "Scale")
    if isinstance(fac, (int, float)):
        s.default_value = fac
    else:
        nt.links.new(fac, s)
    return n.outputs["Vector"]


# --------------------------------------------------------------------------
# FM_ProcMetal: texture-free metals (brushed, forged, damascus)
# --------------------------------------------------------------------------

PROC_INPUTS = [
    ("Color A", "NodeSocketColor", (0.8, 0.8, 0.82, 1.0), None, None),
    ("Color B", "NodeSocketColor", (0.6, 0.6, 0.62, 1.0), None, None),
    ("Roughness", "NodeSocketFloat", 0.2, 0.0, 1.0),
    ("Roughness Variation", "NodeSocketFloat", 0.1, 0.0, 1.0),
    ("Scale", "NodeSocketFloat", 6.0, 0.01, 500.0),
    ("Brush Stretch", "NodeSocketVector", (40.0, 40.0, 1.0), 0.0, 1000.0),
    ("Damascus", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Damascus Scale", "NodeSocketFloat", 80.0, 0.1, 1000.0),
    ("Damascus Distortion", "NodeSocketFloat", 14.0, 0.0, 100.0),
    # squash along the blade's long axis so the pattern flows lengthwise
    # (default assumes the blade points along local Z; use (1, 0.25, 1) for Y)
    ("Pattern Stretch", "NodeSocketVector", (1.0, 1.0, 0.25), 0.0, 100.0),
    ("Hammered", "NodeSocketFloat", 0.0, 0.0, 1.0),
    ("Vector", "NodeSocketVector", (0.0, 0.0, 0.0), None, None),
]
PROC_CONTROLS = [n for n, *_ in PROC_INPUTS if n != "Vector"]


def proc_group():
    ng, build = _fresh_group(PROC_GROUP)
    if not build:
        return ng
    for name, stype, d, lo, hi in PROC_INPUTS:
        _iface(ng, name, "INPUT", stype, d, lo, hi)
    for name, stype in (("Base Color", "NodeSocketColor"), ("Roughness", "NodeSocketFloat"),
                        ("Height", "NodeSocketFloat")):
        _iface(ng, name, "OUTPUT", stype)
    nt = ng
    gi = nt.nodes.new("NodeGroupInput")
    gi.location = (-1200, 0)
    go = nt.nodes.new("NodeGroupOutput")
    go.location = (900, 0)
    I = {s.name: s for s in gi.outputs if s.name}
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-1200, 400)

    # brushed micro-variation: noise on stretched object coords
    st = nt.nodes.new("ShaderNodeVectorMath")
    st.operation = "MULTIPLY"
    st.location = (-1000, 400)
    nt.links.new(I["Vector"], st.inputs[0])
    nt.links.new(I["Brush Stretch"], st.inputs[1])
    bn = nt.nodes.new("ShaderNodeTexNoise")
    bn.location = (-800, 400)
    bn.inputs["Detail"].default_value = 6.0
    nt.links.new(st.outputs["Vector"], bn.inputs["Vector"])
    nt.links.new(I["Scale"], bn.inputs["Scale"])
    brushed = bn.outputs["Fac"]

    # broad cloudy variation
    cn = nt.nodes.new("ShaderNodeTexNoise")
    cn.location = (-800, 150)
    cn.inputs["Detail"].default_value = 4.0
    cn.inputs["Scale"].default_value = 3.0
    nt.links.new(I["Vector"], cn.inputs["Vector"])
    cloud = cn.outputs["Fac"]

    # damascus: distorted wave bands
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.wave_type = "BANDS"
    wv.bands_direction = "X"
    wv.wave_profile = "SIN"
    wv.location = (-800, -150)
    wv.inputs["Detail"].default_value = 6.0
    wv.inputs["Detail Scale"].default_value = 0.6
    wv.inputs["Detail Roughness"].default_value = 0.6
    ps = nt.nodes.new("ShaderNodeVectorMath")
    ps.operation = "MULTIPLY"
    ps.location = (-1000, -150)
    nt.links.new(I["Vector"], ps.inputs[0])
    nt.links.new(I["Pattern Stretch"], ps.inputs[1])
    nt.links.new(ps.outputs["Vector"], wv.inputs["Vector"])
    nt.links.new(I["Damascus Scale"], wv.inputs["Scale"])
    nt.links.new(I["Damascus Distortion"], wv.inputs["Distortion"])
    bands = _maprange(nt, wv.outputs["Fac"], 0.35, 0.65, loc=(-600, -150))

    # hammered: voronoi dimples
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.feature = "F1"
    vo.location = (-800, -450)
    vo.inputs["Scale"].default_value = 30.0
    nt.links.new(I["Vector"], vo.inputs["Vector"])
    dimple = _math(nt, "POWER", _sock(vo.outputs, "Distance"), 0.5, loc=(-600, -450))

    # colour: cloud mix of A/B, then damascus bands override
    base = _mix(nt, "RGBA", cloud, I["Color A"], I["Color B"], loc=(-400, 300))
    base2 = _mix(nt, "RGBA", brushed, base, I["Color A"], loc=(-200, 300))
    dam_col = _mix(nt, "RGBA", bands, I["Color B"], I["Color A"], loc=(-200, 0))
    col = _mix(nt, "RGBA", I["Damascus"], base2, dam_col, loc=(0, 300))

    # roughness
    rv = _math(nt, "SUBTRACT", brushed, 0.5, loc=(-400, 600))
    rv = _math(nt, "MULTIPLY", rv, I["Roughness Variation"], loc=(-200, 600))
    rough = _math(nt, "ADD", I["Roughness"], rv, clamp=True, loc=(0, 600))
    dam_r = _math(nt, "MULTIPLY_ADD", bands, -0.12, loc=(200, 600))
    nt.links.new(rough, dam_r.node.inputs[2])
    rough = _mix(nt, "FLOAT", I["Damascus"], rough, dam_r, loc=(400, 600))

    # height: bands + dimples
    hb = _math(nt, "MULTIPLY", bands, I["Damascus"], loc=(0, -150))
    hb = _math(nt, "MULTIPLY", hb, 0.15, loc=(200, -150))
    hd = _math(nt, "MULTIPLY", dimple, I["Hammered"], loc=(0, -450))
    hd = _math(nt, "MULTIPLY", hd, 0.3, loc=(200, -450))
    h = _math(nt, "ADD", hb, hd, loc=(400, -300))
    h = _math(nt, "ADD", h, 0.5, loc=(600, -300))

    O = {s.name: s for s in go.inputs if s.name}
    nt.links.new(col, O["Base Color"])
    nt.links.new(rough, O["Roughness"])
    nt.links.new(h, O["Height"])
    return ng


PROC_PRESETS = {
    "POLISHED_STEEL": ("Polished Steel", {
        "Color A": (0.80, 0.80, 0.82, 1), "Color B": (0.62, 0.63, 0.65, 1),
        "Roughness": 0.16, "Roughness Variation": 0.12}),
    "DARK_IRON": ("Dark Forged Iron", {
        "Color A": (0.27, 0.27, 0.28, 1), "Color B": (0.13, 0.13, 0.14, 1),
        "Roughness": 0.48, "Roughness Variation": 0.25, "Hammered": 0.6}),
    "DAMASCUS": ("Damascus Steel", {
        "Color A": (0.80, 0.80, 0.82, 1), "Color B": (0.30, 0.30, 0.32, 1),
        "Roughness": 0.28, "Roughness Variation": 0.08, "Damascus": 1.0}),
    "GOLD": ("Gold", {
        "Color A": (1.0, 0.77, 0.34, 1), "Color B": (0.86, 0.61, 0.24, 1),
        "Roughness": 0.22, "Roughness Variation": 0.1}),
    "SILVER": ("Silver", {
        "Color A": (0.97, 0.96, 0.92, 1), "Color B": (0.85, 0.84, 0.80, 1),
        "Roughness": 0.18, "Roughness Variation": 0.1}),
    "BRONZE": ("Bronze", {
        "Color A": (0.71, 0.43, 0.18, 1), "Color B": (0.55, 0.32, 0.13, 1),
        "Roughness": 0.35, "Roughness Variation": 0.15, "Hammered": 0.3}),
    "COPPER": ("Copper", {
        "Color A": (0.96, 0.64, 0.54, 1), "Color B": (0.80, 0.48, 0.38, 1),
        "Roughness": 0.3, "Roughness Variation": 0.12}),
}

# edge colour for metals: a lighter, cleaner version of the base
PROC_EDGE = {
    "POLISHED_STEEL": (0.92, 0.92, 0.93, 1), "DARK_IRON": (0.62, 0.62, 0.64, 1),
    "DAMASCUS": (0.88, 0.88, 0.90, 1), "GOLD": (1.0, 0.86, 0.55, 1),
    "SILVER": (1.0, 1.0, 0.98, 1), "BRONZE": (0.90, 0.66, 0.42, 1),
    "COPPER": (1.0, 0.78, 0.68, 1),
}


# --------------------------------------------------------------------------
# material construction
# --------------------------------------------------------------------------

def _new_material(name):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True  # no-op/deprecated on newer versions
    except (AttributeError, TypeError):
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    out.target = "ALL"
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (550, 0)
    bsdf.name = "FM_BSDF"
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf


def _height_node(nt, src):
    """Named passthrough so the baker can find the height signal."""
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = "ADD"
    n.inputs[1].default_value = 0.0
    n.name = "FM_HEIGHT"
    n.label = "Height (bake)"
    n.location = (550, -450)
    nt.links.new(src, n.inputs[0])
    return n.outputs[0]


def _surface_node(nt, loc=(250, 0)):
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = surface_group()
    g.name = "FM_Surface"
    g.label = "Forge Surface"
    g.location = loc
    g.width = 220
    return g


def _load_image(path, data):
    img = bpy.data.images.load(path, check_existing=True)
    if data:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            img.colorspace_settings.is_data = True
    return img


# --------------------------------------------------------------------------
# coordinates: real-world metres, object-scale independent, long-axis aware
# --------------------------------------------------------------------------

COORDS_GROUP = "FM_Coords"


def coords_group():
    """Object (m): object-space position scaled to metres (object scale undone), with
    the axes permuted so the object's long axis (object property fm_axis: 0=Z, 1=X,
    2=Y) becomes Z - grain, planks and brushing then follow the object.
    UV (m): UVs times the object's UV density (property fm_uv_density, m per UV unit)."""
    ng, build = _fresh_group(COORDS_GROUP)
    if not build:
        return ng
    _iface(ng, "Object (m)", "OUTPUT", "NodeSocketVector")
    _iface(ng, "UV (m)", "OUTPUT", "NodeSocketVector")
    _iface(ng, "Object raw (m)", "OUTPUT", "NodeSocketVector")
    _iface(ng, "Normal", "OUTPUT", "NodeSocketVector")
    _iface(ng, "Normal (axis)", "OUTPUT", "NodeSocketVector")
    nt = ng
    go = nt.nodes.new("NodeGroupOutput")
    go.location = (1000, 0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-900, 0)

    # object scale = length of each local unit axis transformed to world space
    comps = []
    for i, axis in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        vt = nt.nodes.new("ShaderNodeVectorTransform")
        vt.vector_type = "VECTOR"
        vt.convert_from = "OBJECT"
        vt.convert_to = "WORLD"
        vt.inputs["Vector"].default_value = axis
        vt.location = (-700, 300 - i * 150)
        ln = nt.nodes.new("ShaderNodeVectorMath")
        ln.operation = "LENGTH"
        ln.location = (-500, 300 - i * 150)
        nt.links.new(vt.outputs["Vector"], ln.inputs[0])
        comps.append(_sock(ln.outputs, "Value"))
    sc = nt.nodes.new("ShaderNodeCombineXYZ")
    sc.location = (-300, 200)
    for i, c in enumerate(comps):
        nt.links.new(c, sc.inputs[i])
    pm = nt.nodes.new("ShaderNodeVectorMath")
    pm.operation = "MULTIPLY"
    pm.location = (-100, 0)
    nt.links.new(tc.outputs["Object"], pm.inputs[0])
    nt.links.new(sc.outputs["Vector"], pm.inputs[1])

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (100, 0)
    nt.links.new(pm.outputs["Vector"], sep.inputs[0])
    X, Y, Z = sep.outputs[0], sep.outputs[1], sep.outputs[2]

    def comb(a, b, c, y):
        n = nt.nodes.new("ShaderNodeCombineXYZ")
        n.location = (300, y)
        for i, src in enumerate((a, b, c)):
            nt.links.new(src, n.inputs[i])
        return n.outputs["Vector"]
    vz = comb(X, Y, Z, 200)      # long axis Z (default)
    vx = comb(Y, Z, X, 0)        # long axis X -> becomes Z
    vy = comb(Z, X, Y, -200)     # long axis Y -> becomes Z

    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "OBJECT"
    at.attribute_name = "fm_axis"
    at.location = (100, -400)
    is_x = _math(nt, "COMPARE", at.outputs["Fac"], 1.0, loc=(300, -400))
    is_x.node.inputs[2].default_value = 0.5
    is_y = _math(nt, "COMPARE", at.outputs["Fac"], 2.0, loc=(300, -550))
    is_y.node.inputs[2].default_value = 0.5

    def vmix(fac, a, b, y):
        n = nt.nodes.new("ShaderNodeMix")
        n.data_type = "VECTOR"
        n.location = (550, y)
        nt.links.new(fac, _sock(n.inputs, "Factor", "VALUE"))
        nt.links.new(a, _sock(n.inputs, "A", "VECTOR"))
        nt.links.new(b, _sock(n.inputs, "B", "VECTOR"))
        return _sock(n.outputs, "Result", "VECTOR")
    v = vmix(is_x, vz, vx, 100)
    v = vmix(is_y, v, vy, -50)
    nt.links.new(v, go.inputs["Object (m)"])
    nt.links.new(pm.outputs["Vector"], go.inputs["Object raw (m)"])
    nt.links.new(tc.outputs["Normal"], go.inputs["Normal"])
    # the same axis permutation applied to the object-space normal
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nsep.location = (100, -250)
    nt.links.new(tc.outputs["Normal"], nsep.inputs[0])
    NX, NY, NZ = nsep.outputs[0], nsep.outputs[1], nsep.outputs[2]
    nz_ = comb(NX, NY, NZ, -900)
    nx_ = comb(NY, NZ, NX, -1050)
    ny_ = comb(NZ, NX, NY, -1200)
    nv = vmix(is_x, nz_, nx_, -950)
    nv = vmix(is_y, nv, ny_, -1100)
    nt.links.new(nv, go.inputs["Normal (axis)"])

    # UV in metres (density 0 / missing -> 1, i.e. plain UVs)
    ad = nt.nodes.new("ShaderNodeAttribute")
    ad.attribute_type = "OBJECT"
    ad.attribute_name = "fm_uv_density"
    ad.location = (100, -750)
    missing = _math(nt, "COMPARE", ad.outputs["Fac"], 0.0, loc=(300, -750))
    missing.node.inputs[2].default_value = 1e-6
    dens = _math(nt, "ADD", ad.outputs["Fac"], missing, loc=(450, -750))
    uvs = nt.nodes.new("ShaderNodeVectorMath")
    uvs.operation = "SCALE"
    uvs.location = (650, -700)
    nt.links.new(tc.outputs["UV"], uvs.inputs[0])
    nt.links.new(dens, _sock(uvs.inputs, "Scale"))
    nt.links.new(uvs.outputs["Vector"], go.inputs["UV (m)"])
    return ng


def _coords_node(nt, loc):
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = coords_group()
    g.name = "FM_Coords"
    g.label = "Real-World Coordinates"
    g.location = loc
    return g


def _sized_mapping(nt, coords, tile_size, size_mult):
    """coords (m) / (tile size * user size) -> Mapping (rotation). Returns the Mapping node."""
    ts = nt.nodes.new("ShaderNodeValue")
    ts.name = "FM_TileSize"
    ts.label = "Tile Size (m)"
    ts.outputs[0].default_value = tile_size
    ts.location = (-1700, -250)
    sz = nt.nodes.new("ShaderNodeValue")
    sz.name = "FM_Size"
    sz.label = "Size ×"
    sz.outputs[0].default_value = size_mult
    sz.location = (-1700, -350)
    den = _math(nt, "MULTIPLY", ts.outputs[0], sz.outputs[0], loc=(-1500, -300))
    inv = _math(nt, "DIVIDE", 1.0, den, loc=(-1350, -300))
    scl = nt.nodes.new("ShaderNodeVectorMath")
    scl.operation = "SCALE"
    scl.location = (-1300, 0)
    nt.links.new(coords, scl.inputs[0])
    nt.links.new(inv, _sock(scl.inputs, "Scale"))
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.name = "FM_Mapping"
    mp.label = "Rotation"
    mp.location = (-1100, 0)
    nt.links.new(scl.outputs["Vector"], mp.inputs["Vector"])
    return mp


def prepare_object(obj):
    """Store read-only measurements the shaders use: long axis and UV density.

    Only two custom properties are written on the object; the mesh, its UVs and
    its transform are never changed.
    """
    if obj is None or obj.type != "MESH":
        return
    d = obj.dimensions
    dims = (d.x, d.y, d.z)
    longest = max(range(3), key=lambda i: dims[i])
    # 0 = Z (default), 1 = X, 2 = Y; only switch when clearly elongated
    if dims[longest] > 1.3 * sorted(dims)[1]:
        obj["fm_axis"] = {2: 0, 0: 1, 1: 2}[longest]
    else:
        obj["fm_axis"] = 0
    me = obj.data
    uv = me.uv_layers.active
    if uv is None or not me.polygons:
        obj["fm_uv_density"] = 0.0
        return
    try:
        import numpy as np
        n = len(me.polygons)
        areas = np.empty(n, np.float64)
        me.polygons.foreach_get("area", areas)
        s = obj.scale
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
        nxt[starts + counts - 1] = starts          # wrap to each face's first loop
        x, y = co[:, 0], co[:, 1]
        cross = x * y[nxt] - y * x[nxt]           # shoelace terms
        uv_area = float(np.abs(np.add.reduceat(cross, starts)).sum() * 0.5)
        obj["fm_uv_density"] = float((area3d / uv_area) ** 0.5) if uv_area > 1e-12 else 0.0
    except Exception:
        obj["fm_uv_density"] = 0.0


def build_texture_material(item, mapping="UV", scale=1.0, metallic=None):
    """Create a material from a scanned library item.

    mapping: 'UV' uses the mesh UVs (best for baking); 'BOX' uses object-space
    triplanar projection (no UVs needed; normal detail comes from the height map).
    """
    maps = item["maps"]
    mat, nt, bsdf = _new_material("FM_" + item["name"])
    mat["fm_source"] = item["name"]
    mat["fm_kind"] = "texture"

    crd = _coords_node(nt, (-1700, 0))
    mp = _sized_mapping(nt, crd.outputs["Object (m)" if mapping == "BOX" else "UV (m)"],
                        item.get("size", 2.0), scale)
    mat["fm_mapping"] = mapping

    def tex(kind, y, data):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = _load_image(maps[kind], data)
        n.location = (-800, y)
        n.label = kind
        n.name = "FM_TEX_" + kind
        if mapping == "BOX":
            n.projection = "BOX"
            n.projection_blend = 0.25
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    surf = _surface_node(nt)
    S = {s.name: s for s in surf.inputs}
    nt.links.new(crd.outputs["Object (m)"], S["Vector"])

    col = tex("color", 400, False)
    color_out = col.outputs["Color"]
    if "ao" in maps:
        ao = tex("ao", 700, True)
        aom = _mix(nt, "RGBA", 1.0, color_out, ao.outputs["Color"], loc=(-500, 600),
                   blend="MULTIPLY")
        color_out = aom
    nt.links.new(color_out, S["Base Color"])

    if "arm" in maps:
        arm = tex("arm", 100, True)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-500, 100)
        nt.links.new(arm.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs[1], S["Roughness"])
        nt.links.new(sep.outputs[2], S["Metallic"])
    else:
        if "rough" in maps:
            r = tex("rough", 100, True)
            nt.links.new(r.outputs["Color"], S["Roughness"])
        if "metal" in maps:
            m = tex("metal", -150, True)
            nt.links.new(m.outputs["Color"], S["Metallic"])
        else:
            if metallic is None:
                metallic = 1.0 if item.get("category") == "METAL" else 0.0
            S["Metallic"].default_value = metallic

    height_src = None
    if "height" in maps:
        h = tex("height", -400, True)
        nt.links.new(h.outputs["Color"], S["Height"])
        height_src = surf.outputs["Height"]

    # normal: tangent normal map in UV mode, bump from height in BOX mode
    normal_in = None
    if mapping == "UV" and "normal_gl" in maps:
        nm_tex = tex("normal_gl", -700, True)
        src = nm_tex.outputs["Color"]
        if maps.get("normal_is_dx"):
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            sep.location = (-500, -700)
            nt.links.new(src, sep.inputs["Color"])
            inv = _math(nt, "SUBTRACT", 1.0, sep.outputs[1], loc=(-350, -700))
            comb = nt.nodes.new("ShaderNodeCombineColor")
            comb.location = (-200, -700)
            nt.links.new(sep.outputs[0], comb.inputs[0])
            nt.links.new(inv, comb.inputs[1])
            nt.links.new(sep.outputs[2], comb.inputs[2])
            src = comb.outputs["Color"]
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nmap.name = "FM_NormalMap"
        nmap.space = "TANGENT"
        nmap.location = (0, -650)
        nt.links.new(src, nmap.inputs["Color"])
        normal_in = nmap.outputs["Normal"]

    bump = nt.nodes.new("ShaderNodeBump")
    bump.name = "FM_Bump"
    bump.label = "Detail Bump"
    bump.location = (300, -400)
    bump.inputs["Strength"].default_value = 0.0 if normal_in is not None else 0.6
    bump.inputs["Distance"].default_value = 0.005
    if height_src is not None:
        nt.links.new(height_src, bump.inputs["Height"])
    if normal_in is not None:
        nt.links.new(normal_in, bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(surf.outputs["Base Color"], bsdf.inputs["Base Color"])
    nt.links.new(surf.outputs["Roughness"], bsdf.inputs["Roughness"])
    nt.links.new(surf.outputs["Metallic"], bsdf.inputs["Metallic"])
    _height_node(nt, surf.outputs["Height"])

    _apply_category_defaults(surf, item.get("category"))
    return mat


def _apply_category_defaults(surf, category):
    S = {s.name: s for s in surf.inputs}
    if category == "METAL":
        S["Edge Color"].default_value = (0.82, 0.82, 0.84, 1)
        S["Edge Metallic"].default_value = 1.0
        S["Edge Roughness"].default_value = 0.25
    else:
        # worn leather / wood / fabric edges get lighter and smoother, not metallic
        S["Edge Color"].default_value = (0.55, 0.45, 0.35, 1)
        S["Edge Metallic"].default_value = 0.0
        S["Edge Roughness"].default_value = 0.45


def build_procedural_material(preset):
    from . import procedural
    return procedural.build(preset)


def assign_material(obj, mat, all_slots=False):
    """Put mat on the object's active slot (or all slots).

    The replaced material is kept with a fake user so it is never lost on save.
    Mesh data is not modified except for adding a slot when there is none.
    """
    kept = []
    if not obj.material_slots:
        obj.data.materials.append(mat)
        return kept
    slots = obj.material_slots if all_slots else [obj.material_slots[obj.active_material_index]]
    for slot in slots:
        old = slot.material
        if old and old != mat and not old.use_fake_user:
            old.use_fake_user = True
            kept.append(old.name)
        slot.material = mat
    return kept


def find_surface(mat):
    if not mat or not mat.node_tree:
        return None
    return mat.node_tree.nodes.get("FM_Surface")
