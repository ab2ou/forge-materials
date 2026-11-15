# SPDX-License-Identifier: GPL-3.0-or-later
"""Saved looks (a material source + every knob value) and favourites.

Stored as small JSON files in the add-on's user folder, so they are shared by
every .blend file and survive add-on updates.
"""

import json
import os
import re
import time

import bpy

from . import library, nodes, procedural


def _base_dir():
    from .ui import cache_path
    return os.path.dirname(cache_path())


def _looks_dir():
    d = os.path.join(_base_dir(), "looks")
    os.makedirs(d, exist_ok=True)
    return d


def _val(v):
    try:
        return [float(x) for x in v]
    except TypeError:
        return float(v)


# --------------------------------------------------------------------------
# looks
# --------------------------------------------------------------------------

def capture(mat):
    nt = mat.node_tree
    data = {"version": 1, "kind": mat.get("fm_kind", ""), "source": mat.get("fm_source", ""),
            "mapping": mat.get("fm_mapping", "UV"), "hue": mat.get("fm_hue"), "sat": mat.get("fm_sat"),
            "surface": {}, "proc": {}}
    s = nt.nodes.get("FM_Surface")
    if s:
        for name in nodes.SURFACE_CONTROLS:
            sock = s.inputs.get(name)
            if sock is not None and not sock.is_linked:
                data["surface"][name] = _val(sock.default_value)
    p = nt.nodes.get("FM_Proc")
    if p:
        for sock in p.inputs:
            if sock.name not in ("Vector", "Normal") and not sock.is_linked:
                data["proc"][sock.name] = _val(sock.default_value)
    n = nt.nodes.get("FM_Size")
    if n:
        data["size"] = float(n.outputs[0].default_value)
    n = nt.nodes.get("FM_Mapping")
    if n:
        data["rotation"] = float(n.inputs["Rotation"].default_value[2])
    n = nt.nodes.get("FM_Bump")
    if n:
        data["bump"] = float(n.inputs["Strength"].default_value)
    n = nt.nodes.get("FM_NormalMap")
    if n:
        data["normal"] = float(n.inputs["Strength"].default_value)
    return data


def save(mat, name=None):
    data = capture(mat)
    if not name:
        base = mat.get("fm_source", "look")
        label = procedural.PRESETS.get(base, {}).get("label") or \
            (library.get(base) or {}).get("label") or base
        name = f"{label} {time.strftime('%m-%d %H%M')}"
    data["name"] = name
    fn = re.sub(r"[^A-Za-z0-9_\- ]+", "_", name).strip() or "look"
    path = os.path.join(_looks_dir(), fn + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return path, name


def list_looks():
    out = []
    d = _looks_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        data["file"] = os.path.join(d, fn)
        out.append(data)
    return out


def delete(path):
    if path and os.path.dirname(os.path.abspath(path)) == os.path.abspath(_looks_dir()):
        os.remove(path)


def look_items():
    """Looks as dial/list items (thumbnail of the source material)."""
    items = []
    for lk in list_looks():
        src = lk.get("source", "")
        it = {"name": "look:" + os.path.basename(lk["file"]), "label": lk.get("name", src),
              "look": lk, "category": "LOOKS"}
        if lk.get("kind") == "procedural" and src in procedural.PRESETS:
            pi = procedural.preset_item(src)
            it.update(proc=True, color=pi["color"])
            for k in ("thumb", "thumb_key", "thumb_direct"):
                if k in pi:
                    it[k] = pi[k]
        else:
            lib = library.get(src)
            if lib:
                it["thumb"] = lib["thumb"]
                it["thumb_key"] = src
        items.append(it)
    return items


def build(look, mapping_override=None, size_default=1.0):
    """Create a material from a saved look. Returns None if the source is missing."""
    src = look.get("source", "")
    if look.get("kind") == "procedural":
        if src not in procedural.PRESETS:
            return None
        mat = procedural.build(src)
    else:
        item = library.get(src)
        if item is None:
            return None
        mat = nodes.build_texture_material(item, mapping_override or look.get("mapping", "UV"),
                                           size_default)
    nt = mat.node_tree
    s = nt.nodes.get("FM_Surface")
    for name, v in look.get("surface", {}).items():
        sock = s.inputs.get(name) if s else None
        if sock is not None and not sock.is_linked:
            sock.default_value = v
    p = nt.nodes.get("FM_Proc")
    for name, v in look.get("proc", {}).items():
        sock = p.inputs.get(name) if p else None
        if sock is not None and not sock.is_linked:
            sock.default_value = v
    if "size" in look and nt.nodes.get("FM_Size"):
        nt.nodes["FM_Size"].outputs[0].default_value = look["size"]
    if "rotation" in look and nt.nodes.get("FM_Mapping"):
        nt.nodes["FM_Mapping"].inputs["Rotation"].default_value = (0, 0, look["rotation"])
    if "bump" in look and nt.nodes.get("FM_Bump"):
        nt.nodes["FM_Bump"].inputs["Strength"].default_value = look["bump"]
    if "normal" in look and nt.nodes.get("FM_NormalMap"):
        nt.nodes["FM_NormalMap"].inputs["Strength"].default_value = look["normal"]
    for k in ("hue", "sat"):
        if look.get(k) is not None:
            mat["fm_" + k] = look[k]
    mat["fm_look"] = look.get("name", "")
    return mat


# --------------------------------------------------------------------------
# favourites
# --------------------------------------------------------------------------

_FAV = {"set": None}


def _fav_path():
    return os.path.join(_base_dir(), "favourites.json")


def favourites():
    if _FAV["set"] is None:
        try:
            with open(_fav_path(), "r", encoding="utf-8") as f:
                _FAV["set"] = set(json.load(f))
        except (OSError, ValueError):
            _FAV["set"] = set()
    return _FAV["set"]


def toggle_favourite(name):
    favs = favourites()
    if name in favs:
        favs.remove(name)
    else:
        favs.add(name)
    try:
        os.makedirs(_base_dir(), exist_ok=True)
        with open(_fav_path(), "w", encoding="utf-8") as f:
            json.dump(sorted(favs), f)
    except OSError:
        pass
    return name in favs


def favourite_items():
    favs = favourites()
    out = [procedural.preset_item(k) for k in procedural.PRESETS if k in favs]
    out += [it for it in library.items() if it["name"] in favs]
    return out
