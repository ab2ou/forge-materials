# SPDX-License-Identifier: GPL-3.0-or-later
"""Texture library scanner.

Understands the Poly Haven folder layout used in the user's library
(<name>/<name>_2k/textures/<name>_diff_2k.jpg ...) and also any loose
folder of PBR images whose file names carry the usual map tokens.
Read-only: never writes into the library.
"""

import json
import os
import re

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff", ".tga", ".webp"}

# token -> map type. Order matters (nor_gl before nor).
MAP_TOKENS = [
    ("nor_gl", "normal_gl"), ("normal_gl", "normal_gl"), ("nor_dx", "normal_dx"),
    ("normal_dx", "normal_dx"), ("normal", "normal_gl"), ("nor", "normal_gl"),
    ("albedo", "color"), ("basecolor", "color"), ("base_color", "color"),
    ("diffuse", "color"), ("diff", "color"), ("color", "color"), ("col", "color"),
    ("roughness", "rough"), ("rough", "rough"),
    ("metallic", "metal"), ("metalness", "metal"), ("metal", "metal"),
    ("displacement", "height"), ("height", "height"), ("disp", "height"),
    ("arm", "arm"), ("ao", "ao"), ("ambientocclusion", "ao"),
]
SKIP_TOKENS = ("thumb", "primary", "renders", "clay", "preview", "spec", "bump", "mask")

RES_RE = re.compile(r"(?:^|[_\-\.])(\d{1,2})k(?:$|[_\-\.])", re.I)

# key, label, real-world size of one texture tile in metres, keywords, exclusions.
# Checked in order; the first match wins.
CATEGORIES = [
    ("LEATHER", "Leather", 0.5, ("leather", "suede"), ()),
    ("FABRIC", "Fabric", 0.6, ("fabric", "denim", "denmin", "linen", "cotton", "jersey", "hessian",
                               "wool", "velvet", "velour", "fleece", "satin", "jacquard", "poplin",
                               "corduroy", "terry", "georgette", "gingham", "boucle", "melange",
                               "stretch", "faux_fur", "teddy", "caban", "terlenka", "pique",
                               "carpet", "tatami", "book_pattern"), ()),
    ("SNOW", "Snow & Ice", 3.0, ("snow", "ice_"), ()),
    ("ROOFING", "Roofing", 2.5, ("roof", "thatch", "riet", "asbestos_sheet"), ()),
    ("WOOD", "Wood", 2.0, ("wood", "veneer", "plank", "bark", "parquet", "plywood", "strand_board",
                           "laminate", "timber", "pine", "oak", "walnut", "cherry", "maple", "teak",
                           "rosewood", "sapele", "okoume", "hinoki", "bamboo", "cedar", "sycamore",
                           "zelkova", "metasequoia", "eucalyptus", "beam_wall", "barrel", "door",
                           "gate"),
     ("textured_concrete", "wood_chip", "aerial_wood")),
    ("METAL", "Metal", 1.5, ("metal", "rusty", "rusted", "rust_coarse", "iron", "corrugated", "grate",
                             "container", "shutter", "steel", "bronze", "copper", "brass"),
     ("rustic",)),
    ("TILES", "Tiles & Floors", 2.0, ("tile", "mosaic", "klinker", "linoleum", "floor_pattern",
                                       "square_floor", "terrazzo", "checkered", "tiling", "terracotta", "clay_floor",
                                       "square_pattern"), ()),
    ("BRICK", "Brick", 2.5, ("brick",), ()),
    ("CONCRETE", "Concrete", 3.0, ("concrete", "cement", "pavers", "paving", "pavement", "precast",
                                   "preconcrete", "rebar", "slab", "garage_floor", "hangar",
                                   "driveway", "anti_slip"), ()),
    ("ASPHALT", "Asphalt & Roads", 4.0, ("asphalt", "road", "bitumen", "tarred", "running_track",
                                         "rubberized_track", "pit_lane"), ()),
    ("PLASTER", "Plaster & Walls", 2.5, ("plaster", "stucco", "wallpaper", "wall_cladding",
                                         "beige_wall", "ceiling", "painted_wall", "factory_wall",
                                         "broken_wall", "clay_block", "clay_wall", "whitewash"), ()),
    ("STONE", "Stone & Marble", 2.5, ("marble", "granite", "slate", "sandstone", "cobblestone", "stone",
                                      "medieval", "castle", "quarry", "coral_fort", "rubble",
                                      "cartago", "block_wall", "monastery", "rabdentse", "defense_wall",
                                      "tiger_rock", "volcanic", "shell_floor", "lichen"), ()),
    ("ROCK", "Rock", 4.0, ("rock", "cliff", "boulder", "seaside", "low_tide"), ()),
    ("GROUND", "Ground & Nature", 3.0, ("dirt", "mud", "soil", "ground", "sand", "gravel", "grass",
                                         "leaves", "leafy", "forest", "forrest", "pebble", "beach",
                                         "terrain", "aerial", "moon", "flour", "roots", "wood_chip",
                                         "mulch", "furrows", "trail", "path", "playground",
                                         "excavated", "burned", "moss"), ()),
]
CATEGORY_LABELS = {k: lbl for k, lbl, *_ in CATEGORIES}
CATEGORY_LABELS["OTHER"] = "Other"
CATEGORY_SIZE = {k: size for k, _, size, *_ in CATEGORIES}

# a few name hints override the category's tile size (metres)
SIZE_HINTS = (("aerial", 20.0), ("veneer", 1.0), ("macro", 0.3), ("moon_flat", 1.0))


def _matches(low, tokens, word):
    """Multi-part keywords ("faux_fur") match as substrings; single words must
    start a name token, so "gate" does not match "corrugated"."""
    if "_" in word:
        return word in low
    return any(t.startswith(word) for t in tokens)


def categorize(name):
    low = name.lower()
    tokens = [t for t in re.split(r"[_\-\s]+", low) if t]
    for key, _label, _size, words, excl in CATEGORIES:
        if any(_matches(low, tokens, e) for e in excl):
            continue
        if any(_matches(low, tokens, w) for w in words):
            return key
    return "OTHER"


def real_size(name, category):
    low = name.lower()
    for hint, size in SIZE_HINTS:
        if hint in low:
            return size
    return CATEGORY_SIZE.get(category, 2.0)


CACHE_VERSION = 4
_LIBRARY = {"root": "", "items": [], "version": CACHE_VERSION}


def _map_type(filename, mat_name):
    base = os.path.splitext(filename)[0].lower()
    stem = base
    if mat_name and stem.startswith(mat_name.lower()):
        stem = stem[len(mat_name):]
    stem = RES_RE.sub("_", stem).strip("_-. ")
    parts = [p for p in re.split(r"[_\-\. ]+", stem) if p]
    joined = "_".join(parts)
    if any(s in parts for s in SKIP_TOKENS):
        return None
    for tok, kind in MAP_TOKENS:
        if "_" in tok:
            if tok in joined:
                return kind
        elif tok in parts:
            return kind
    # numbered colour variants: "col1", "coll1", "color2", "col_1"
    for p in parts:
        if re.fullmatch(r"(col+|colou?r|diff|albedo)\d+", p):
            return "color"
    return None


def _res_value(filename):
    m = RES_RE.search(os.path.basename(filename))
    return int(m.group(1)) if m else 0


def _scan_folder(folder, name, preferred):
    """Collect maps under one material folder (max depth 3)."""
    found = {}
    thumb = ""
    base_depth = folder.rstrip("\\/").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(folder):
        if dirpath.count(os.sep) - base_depth >= 3:
            dirnames[:] = []
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMAGE_EXT:
                continue
            low = fn.lower()
            full = os.path.join(dirpath, fn)
            if "thumb" in low and not thumb:
                thumb = full
                continue
            kind = _map_type(fn, name)
            if not kind:
                continue
            found.setdefault(kind, []).append(full)

    maps = {}
    for kind, paths in found.items():
        def score(p):
            r = _res_value(p)
            pref = 1 if (preferred and r == preferred) else 0
            # prefer exact preferred res, then highest res, then exr/png over jpg
            ext_rank = {".exr": 2, ".png": 1}.get(os.path.splitext(p)[1].lower(), 0)
            return (pref, r, ext_rank)
        maps[kind] = sorted(paths, key=score)[-1]

    # an OpenGL normal is preferred; keep DX only as fallback
    if "normal_gl" not in maps and "normal_dx" in maps:
        maps["normal_gl"] = maps["normal_dx"]
        maps["normal_is_dx"] = "1"
    maps.pop("normal_dx", None)

    if "color" not in maps:
        return None
    if not thumb:
        for fn in os.listdir(folder):
            if "primary" in fn.lower() and os.path.splitext(fn)[1].lower() in IMAGE_EXT:
                thumb = os.path.join(folder, fn)
                break
    return {
        "name": name,
        "label": name.replace("_", " ").title(),
        "category": categorize(name),
        "size": real_size(name, categorize(name)),
        "folder": folder,
        "thumb": thumb or maps["color"],
        "maps": maps,
    }


def scan(root, preferred_res="HIGHEST"):
    preferred = 0 if preferred_res == "HIGHEST" else int(preferred_res.rstrip("kK"))
    items = []
    if not root or not os.path.isdir(root):
        _LIBRARY.update(root=root, items=[])
        return []
    for entry in sorted(os.scandir(root), key=lambda e: e.name.lower()):
        if not entry.is_dir():
            continue
        try:
            item = _scan_folder(entry.path, entry.name, preferred)
        except OSError:
            item = None
        if item:
            items.append(item)
    _LIBRARY.update(root=root, items=items, version=CACHE_VERSION)
    return items


def items():
    return _LIBRARY["items"]


def get(name):
    for it in _LIBRARY["items"]:
        if it["name"] == name:
            return it
    return None


def save_cache(path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_LIBRARY, f)
    except OSError:
        pass


def load_cache(path, root):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    if data.get("root") != root or data.get("version") != CACHE_VERSION:
        return False
    _LIBRARY.update(data)
    return True
