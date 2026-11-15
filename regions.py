# SPDX-License-Identifier: GPL-3.0-or-later
"""Smart Regions: split an object into material regions.

* Auto proposal: split by shape (long objects: by width changes along the
  length, e.g. blade / guard / grip) or at concave creases (any object).
* Interactive: click a part and the region grows over the surface until it
  meets a crease; Alt+click marks what is NOT part of it; the wheel sets how far
  it may grow. Shift+click merges, Ctrl+click splits, L draws a lasso.

The result is stored only as material slots + each face's material index, so
vertices, faces, UVs and normals are never touched. Every region gets a copy
of the material that was there, so the object looks exactly the same until a
new material is put on a region (the dial edits the active slot; Tab cycles).
"""

import math
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

from . import segment

PALETTE = [
    (0.95, 0.45, 0.25), (0.25, 0.62, 0.95), (0.95, 0.82, 0.25), (0.40, 0.85, 0.45),
    (0.80, 0.45, 0.95), (0.25, 0.88, 0.85), (0.95, 0.40, 0.65), (0.65, 0.80, 0.30),
    (0.55, 0.55, 0.95), (0.95, 0.62, 0.40), (0.35, 0.70, 0.60), (0.85, 0.85, 0.85),
]
_CACHE = {}
_STATE = {"kill": False}


def region_color(i):
    return PALETTE[i % len(PALETTE)]


# ------------------------------------------------------------------ analysis
def _fingerprint(mesh):
    n = len(mesh.vertices)
    co = np.empty(n * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    step = max(1, len(co) // 4096)
    return (n, len(mesh.polygons), float(co[::step].sum()))


def get_cells(obj):
    """Analysis of obj's mesh (cached until the mesh changes)."""
    mesh = obj.data
    fp = _fingerprint(mesh)
    hit = _CACHE.get(mesh.name)
    if hit and hit[0] == fp:
        return hit[1]
    target = int(np.clip(len(mesh.polygons) / 25, 4000, 32000))
    C = segment.Cells(mesh, target_cells=target)
    _CACHE.clear()
    _CACHE[mesh.name] = (fp, C)
    return C


ALL_DETAILS = ("guard", "ferrule", "fuller", "hidden")


def add_details(C, lab, names, axis, details=ALL_DETAILS):
    """Sub-parts of a sword-like split: guard centre / details, grip ferrules,
    blade fuller; and buried faces follow the part that covers them."""
    lab = lab.copy()
    names = list(names)

    def new(mask, name):
        if mask is None or not mask.any():
            return False
        lab[mask] = len(names)
        names.append(name)
        return True

    if "guard" in details and "Guard" in names:
        r = C.guard_subparts(lab == names.index("Guard"))
        if r is not None:
            sub, snames = r
            g = names.index("Guard")
            base = len(names)
            for k in range(1, len(snames)):
                lab[sub == k] = base + k - 1
            names += snames[1:]
    if "ferrule" in details and "Grip" in names:
        new(C.find_ferrules(lab == names.index("Grip"), axis), "Ferrule")
    if "fuller" in details and "Blade" in names:
        new(C.find_fuller(lab == names.index("Blade")), "Fuller")
    if "hidden" in details and len(names) > 1:
        hid, cover = C.hidden_cells(lab)
        move = hid & (cover >= 0) & (cover != lab)
        lab[move] = cover[move]
    # drop names that ended up empty
    ids = np.unique(lab)
    remap = -np.ones(len(names), np.int64)
    remap[ids] = np.arange(len(ids))
    return remap[lab], [names[i] for i in ids.tolist()]


def auto_labels(C, method="SHAPE", sensitivity=1.0, details=ALL_DETAILS):
    """(cell labels, names, method used)."""
    if method == "SHAPE":
        r = C.elongated_parts()
        if r is not None:
            lab, names, info = r
            if details:
                lab, names = add_details(C, lab, names, info["axis"], details)
            return lab, names, "SHAPE"
    lab = C.segment(k=1.0 / max(sensitivity, 0.05))
    n = int(lab.max()) + 1
    # biggest first, so "Region 1" is the main part
    areas = np.bincount(lab, weights=C.area, minlength=n)
    order = np.argsort(-areas)
    remap = np.empty(n, np.int64)
    remap[order] = np.arange(n)
    return remap[lab], [f"Region {i + 1}" for i in range(n)], "CREASES"


# ------------------------------------------------------------------ session
class Session:
    """Region editing state; pure data, no drawing (testable headless)."""

    def __init__(self, obj, C):
        self.obj = obj
        self.C = C
        self.labels = np.zeros(C.n, np.int64)
        self.names = ["Region 1"]
        self.slot_of = [None]            # existing material slot a region came from
        self.active = 0
        self.pos, self.neg = [], []
        self.extra = np.zeros(C.n, bool)    # lasso additions
        self.removed = np.zeros(C.n, bool)  # lasso removals
        self.tol = 0.5
        self._dcache = {}
        self.sel = np.zeros(C.n, bool)
        self.history = []
        self.method = None
        self.sensitivity = 1.0
        self.mirror = False
        self.rules = {}                     # region name -> {"kind": "edge", ...}
        self._parts = None
        self._init = None
        self.version = 0                    # bumped on every edit (overlay cache key)

    # ---- undo
    def push(self):
        self.version += 1
        self.history.append((self.labels.copy(), list(self.names), list(self.slot_of), self.active,
                             list(self.pos), list(self.neg), self.extra.copy(), self.removed.copy(),
                             self.tol, self.mirror, {k: dict(v) for k, v in self.rules.items()}))
        del self.history[:-60]

    def undo(self):
        if not self.history:
            return False
        self.version += 1
        (self.labels, self.names, self.slot_of, self.active, self.pos, self.neg,
         self.extra, self.removed, self.tol, self.mirror, self.rules) = self.history.pop()
        self._grow()
        return True

    # ---- initial state
    def init_from_slots(self):
        """Start from the object's existing material slots (re-editing). Faces
        keep their exact slot unless their part of the surface is edited."""
        mesh = self.obj.data
        nf = len(mesh.polygons)
        mi = np.empty(nf, np.int32)
        mesh.polygons.foreach_get("material_index", mi)
        ns = max(len(self.obj.material_slots), 1)
        mi = np.clip(mi, 0, ns - 1)
        used = np.unique(mi)
        if len(used) < 2:
            return False
        votes = np.zeros((self.C.n, ns))
        np.add.at(votes, (self.C.face_cell, mi), self.C._face_area)
        votes[:, np.setdiff1d(np.arange(ns), used)] = -1.0
        cl = votes.argmax(1)
        remap = -np.ones(ns, np.int64)
        remap[used] = np.arange(len(used))
        self.labels = remap[cl]
        stored = list(self.obj.get("fm_region_names", []))
        self.names, self.slot_of = [], []
        for s in used.tolist():
            m = self.obj.material_slots[s].material
            nm = stored[s] if s < len(stored) and stored[s] else (m.name if m else f"Slot {s + 1}")
            while nm in self.names:
                nm += "+"
            self.names.append(nm)
            self.slot_of.append(int(s))
        # exact starting point, by name (indices shift as regions come and go)
        self._init = {"names": list(self.names), "cells": self.labels.copy(), "faces": remap[mi]}
        try:
            import json
            self.rules = {k: v for k, v in json.loads(self.obj.get("fm_region_rules", "{}")).items()
                          if k in self.names and v.get("parent") in self.names}
        except (ValueError, AttributeError, TypeError):
            self.rules = {}
        self._init["rules"] = repr(sorted(self.rules.items()))
        self.method = "SLOTS"
        return True

    def auto(self, method="SHAPE", sensitivity=None, details=ALL_DETAILS):
        self.push()
        if sensitivity is not None:
            self.sensitivity = sensitivity
        lab, names, used = auto_labels(self.C, method, self.sensitivity, details)
        self.labels, self.names = lab, list(names)
        self.slot_of = [None] * len(names)
        self.rules = {}
        if "Fuller" in self.names and "Blade" in self.names:
            self.rules["Fuller"] = {"kind": "band", "parent": "Blade"}
        self._init = None
        self.method = used
        self.active = 0
        self.clear_selection(push=False)
        return used

    # ---- selection
    def clear_selection(self, push=True):
        if push:
            self.push()
        self.pos, self.neg = [], []
        self.extra[:] = False
        self.removed[:] = False
        self.sel[:] = False
        self.tol = 0.5

    def has_selection(self):
        return bool(self.sel.any())

    def _dist(self, seeds):
        key = tuple(sorted(seeds))
        if self._dcache.get("key") != key:
            self._dcache = {"key": key, "d": self.C.bottleneck(list(key))}
        return self._dcache["d"]

    def grow_mask(self, pos, neg, tol):
        """Cells reachable from the clicks without climbing a crease higher than
        tol, minus what is closer to an Alt+click."""
        C = self.C
        if not pos:
            return np.zeros(C.n, bool)
        bp, lp = self._dist(pos) if list(pos) == self.pos else C.bottleneck(list(pos))
        m = bp <= tol
        if neg:
            bn, ln = C.bottleneck(list(neg))
            m &= (bp < bn) | ((bp == bn) & (lp < ln))
        else:
            bn = None
        # the crease "wall" is a few cells thick; reclaim the cells on this side
        # of it: step over edges that do not fold inward
        i, j = C.edges[:, 0], C.edges[:, 1]
        soft = C.w < 0.12 * max(1.0, tol / 0.35)
        for _ in range(3):
            add = np.zeros(C.n, bool)
            e = soft & (m[i] ^ m[j])
            add[np.where(m[i[e]], j[e], i[e])] = True
            add &= ~m
            if bn is not None:
                add &= ~((bn < bp) | ((bn == bp) & (ln <= lp)))
            if not add.any():
                break
            m |= add
        return m

    def _grow(self):
        sel = (self.grow_mask(self.pos, self.neg, self.tol) | self.extra) & ~self.removed
        if self.mirror and sel.any():
            sel = self.C.mirror_mask(sel)
        self.sel = sel

    def _snap(self, cell):
        """A click right on a crease is ambiguous; move the seed a few cells off
        the crease, to the side the clicked surface faces."""
        C = self.C
        if C.wall[cell] < 0.15:
            return int(cell)
        ring = {int(cell)}
        frontier = [int(cell)]
        for _ in range(4):
            nxt = []
            for c in frontier:
                for k in C.adj[C.indptr[c]:C.indptr[c + 1]].tolist():
                    if k not in ring:
                        ring.add(k)
                        nxt.append(k)
            frontier = nxt
        cand = np.array(sorted(ring))
        cand = cand[C.wall[cand] < 0.15]
        if not len(cand):
            return int(cell)
        n0 = C.nor[cell]
        score = np.linalg.norm(C.pos[cand] - C.pos[cell], axis=1) / C.cell + 4.0 * (1.0 - C.nor[cand] @ n0)
        return int(cand[np.argmin(score)])

    def click(self, cell, positive=True):
        self.push()
        (self.pos if positive else self.neg).append(self._snap(cell))
        self._grow()

    def preview(self, cell):
        """What a click on `cell` would select (not stored)."""
        return self.grow_mask(self.pos + [self._snap(cell)], self.neg, self.tol) | self.extra

    def step_tol(self, up=True, ratio=1.15):
        """Wheel: jump to the next level at which the selection visibly grows
        (or shrinks), i.e. the next crease the region would spill over."""
        if not self.pos:
            return False
        b, _ = self._dist(self.pos)
        ok = np.isfinite(b)
        o = np.argsort(b[ok])
        bs = b[ok][o]
        ca = np.cumsum(self.C.area[ok][o])
        area = lambda t: ca[np.searchsorted(bs, t, side="right") - 1] if t >= bs[0] else 0.0
        cur = area(self.tol)
        lv = np.unique(np.round(bs, 4))
        if up:
            cand = lv[(lv > self.tol)]
            cand = [t for t in cand.tolist() if area(t) >= cur * ratio + 1e-12]
            t = cand[0] if cand else None
        else:
            cand = lv[(lv < self.tol)]
            cand = [t for t in cand.tolist() if area(t) * ratio <= cur and area(t) > 0]
            t = cand[-1] if cand else None
        if t is None:
            return False
        self.push()
        self.tol = float(t) + (1e-6 if up else 0.0)
        self._grow()
        return True

    def lasso(self, mask, subtract=False):
        self.push()
        if subtract:
            self.removed |= mask
            self.extra &= ~mask
        else:
            self.extra |= mask
            self.removed &= ~mask
        self._grow()

    # ---- smarter selection
    def parts(self):
        """Whole-part labels (shape split, else creases) for double-click."""
        if self._parts is None:
            self._parts = auto_labels(self.C, "SHAPE", self.sensitivity, details=("hidden",))[0]
        return self._parts

    def select_part(self, cell):
        """Double-click: the whole part under the cursor (e.g. the entire guard)."""
        self.push()
        part = self.parts()
        self.extra |= part == part[int(cell)]
        self.removed &= ~self.extra
        self._grow()

    def toggle_mirror(self):
        """Mirror every selection across the object's symmetry planes."""
        if not self.mirror and not self.C.mirror_maps():
            return None
        self.push()
        self.mirror = not self.mirror
        self._grow()
        return self.mirror

    def select_similar(self, tol_ratio=0.25, max_matches=64):
        """Add every other piece shaped like the selection (all the spikes, all
        the rivets...). Pieces are cut at the current crease level and compared
        by area and extents, so rotated / mirrored copies match too."""
        if not self.sel.any():
            return 0
        C = self.C
        ra, rext = C.mask_signature(self.sel)
        comp = C.pieces(self.tol)
        _, A, ext, ids = C.shape_signature(comp, [])
        lo, hi = 1.0 - tol_ratio, 1.0 / (1.0 - tol_ratio)
        ok = (A >= ra * lo) & (A <= ra * hi)
        ok &= np.all((ext >= rext * lo - 0.25 * C.cell) & (ext <= rext * hi + 0.25 * C.cell), axis=1)
        cand = ids[ok]
        # skip the selected piece itself
        inside = np.bincount(comp[self.sel & (comp >= 0)], minlength=int(comp.max()) + 1)
        found = [int(c) for c in cand.tolist()
                 if inside[c] < 0.5 * np.count_nonzero(comp == c)][:max_matches]
        if not found:
            return 0
        self.push()
        self.extra |= np.isin(comp, found)
        self.removed &= ~self.extra
        self._grow()
        return len(found)

    def add_edge_band(self, frac=0.14, parent=None):
        """Make the sharpened edge strip of the blade its own region. The strip
        is refined per face when applied, so its border is exact."""
        if parent is None:
            for nm in ("Blade", "Body"):
                if nm in self.names:
                    parent = self.names.index(nm)
                    break
            else:
                parent = self.active
        pname = self.names[parent]
        if self.rules.get(pname, {}).get("kind") == "edge":   # clicked the edge itself: use its parent
            pname = self.rules[pname]["parent"]
            parent = self.names.index(pname)
        cells = np.nonzero(self.labels == parent)[0]
        if len(cells) < 20:
            return None
        self.push()
        # drop an existing edge region of this parent first (frac change)
        for nm, r in list(self.rules.items()):
            if r.get("kind") == "edge" and r.get("parent") == pname and nm in self.names:
                self.labels[self.labels == self.names.index(nm)] = parent
                del self.rules[nm]
        self._compact()
        parent = self.names.index(pname)
        cells = np.nonzero(self.labels == parent)[0]
        frame = self.C.edge_frame(cells)
        band = self.C.edge_band(frame, self.C.pos[cells], frac)
        n = len(self.names)
        name = "Edge" if "Edge" not in self.names else f"{pname} Edge"
        self.labels[cells[band]] = n
        self.names.append(name)
        self.slot_of.append(None)
        self.rules[name] = {"kind": "edge", "parent": pname, "frac": float(frac)}
        self._compact()
        return name

    def remove_edge_band(self):
        if not any(r.get("kind") == "edge" for r in self.rules.values()):
            return False
        self.push()
        for nm, r in list(self.rules.items()):
            if r.get("kind") != "edge":
                continue
            if nm in self.names and r.get("parent") in self.names:
                self.labels[self.labels == self.names.index(nm)] = self.names.index(r["parent"])
            del self.rules[nm]
        self._compact()
        return True

    def rename(self, r, name):
        name = name.strip()[:40]
        if not name or name in self.names:
            return False
        self.push()
        old = self.names[r]
        self.names[r] = name
        if self._init:
            self._init["names"] = [name if n == old else n for n in self._init["names"]]
        if old in self.rules:
            self.rules[name] = self.rules.pop(old)
        for v in self.rules.values():
            if v.get("parent") == old:
                v["parent"] = name
        return True

    def face_labels(self, precise=True):
        """Final per-face labels. Borders are decided face by face along the
        real creases (precise), the blade edge strip is measured from the
        blade's outline per face, and stray specks are cleaned up."""
        C = self.C
        fl = C.face_labels_smooth(self.labels) if precise else C.face_labels(self.labels)
        edges, bands = {}, {}
        for nm, r in self.rules.items():
            if nm not in self.names or r.get("parent") not in self.names:
                continue
            if r.get("kind") == "edge":
                edges[self.names.index(nm)] = (self.names.index(r["parent"]), r["frac"])
            elif r.get("kind") == "band":
                bands[self.names.index(nm)] = self.names.index(r["parent"])
        # the edge strip is measured afresh below, from its parent's faces
        for e, (p, _) in edges.items():
            fl[fl == e] = p
        if precise and getattr(C, "_ftopo", None) is not None:
            fl = C.clean_speckles(fl)
        # bands (the fuller): a smooth strip down the parent, face by face
        for f, p in bands.items():
            faces = np.nonzero((fl == f) | (fl == p))[0]
            fcells = np.nonzero(self.labels == f)[0]
            if not len(faces) or len(fcells) < 10:
                continue
            all_cells = np.nonzero(np.isin(self.labels, [f, p] + list(edges)))[0]
            frame = C.edge_frame(all_cells)
            inside = C.band_faces(frame, fcells, faces)
            fl[faces] = np.where(inside, f, p)
        for e, (p, frac) in edges.items():
            faces = np.nonzero(fl == p)[0]
            if not len(faces):
                continue
            cells = np.nonzero((self.labels == e) | (self.labels == p))[0]
            fa = C._face_area
            spacing = float(np.sqrt(np.median(fa[faces]))) * 1.2
            frame = C.edge_frame(cells, outline_points=C._fcen.astype(np.float64), spacing=spacing)
            band = C.edge_band(frame, C._fcen[faces].astype(np.float64), frac)
            fl[faces] = np.where(band, e, p)
        if precise and getattr(C, "_ftopo", None) is not None:
            fl = C.smooth_face_borders(fl)
            fl = C.clean_islands(fl)
        # untouched parts keep exactly the faces they had when re-entered
        if self._init:
            cur = np.array([self.names.index(n) if n in self.names else -1 for n in self._init["names"]])
            same_cell = cur[self._init["cells"]] == self.labels
            if self._init.get("rules") != repr(sorted(self.rules.items())):
                # the edge strip changed: its parts are recomputed, not kept
                touched = set()
                for nm, r in self.rules.items():
                    touched |= {nm, r.get("parent")}
                for k, nm in enumerate(self.names):
                    if nm in touched:
                        same_cell &= self.labels != k
            keep = same_cell[C.face_cell] & (cur[self._init["faces"]] >= 0)
            fl[keep] = cur[self._init["faces"][keep]]
        return fl

    # ---- region edits
    def selection_to_new(self):
        if not self.sel.any():
            return False
        self.push()
        n = len(self.names)
        self.labels[self.sel] = n
        self.names.append(f"Region {n + 1}")
        self.slot_of.append(None)
        self.active = n
        self._compact()
        self.clear_selection(push=False)
        return True

    def selection_to_active(self):
        if not self.sel.any():
            return False
        self.push()
        self.labels[self.sel] = self.active
        self._compact()
        self.clear_selection(push=False)
        return True

    def merge(self, a, b):
        """Merge region b into region a."""
        if a == b:
            return False
        self.push()
        self.labels[self.labels == b] = a
        self._compact()
        return True

    def split(self, r):
        """Split region r at its creases (finer than the current proposal)."""
        m = self.labels == r
        if m.sum() < 8:
            return 0
        best = None
        for k in (0.6, 0.35, 0.2, 0.1):
            fine = self.C.segment(k=k, min_frac=0.0)
            sub = fine[m]
            ids, inv = np.unique(sub, return_inverse=True)
            areas = np.bincount(inv, weights=self.C.area[m])
            keep = areas >= 0.04 * areas.sum()
            if keep.sum() >= 2:
                best = (sub, ids, areas, keep)
                break
        if best is None:
            return 0
        self.push()
        sub, ids, areas, keep = best
        big = ids[keep]
        # small fragments join the largest piece
        main = ids[np.argmax(areas)]
        sub = np.where(np.isin(sub, big), sub, main)
        cells = np.nonzero(m)[0]
        added = 0
        for pid in big.tolist():
            if pid == main:
                continue
            n = len(self.names)
            self.labels[cells[sub == pid]] = n
            self.names.append(f"{self.names[r]} {added + 2}")
            self.slot_of.append(None)
            added += 1
        self._compact()
        return added + 1

    def _compact(self):
        ids = np.unique(self.labels)
        if len(ids) == len(self.names):
            return
        remap = -np.ones(len(self.names), np.int64)
        remap[ids] = np.arange(len(ids))
        self.labels = remap[self.labels]
        self.names = [self.names[i] for i in ids.tolist()]
        self.slot_of = [self.slot_of[i] for i in ids.tolist()]
        self.active = int(remap[self.active]) if remap[self.active] >= 0 else 0

    def region_areas(self):
        return np.bincount(self.labels, weights=self.C.area, minlength=len(self.names))

    def face_colors(self, hover_region=-1, preview=None, face_labels=None):
        """Overlay colours per face. face_labels (exact borders) is used when
        given; otherwise faces take their cell's colour."""
        cc = self.cell_colors(hover_region, preview)
        col = cc[self.C.face_cell]
        if face_labels is not None:
            rgb = np.array([region_color(i) for i in range(len(self.names))], np.float32)
            plain = ~(self.sel | (preview if preview is not None else False))[self.C.face_cell]
            col[plain, :3] = rgb[face_labels[plain]]
        return col

    # ---- colours for the overlay (one RGBA per cell)
    def cell_colors(self, hover_region=-1, preview=None):
        rgb = np.array([region_color(i) for i in range(len(self.names))], np.float32)
        col = np.empty((self.C.n, 4), np.float32)
        col[:, :3] = rgb[self.labels]
        col[:, 3] = 0.38
        if 0 <= self.active < len(self.names):
            col[self.labels == self.active, 3] = 0.50
        if hover_region >= 0:
            col[self.labels == hover_region, 3] = 0.62
        if preview is not None and preview.any():
            col[preview, :3] = col[preview, :3] * 0.4 + 0.6
            col[preview, 3] = 0.55
        if self.sel.any():
            col[self.sel] = (1.0, 0.92, 0.2, 0.78)
        return col


# ------------------------------------------------------------------ apply
def apply_regions(obj, face_labels, names, slot_of):
    """Write regions as material slots + face material indices.

    Only material slots and polygon material_index change. Existing slots are
    kept and reused by the region that holds most of their faces; every other
    region gets a copy of the material it had, named after the region."""
    mesh = obj.data
    nf = len(mesh.polygons)
    old = np.empty(nf, np.int32)
    mesh.polygons.foreach_get("material_index", old)
    mats = [s.material for s in obj.material_slots]
    if not mats:
        mesh.materials.append(None)
        mats = [None]
    old = np.clip(old, 0, len(mats) - 1)
    R = len(names)
    share = np.zeros((R, len(mats)))
    np.add.at(share, (face_labels, old), 1)

    target = [None] * R
    taken = set()
    for r in range(R):                      # explicit (re-edit) slots first
        s = slot_of[r] if r < len(slot_of) else None
        if s is not None and s < len(mats) and s not in taken:
            target[r] = s
            taken.add(s)
    # each remaining slot goes to the region holding most of its faces
    for s in np.argsort(-share.max(0)).tolist():
        if s in taken:
            continue
        r = int(np.argmax(share[:, s]))
        if share[r, s] > 0 and target[r] is None:
            target[r] = s
            taken.add(s)

    # empty slots this tool made earlier can be reused before adding new ones
    used_after = set(t for t in target if t is not None)
    spare = [s for s, m in enumerate(mats)
             if s not in used_after and not np.any(old == s) and (m is None or m.get("fm_region"))]
    stored = list(obj.get("fm_region_names", []))
    stored += [""] * (len(mats) - len(stored))
    created = []
    for r in range(R):
        if target[r] is not None:
            continue
        src = mats[int(np.argmax(share[r]))] if share[r].sum() > 0 else mats[0]
        if src is not None:
            m = src.copy()
            base = src.get("fm_region_base", src.name)
            m.name = f"{base}_{names[r]}"
            m["fm_region_base"] = base
        else:
            m = bpy.data.materials.new(f"Region_{names[r]}")
        m["fm_region"] = names[r]
        if spare:
            s = spare.pop(0)
            obj.material_slots[s].material = m
        else:
            mesh.materials.append(m)
            s = len(obj.material_slots) - 1
            stored.append("")
        target[r] = s
        created.append(m.name)
    keep = set(target)
    stored = [nm if s in keep else "" for s, nm in enumerate(stored)]
    for r in range(R):
        if target[r] < len(stored):
            stored[target[r]] = names[r]
    new_idx = np.array(target, np.int32)[face_labels]
    mesh.polygons.foreach_set("material_index", new_idx)
    mesh.update()
    obj["fm_region_names"] = stored[:len(obj.material_slots)]
    return target, created


def session_apply(sess):
    import json
    fl = sess.face_labels()
    out = apply_regions(sess.obj, fl, sess.names, sess.slot_of)
    sess.obj["fm_region_rules"] = json.dumps({k: v for k, v in sess.rules.items() if k in sess.names})
    return out


# ------------------------------------------------------------------ drawing
_SHADER = None


def _shader():
    global _SHADER
    if _SHADER is None:
        import gpu
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "FLOAT", "cid")
        iface = gpu.types.GPUStageInterfaceInfo("fm_region_iface")
        iface.flat("VEC4", "col")
        info.vertex_out(iface)
        info.push_constant("MAT4", "mvp")
        info.push_constant("FLOAT", "zoff")
        info.sampler(0, "FLOAT_2D", "tex")
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(
            "void main() {"
            "  int id = int(cid + 0.5);"
            "  ivec2 sz = textureSize(tex, 0);"
            "  col = texelFetch(tex, ivec2(id % sz.x, id / sz.x), 0);"
            "  gl_Position = mvp * vec4(pos, 1.0);"
            "  gl_Position.z -= zoff * gl_Position.w;"
            "}")
        info.fragment_source(
            "void main() {"
            "  if (col.a <= 0.001) { discard; }"
            "  fragColor = col;"
            "}")
        _SHADER = gpu.shader.create_from_info(info)
    return _SHADER


class Overlay:
    def __init__(self, obj, C):
        import gpu
        mesh = obj.data
        mesh.calc_loop_triangles()
        nt = len(mesh.loop_triangles)
        tv = np.empty(nt * 3, np.int32)
        tp = np.empty(nt, np.int32)
        mesh.loop_triangles.foreach_get("vertices", tv)
        mesh.loop_triangles.foreach_get("polygon_index", tp)
        co = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        pos = co[tv]
        cid = np.repeat(tp.astype(np.float32), 3)            # face id per corner
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
        fmt.attr_add(id="cid", comp_type="F32", len=1, fetch_mode="FLOAT")
        vbo = gpu.types.GPUVertBuf(fmt, len(pos))
        vbo.attr_fill("pos", np.ascontiguousarray(pos))
        vbo.attr_fill("cid", np.ascontiguousarray(cid.reshape(-1, 1)))
        self.batch = gpu.types.GPUBatch(type="TRIS", buf=vbo)
        self.W = 2048
        self.H = max(1, math.ceil(len(mesh.polygons) / self.W))
        self.tex = None

    def set_colors(self, col):
        import gpu
        buf = np.zeros((self.W * self.H, 4), np.float32)
        buf[:len(col)] = col
        data = gpu.types.Buffer("FLOAT", self.W * self.H * 4, buf.ravel())
        self.tex = gpu.types.GPUTexture((self.W, self.H), format="RGBA16F", data=data)

    def draw(self, obj, rv3d):
        import gpu
        if self.tex is None:
            return
        sh = _shader()
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("ALPHA")
        sh.bind()
        sh.uniform_float("mvp", rv3d.perspective_matrix @ obj.matrix_world)
        sh.uniform_float("zoff", 4e-5 if rv3d.is_perspective else 2e-5)
        sh.uniform_sampler("tex", self.tex)
        self.batch.draw(sh)
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("NONE")


HELP = [
    "Click: select this piece   Double-click: whole part   Alt+Click: not this   Wheel: bigger / smaller",
    "M: mirror (symmetry)   S: select similar pieces   L: lasso (Alt = remove)   C: clear",
    "Space: selection -> new region   E: add to active   Shift+Click: merge   Ctrl+Click: split",
    "B: blade edge strip (press again: wider / off)   F2: rename   Tab: next region",
    "A: auto split (shape / creases)   [ ]: crease sensitivity   Ctrl+Z: undo   Enter: apply   Esc: cancel",
]


def _draw_px(op, context):
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader
    if context.area != op.area:
        return
    s = op.sess
    ui = context.preferences.system.ui_scale
    fid = 0
    x0, y = 20 * ui, 30 * ui
    sh = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    # dark backing so the text reads over any model
    blf.size(fid, 12 * ui)
    pw = max(blf.dimensions(fid, ln)[0] for ln in HELP) + 24 * ui
    ph = (len(HELP) * 17 + 8 + 22 + 19 * len(s.names) + 30) * ui
    bx, by = x0 - 10 * ui, y - 12 * ui
    b = batch_for_shader(sh, "TRIS", {"pos": [(bx, by), (bx + pw, by), (bx + pw, by + ph),
                                             (bx, by), (bx + pw, by + ph), (bx, by + ph)]})
    sh.bind()
    sh.uniform_float("color", (0.05, 0.05, 0.06, 0.62))
    b.draw(sh)
    lines = list(reversed(HELP))
    blf.size(fid, 12 * ui)
    for ln in lines:
        blf.color(fid, 0.85, 0.85, 0.85, 0.9)
        blf.position(fid, x0, y, 0)
        blf.draw(fid, ln)
        y += 17 * ui
    y += 8 * ui
    info = f"Grow level: {s.tol:.2f}" if s.has_selection() else "Click a part to select it"
    if s.method == "CREASES":
        info += f"    Sensitivity: {s.sensitivity:.2f}"
    if s.mirror:
        info += "    MIRROR on"
    if op.lasso_mode:
        info += "    LASSO: drag with the left mouse"
    if op.typing is not None:
        info = f"Rename '{s.names[s.active]}':  {op.typing}_    (Enter = ok, Esc = cancel)"
    if op.message:
        info += "    " + op.message
    blf.color(fid, 1.0, 0.92, 0.4, 1.0)
    blf.position(fid, x0, y, 0)
    blf.draw(fid, info)
    y += 22 * ui
    areas = s.region_areas()
    tot = max(areas.sum(), 1e-12)
    blf.size(fid, 13 * ui)
    for i in reversed(range(len(s.names))):
        c = region_color(i)
        sq = 11 * ui
        b = batch_for_shader(sh, "TRIS", {"pos": [(x0, y), (x0 + sq, y), (x0 + sq, y + sq),
                                                 (x0, y), (x0 + sq, y + sq), (x0, y + sq)]})
        sh.bind()
        sh.uniform_float("color", (*c, 1.0))
        b.draw(sh)
        act = i == s.active
        blf.color(fid, 1, 1, 1, 1.0 if act else 0.75)
        blf.position(fid, x0 + sq + 8 * ui, y, 0)
        blf.draw(fid, ("> " if act else "") + f"{s.names[i]}   {100 * areas[i] / tot:.0f}%")
        y += 19 * ui
    blf.size(fid, 15 * ui)
    blf.color(fid, 1, 1, 1, 1)
    blf.position(fid, x0, y + 4 * ui, 0)
    blf.draw(fid, f"Smart Regions  -  {op.obj.name}")
    if op.lasso_pts and len(op.lasso_pts) > 1:
        b = batch_for_shader(sh, "LINE_STRIP", {"pos": op.lasso_pts + [op.lasso_pts[0]]})
        sh.bind()
        sh.uniform_float("color", (1, 0.92, 0.3, 1))
        b.draw(sh)
    gpu.state.blend_set("NONE")


def _draw_3d(op, context):
    if context.area != op.area or op.overlay is None:
        return
    op.overlay.draw(op.obj, context.region_data)


def _point_in_poly(px, py, poly):
    """Vectorised even-odd test of points (px, py) against a 2D polygon."""
    inside = np.zeros(len(px), bool)
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xi)
    return inside


class FM_OT_regions(bpy.types.Operator):
    """Split the object into material regions: an automatic proposal you refine
    by clicking. Only material slots and face material indices change"""
    bl_idname = "fm.regions"
    bl_label = "Smart Regions"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and context.mode == "OBJECT"

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "Run from the 3D Viewport")
            return {"CANCELLED"}
        obj = context.active_object
        t = time.time()
        self.obj = obj
        self.area = context.area
        self.region = next(r for r in context.area.regions if r.type == "WINDOW")
        C = get_cells(obj)
        self.sess = Session(obj, C)
        if not self.sess.init_from_slots():
            self.sess.auto("SHAPE")
            self.sess.history.clear()
        try:
            self.overlay = Overlay(obj, C)
        except Exception as e:  # no GPU (background) or unsupported backend
            self.report({"ERROR"}, f"Smart Regions overlay unavailable: {e}")
            return {"CANCELLED"}
        # face indices from a ray cast only match when modifiers keep the topology
        dg = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        self.ray_ok = len(ev.data.polygons) == len(obj.data.polygons)
        self.hover_cell = -1
        self.hover_region = -1
        self.preview = None
        self.preview_for = -1
        self.last_move = time.time()
        self.lasso_mode = False
        self.lasso_pts = []
        self.typing = None
        self.refined = None
        self.changed_at = time.time()
        self.message = "" if self.ray_ok else "modifiers change the mesh: clicks disabled, use A / lasso"
        _STATE["kill"] = False
        self._refresh()
        self.h3d = bpy.types.SpaceView3D.draw_handler_add(_draw_3d, (self, context), "WINDOW", "POST_VIEW")
        self.hpx = bpy.types.SpaceView3D.draw_handler_add(_draw_px, (self, context), "WINDOW", "POST_PIXEL")
        self.timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.message = self.message or f"analysed in {time.time() - t:.1f}s"
        self.area.tag_redraw()
        return {"RUNNING_MODAL"}

    # ---- helpers
    def _refresh(self, refined=None):
        """Recolour the overlay. Right after a change faces take their cell's
        colour (instant); once idle, borders are refined per face."""
        if refined is None:
            cache = getattr(self, "_fl_cache", None)
            if cache and cache[0] == self._label_key():
                refined = cache[1]            # nothing changed since the last refine
            else:
                self.refined = None
                self.changed_at = time.time()
        self.overlay.set_colors(self.sess.face_colors(self.hover_region, self.preview, refined))
        self.area.tag_redraw()

    def _label_key(self):
        s = self.sess
        return (hash(s.labels.tobytes()), repr(sorted(s.rules.items())), tuple(s.names))

    def _refine_if_idle(self):
        if self.refined is not None or time.time() - self.changed_at < 0.5:
            return
        cache = getattr(self, "_fl_cache", None)
        if cache and cache[0] == self._label_key():
            self.refined = cache[1]
            return
        if self.sess.C.n_faces > 1_500_000:
            self.refined = False
            return
        self.refined = self.sess.face_labels(precise=self.sess.C.n_faces <= 300_000)
        self._fl_cache = (self._label_key(), self.refined)
        self._refresh(self.refined)

    def _cell_at(self, context, x, y):
        if not self.ray_ok:
            return -1
        from bpy_extras import view3d_utils
        rv3d = context.region_data
        org = view3d_utils.region_2d_to_origin_3d(self.region, rv3d, (x, y))
        vec = view3d_utils.region_2d_to_vector_3d(self.region, rv3d, (x, y))
        inv = self.obj.matrix_world.inverted()
        o = inv @ org
        d = (inv.to_3x3() @ vec).normalized()
        dg = context.evaluated_depsgraph_get()
        ok, _loc, _n, fi = self.obj.ray_cast(o, d, depsgraph=dg)
        if not ok or fi < 0 or fi >= self.sess.C.n_faces:
            return -1
        return int(self.sess.C.face_cell[fi])

    def _finish(self, context, apply):
        bpy.types.SpaceView3D.draw_handler_remove(self.h3d, "WINDOW")
        bpy.types.SpaceView3D.draw_handler_remove(self.hpx, "WINDOW")
        context.window_manager.event_timer_remove(self.timer)
        self.area.tag_redraw()
        if not apply:
            return {"CANCELLED"}
        target, created = session_apply(self.sess)
        self.obj.active_material_index = target[self.sess.active]
        self.report({"INFO"}, f"{len(self.sess.names)} regions as material slots"
                              + (f" ({len(created)} new)" if created else ""))
        return {"FINISHED"}

    def _lasso_cells(self, context):
        from bpy_extras.view3d_utils import location_3d_to_region_2d  # noqa: F401
        C = self.sess.C
        rv3d = context.region_data
        M = np.array(rv3d.perspective_matrix @ self.obj.matrix_world)
        p = np.c_[C.pos, np.ones(C.n)] @ M.T
        w = np.maximum(p[:, 3], 1e-9)
        sx = (p[:, 0] / w * 0.5 + 0.5) * self.region.width
        sy = (p[:, 1] / w * 0.5 + 0.5) * self.region.height
        inside = _point_in_poly(sx, sy, self.lasso_pts) & (p[:, 3] > 0)
        # only surfaces facing the viewer
        inv = np.array(self.obj.matrix_world.inverted())
        if rv3d.is_perspective:
            eye = np.array(rv3d.view_matrix.inverted().translation)
            eye_l = (inv @ np.r_[eye, 1.0])[:3]
            facing = np.einsum("ij,ij->i", C.nor, eye_l - C.pos) > 0
        else:
            vd = np.array(rv3d.view_rotation @ __import__("mathutils").Vector((0, 0, 1)))
            vd_l = inv[:3, :3] @ vd
            facing = C.nor @ vd_l > 0
        return inside & facing

    # ---- modal
    def modal(self, context, event):
        s = self.sess
        if _STATE.get("kill"):
            return self._finish(context, apply=False)
        if event.type == "TIMER":
            if (self.hover_cell >= 0 and self.preview_for != self.hover_cell
                    and time.time() - self.last_move > 0.15 and not self.lasso_mode):
                self.preview_for = self.hover_cell
                self.preview = s.preview(self.hover_cell) if not s.has_selection() else None
                self._refresh()
            elif self.preview is None:
                self._refine_if_idle()
            return {"PASS_THROUGH"}

        x = event.mouse_x - self.region.x
        y = event.mouse_y - self.region.y
        inside = 0 <= x <= self.region.width and 0 <= y <= self.region.height
        if not inside and event.type not in {"ESC", "RET", "NUMPAD_ENTER"}:
            return {"PASS_THROUGH"}

        # ---- lasso
        if self.lasso_mode:
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                self.lasso_pts = [(x, y)]
                return {"RUNNING_MODAL"}
            if event.type == "MOUSEMOVE" and self.lasso_pts:
                self.lasso_pts.append((x, y))
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self.lasso_pts:
                if len(self.lasso_pts) > 2:
                    s.lasso(self._lasso_cells(context), subtract=event.alt)
                self.lasso_pts = []
                self.lasso_mode = False
                self.message = ""
                self._refresh()
                return {"RUNNING_MODAL"}
            if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
                self.lasso_mode = False
                self.lasso_pts = []
                self.message = ""
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}

        # ---- rename (F2): type the new name
        if self.typing is not None:
            if event.value != "PRESS":
                return {"RUNNING_MODAL"}
            if event.type in {"RET", "NUMPAD_ENTER"}:
                if s.rename(s.active, self.typing):
                    self.message = f"renamed: {s.names[s.active]}"
                self.typing = None
            elif event.type == "ESC":
                self.typing = None
            elif event.type == "BACK_SPACE":
                self.typing = self.typing[:-1]
            elif event.unicode and event.unicode.isprintable():
                self.typing = (self.typing + event.unicode)[:40]
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "DOUBLE_CLICK":
            c = self._cell_at(context, x, y)
            if c >= 0:
                s.select_part(c)
                self.message = "whole part"
                self.preview, self.preview_for = None, -1
                self._refresh()
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE":
            self.last_move = time.time()
            c = self._cell_at(context, x, y)
            r = int(s.labels[c]) if c >= 0 else -1
            if c != self.hover_cell:
                self.hover_cell = c
                if self.preview is not None:
                    self.preview = None
                    self.preview_for = -1
                    self.hover_region = r
                    self._refresh()
                elif r != self.hover_region:
                    self.hover_region = r
                    self._refresh()
            return {"PASS_THROUGH"}

        if event.value != "PRESS" and event.type not in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            return {"PASS_THROUGH"}

        t = event.type
        if t in {"RET", "NUMPAD_ENTER"}:
            if s.has_selection():
                s.selection_to_new()
            return self._finish(context, apply=True)
        if t == "ESC":
            if s.has_selection():
                s.clear_selection()
                self._refresh()
                return {"RUNNING_MODAL"}
            return self._finish(context, apply=False)
        if t == "Z" and (event.ctrl or event.oskey):
            self.message = "undone" if s.undo() else "nothing to undo"
            self.preview, self.preview_for = None, -1
            self._refresh()
            return {"RUNNING_MODAL"}
        if t == "LEFTMOUSE":
            c = self._cell_at(context, x, y)
            if c < 0:
                return {"RUNNING_MODAL"}
            r = int(s.labels[c])
            if event.shift:
                if s.merge(s.active, r):
                    self.message = "merged"
            elif event.ctrl:
                n = s.split(r)
                self.message = f"split into {n}" if n else "no crease to split at - use clicks or lasso"
            else:
                s.click(c, positive=not event.alt)
                self.message = ""
            self.preview, self.preview_for = None, -1
            self.hover_region = int(s.labels[c])
            self._refresh()
            return {"RUNNING_MODAL"}
        if t in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and s.has_selection() and not event.ctrl:
            if not s.step_tol(up=t == "WHEELUPMOUSE"):
                self.message = "no bigger piece" if t == "WHEELUPMOUSE" else "smallest piece"
            self._refresh()
            return {"RUNNING_MODAL"}
        if t == "SPACE":
            if s.selection_to_new():
                self.message = f"new region: {s.names[s.active]}"
                self._refresh()
            return {"RUNNING_MODAL"}
        if t == "E":
            if s.selection_to_active():
                self._refresh()
            return {"RUNNING_MODAL"}
        if t == "C":
            s.clear_selection()
            self._refresh()
            return {"RUNNING_MODAL"}
        if t == "TAB":
            s.active = (s.active + (-1 if event.shift else 1)) % len(s.names)
            self._refresh()
            return {"RUNNING_MODAL"}
        if t == "M":
            r = s.toggle_mirror()
            self.message = ("mirror on" if r else "mirror off") if r is not None else "no symmetry found"
            self._refresh()
            return {"RUNNING_MODAL"}
        if t == "S" and not event.ctrl:
            n = s.select_similar()
            self.message = f"{n} similar piece(s) added" if n else "no similar pieces"
            self._refresh()
            return {"RUNNING_MODAL"}
        if t == "B":
            cur = [v["frac"] for v in s.rules.values() if v.get("kind") == "edge"]
            steps = [0.10, 0.16, 0.24]
            nxt = steps[0] if not cur else next((f for f in steps if f > cur[0] + 1e-6), None)
            if nxt is None:
                s.remove_edge_band()
                self.message = "edge strip removed"
            else:
                nm = s.add_edge_band(nxt)
                self.message = f"edge strip {int(nxt * 100)}% of half width" if nm else "no blade region"
            self._refresh()
            return {"RUNNING_MODAL"}
        if t == "F2":
            self.typing = ""
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if t == "L":
            self.lasso_mode = True
            self.message = ""
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if t == "A":
            if s.method == "SHAPE":
                used = s.auto("CREASES", s.sensitivity)
            else:
                used = s.auto("SHAPE")
            self.message = "auto: by shape" if used == "SHAPE" else "auto: at creases"
            self.preview, self.preview_for = None, -1
            self._refresh()
            return {"RUNNING_MODAL"}
        if t in {"LEFT_BRACKET", "RIGHT_BRACKET"}:
            f = 1.4 if t == "RIGHT_BRACKET" else 1 / 1.4
            s.auto("CREASES", float(np.clip(s.sensitivity * f, 0.1, 10)))
            self.message = "auto: at creases"
            self._refresh()
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


class FM_OT_regions_auto(bpy.types.Operator):
    """Split the object into material regions automatically (no clicking).
    Only material slots and face material indices change"""
    bl_idname = "fm.regions_auto"
    bl_label = "Auto Regions"
    bl_options = {"REGISTER", "UNDO"}

    method: EnumProperty(name="Method", default="SHAPE", items=[
        ("SHAPE", "By Shape", "Long objects (swords, tools, spears): split where the width "
                              "changes along the length; falls back to creases"),
        ("CREASES", "At Creases", "Any object: split where the surface folds inward")])
    sensitivity: FloatProperty(name="Sensitivity", default=1.0, min=0.1, max=10.0,
                               description="Higher finds more, smaller regions (creases method)")
    guard_parts: BoolProperty(name="Guard Parts", default=True,
                              description="Split an ornate guard: centre piece and repeated details")
    ferrules: BoolProperty(name="Grip Ferrules", default=True,
                           description="Metal collars at the ends of the grip as their own region")
    fuller: BoolProperty(name="Blade Fuller", default=True,
                         description="The groove down the middle of the blade as its own region")
    hidden: BoolProperty(name="Buried Faces Follow Cover", default=True,
                         description="Faces hidden inside another part take that part's region")
    edge: BoolProperty(name="Blade Edge Strip", default=False,
                       description="Also make the sharpened edge of the blade its own region")
    edge_width: FloatProperty(name="Edge Width", default=0.14, min=0.03, max=0.5, subtype="FACTOR",
                              description="Edge strip width, as a fraction of the blade's half width")

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and context.mode == "OBJECT"

    def execute(self, context):
        obj = context.active_object
        C = get_cells(obj)
        sess = Session(obj, C)
        sess.sensitivity = self.sensitivity
        details = tuple(k for k, on in (("guard", self.guard_parts), ("ferrule", self.ferrules),
                                         ("fuller", self.fuller), ("hidden", self.hidden)) if on)
        used = sess.auto(self.method, details=details)
        if self.edge:
            sess.add_edge_band(self.edge_width)
        target, created = session_apply(sess)
        obj.active_material_index = target[0]
        how = "by shape" if used == "SHAPE" else "at creases"
        self.report({"INFO"}, f"{len(sess.names)} regions {how}: " + ", ".join(sess.names))
        return {"FINISHED"}


# region name word -> (library category to try first, procedural fallback)
SUGGEST = [
    (("edge",), None, "SILVER"),
    (("fuller", "groove"), None, "DARK_IRON"),
    (("centre", "center", "gem", "orb"), None, "SILVER"),
    (("blade", "body", "tip", "spike", "point", "head"), None, "POLISHED_STEEL"),
    (("grip", "handle", "hilt", "wrap", "strap"), "LEATHER", "RED_CLOTH"),
    (("guard", "pommel", "cap", "crown", "ring", "collar", "ferrule"), None, "GOLD"),
]


def suggest_for(name):
    low = name.lower()
    for words, cat, preset in SUGGEST:
        if any(w in low for w in words):
            return cat, preset
    return None, "DARK_IRON"


class FM_OT_regions_suggest(bpy.types.Operator):
    """Give every region a fitting Forge material by its name (blade: steel,
    grip: leather, guard and pommel: gold, edge: bright steel). Regions you
    already changed are left alone; replaced materials are kept"""
    bl_idname = "fm.regions_suggest"
    bl_label = "Suggest Materials"
    bl_options = {"REGISTER", "UNDO"}

    overwrite: BoolProperty(name="Also Replace Changed Regions", default=False)

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and len(o.material_slots) > 1

    def execute(self, context):
        from . import library, nodes, procedural
        obj = context.active_object
        names = list(obj.get("fm_region_names", []))
        mi = np.empty(len(obj.data.polygons), np.int32)
        obj.data.polygons.foreach_get("material_index", mi)
        used = set(np.unique(mi).tolist())
        done, cache = [], {}
        active = obj.active_material_index
        nodes.prepare_object(obj)
        for i, slot in enumerate(obj.material_slots):
            if i not in used:
                continue
            nm = names[i] if i < len(names) and names[i] else (slot.material.name if slot.material else "")
            m = slot.material
            untouched = m is None or "fm_region_base" in m or not m.get("fm_kind")
            if not (untouched or self.overwrite):
                continue
            cat, preset = suggest_for(nm)
            key = cat or preset
            if key not in cache:
                mat = None
                if cat:
                    item = next((it for it in library.items() if it.get("category") == cat), None)
                    if item:
                        try:
                            mat = nodes.build_texture_material(item, "BOX", 1.0)
                        except RuntimeError:
                            mat = None
                if mat is None:
                    mat = procedural.build(preset)
                cache[key] = mat
            obj.active_material_index = i
            nodes.assign_material(obj, cache[key])
            done.append(f"{nm}: {cache[key].name}")
        obj.active_material_index = active
        if not done:
            self.report({"INFO"}, "Nothing to change (every region already has its own material)")
            return {"CANCELLED"}
        self.report({"INFO"}, "; ".join(done))
        return {"FINISHED"}


class FM_OT_region_activate(bpy.types.Operator):
    """Make this region (material slot) the one the dial and panels edit"""
    bl_idname = "fm.region_activate"
    bl_label = "Activate Region"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty()

    def execute(self, context):
        obj = context.active_object
        if obj and 0 <= self.index < len(obj.material_slots):
            obj.active_material_index = self.index
        return {"FINISHED"}


classes = (FM_OT_regions, FM_OT_regions_auto, FM_OT_regions_suggest, FM_OT_region_activate)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    _STATE["kill"] = True
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    _CACHE.clear()
