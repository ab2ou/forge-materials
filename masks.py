# SPDX-License-Identifier: GPL-3.0-or-later
"""Region masks in texture space.

Material regions are normally faces with a material slot each, so a border can
only be as fine as the faces. For the final textures the regions are instead
turned into soft per-texel weights, computed from each texel's 3D position:
borders then run through faces at texel precision, without steps, and the
blade edge / fuller strips are measured exactly. Pure numpy (no bpy), so the
background bake process can use it.
"""

import json

import numpy as np


def slot_labels(C, mesh_material_index, nslots):
    """Per-cell region (= material slot) by area vote, and the used slots."""
    mi = np.clip(np.asarray(mesh_material_index), 0, max(nslots - 1, 0))
    used = np.unique(mi)
    votes = np.zeros((C.n, max(nslots, 1)))
    np.add.at(votes, (C.face_cell, mi), C._face_area)
    votes[:, np.setdiff1d(np.arange(max(nslots, 1)), used)] = -1.0
    return votes.argmax(1), used


def parse_rules(obj_rules_json, slot_names):
    """Rules stored on the object ({region name: rule}) -> {slot: (kind, parent slot, frac)}."""
    try:
        rules = json.loads(obj_rules_json or "{}")
    except (ValueError, TypeError):
        return {}
    idx = {n: i for i, n in enumerate(slot_names) if n}
    out = {}
    for nm, r in rules.items():
        if nm in idx and r.get("parent") in idx:
            out[idx[nm]] = (r.get("kind"), idx[r["parent"]], float(r.get("frac", 0.0)))
    return out


def texel_cells(C, cell_float, pos):
    """Cell of each texel: the baked cell index where it is a clean integer,
    else the nearest cell to the texel's position."""
    c = np.rint(cell_float).astype(np.int64)
    ok = (np.abs(cell_float - c) < 1e-3) & (c >= 0) & (c < C.n)
    if (~ok).any():
        near = C._nearest(pos[~ok], 3.0 * C.cell)
        c[~ok] = near
    return c


def region_weights(C, labels, regions, rules, pos, nor, tcell, texel, sigma=0.8, normal_k=4.0):
    """Soft region weights per texel.

    labels: region (slot) per cell; regions: list of region (slot) ids, column
    order of the result; rules: {slot: (kind, parent, frac)}; pos/nor: texel
    positions and normals (object space); tcell: texel cells (-1 = none);
    texel: world size of one texel (sets how sharp the anti-aliased border is).
    """
    R = len(regions)
    col = {r: k for k, r in enumerate(regions)}
    N = len(pos)
    W = np.zeros((N, R), np.float32)
    # the smooth vote runs on the base regions (strips folded into their parent)
    base = labels.copy()
    for s, (kind, parent, _) in rules.items():
        base[base == s] = parent
    i, j = C.edges[:, 0], C.edges[:, 1]
    border = np.zeros(C.n, bool)
    diff = base[i] != base[j]
    border[i[diff]] = True
    border[j[diff]] = True
    L = int(max(labels.max(), max(regions))) + 1
    lut = -np.ones(L, np.int64)
    for r, k in col.items():
        lut[r] = k
    valid = tcell >= 0
    inner = valid & ~border[np.maximum(tcell, 0)]
    ii = np.nonzero(inner)[0]
    if len(ii):
        W[ii, lut[base[tcell[ii]]]] = 1.0
    bt = np.nonzero(valid & ~inner)[0]
    if len(bt):
        s2 = (sigma * C.cell) ** 2
        tau = float(np.clip(texel / (2.5 * sigma * C.cell), 0.01, 0.3))
        ip, adj = C.indptr, C.adj
        for ch in range(0, len(bt), 300_000):
            t = bt[ch:ch + 300_000]
            cell = tcell[t]
            deg = ip[cell + 1] - ip[cell]
            rep = np.repeat(np.arange(len(t)), deg + 1)
            starts = np.repeat(ip[cell], deg + 1)
            off = np.arange(len(rep)) - np.repeat(np.cumsum(deg + 1) - (deg + 1), deg + 1)
            cand = np.where(off == 0, np.repeat(cell, deg + 1), adj[np.minimum(starts + off - 1, len(adj) - 1)])
            d = pos[t][rep] - C.pos[cand]
            d2 = np.einsum("ij,ij->i", d, d)
            facing = np.einsum("ij,ij->i", nor[t][rep], C.nor[cand])
            w = C.area[cand] * np.exp(-d2 / s2) * np.exp(-normal_k * (1.0 - facing))
            sc = np.zeros((len(t), L))
            np.add.at(sc, (rep, base[cand]), w)
            sc /= np.maximum(sc.sum(1, keepdims=True), 1e-30)
            # soft arg-max: a crisp border, anti-aliased over about a texel
            ex = np.exp((sc - sc.max(1, keepdims=True)) / tau)
            ex /= ex.sum(1, keepdims=True)
            for lab in np.unique(base):
                if lab in col:
                    W[t, col[int(lab)]] = ex[:, lab]
    # strips inside their parent, measured per texel
    def soft_inside(dist):
        return np.clip(0.5 - dist / max(texel, 1e-9), 0.0, 1.0).astype(np.float32)

    order = sorted(rules.items(), key=lambda kv: 0 if kv[1][0] == "edge" else 1)
    for s, (kind, parent, frac) in order:
        if s not in col or parent not in col:
            continue
        pc = col[parent]
        on = np.nonzero(W[:, pc] > 0)[0]
        if not len(on):
            continue
        group = [parent, s] + [k for k, v in rules.items() if v[1] == parent]
        cells = np.nonzero(np.isin(labels, group))[0]
        if kind == "edge":
            spacing = float(np.sqrt(np.median(C._face_area))) * 1.2
            frame = C.edge_frame(cells, outline_points=C._fcen.astype(np.float64), spacing=spacing)
            if "grid" not in frame:
                continue
            m = soft_inside(C.edge_dist(frame, pos[on], frac))
        elif kind == "band":
            fcells = np.nonzero(labels == s)[0]
            if len(fcells) < 10:
                continue
            frame = C.edge_frame(cells)
            prof = C.band_profile(frame, fcells)
            if prof is None:
                continue
            m = soft_inside(C.band_dist(frame, prof, pos[on]))
        else:
            continue
        take = W[on, pc] * m
        W[on, col[s]] += take
        W[on, pc] -= take
    tot = W.sum(1, keepdims=True)
    W = np.where(tot > 0, W / np.maximum(tot, 1e-12), W)
    return W


def despeckle(W, cov, res, rounds=2):
    """Single texels whose region disagrees with all their (agreeing)
    neighbours in the texture are noise from a texel straddling two UV
    islands: give them their neighbours' region."""
    if W.shape[1] < 2:
        return W
    lab = -np.ones(res * res, np.int64)
    idx = np.nonzero(cov)[0]
    lab[idx] = W.argmax(1)
    for _ in range(rounds):
        img = lab.reshape(res, res)
        pad = np.pad(img, 1, constant_values=-1)
        nb = np.stack([pad[1 + dy:res + 1 + dy, 1 + dx:res + 1 + dx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)], -1)
        valid = nb >= 0
        nvalid = valid.sum(-1)
        first = np.where(valid, nb, np.iinfo(np.int64).max).min(-1)
        unanimous = (nvalid >= 5) & np.all(~valid | (nb == first[..., None]), -1)
        fix = (img >= 0) & unanimous & (img != first)
        if not fix.any():
            break
        flat = fix.ravel()
        img2 = img.copy()
        img2[fix] = first[fix]
        lab = img2.ravel()
        # rows of W for the fixed texels: one-hot of the neighbours' region
        pos_in_w = -np.ones(res * res, np.int64)
        pos_in_w[idx] = np.arange(len(idx))
        rows = pos_in_w[np.nonzero(flat)[0]]
        W[rows] = 0.0
        W[rows, lab[np.nonzero(flat)[0]]] = 1.0
    return W
