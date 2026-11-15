# SPDX-License-Identifier: GPL-3.0-or-later
"""AI Hybrid: the shape split decides the main parts and every border; the
image segmenter (SAM 2, run outside Blender) only proposes sub-parts inside a
part. A proposal is kept when it is backed by a real crease, is not a mirror
copy of another proposal, and is big enough. Every cell gets a label (unseen
cells are flooded from the seen ones, borders snap to the strongest crease)."""
import numpy as np


def _cc(C, mask):
    """connected pieces of a cell mask (-1 outside)"""
    i, j = C.edges[:, 0], C.edges[:, 1]
    keep = mask[i] & mask[j]
    ei, ej = i[keep], j[keep]
    lab = np.arange(C.n)
    for _ in range(1000):
        m = np.minimum(lab[ei], lab[ej])
        new = lab.copy()
        np.minimum.at(new, ei, m)
        np.minimum.at(new, ej, m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    lab[~mask] = -1
    return lab


def _absorb_crumbs(C, sub, P, min_area):
    """pieces smaller than min_area join the label they share most border with"""
    i, j = C.edges[:, 0], C.edges[:, 1]
    for _ in range(8):
        crumbs = np.zeros(C.n, bool)
        for l in np.unique(sub[P]):
            pcs = _cc(C, sub == l)
            m = pcs >= 0
            pid, inv = np.unique(pcs[m], return_inverse=True)
            pa_ = np.bincount(inv, weights=C.area[m])
            crumbs |= np.isin(pcs, pid[pa_ < min_area]) & m
        if not crumbs.any():
            break
        pcs = _cc(C, crumbs)
        for p in np.unique(pcs[pcs >= 0]):
            pm = pcs == p
            e = (pm[i] & ~pm[j] & P[j]) | (pm[j] & ~pm[i] & P[i])
            nb = np.where(pm[i[e]], sub[j[e]], sub[i[e]])
            if len(nb):
                sub[pm] = np.bincount(nb).argmax()
    return sub


class _Pts:
    """just enough of a Cells object for Cells._nearest on a subset"""
    def __init__(self, pos, cell):
        self.pos, self.cell = pos, cell


def _part_mirrors(C, P, min_score=0.6):
    """Mirror planes of one part (through its own centre, across its own main
    axes): cell maps (-1 = no partner)."""
    cells = np.nonzero(P)[0]
    p, a, nr = C.pos[cells], C.area[cells], C.nor[cells]
    mu = (p * a[:, None]).sum(0) / a.sum()
    q0 = p - mu
    _, vecs = np.linalg.eigh((q0 * a[:, None]).T @ q0)
    out = []
    for ax in vecs.T:
        d = q0 @ ax
        q = p - 2.0 * d[:, None] * ax[None, :]
        qn = nr - 2.0 * (nr @ ax)[:, None] * ax[None, :]
        # partner = nearest cell facing the mirrored way (a thin part's front
        # must find its back, not itself)
        mp = -np.ones(len(cells), np.int64)
        side = nr @ ax
        qside = qn @ ax
        for sgn in (1.0, -1.0):
            cand = np.nonzero(side * sgn >= -0.05)[0]
            want = np.nonzero(qside * sgn > 0.05)[0]
            if not len(cand) or not len(want):
                continue
            sub_c = _Pts(p[cand], C.cell)
            got = type(C)._nearest(sub_c, q[want], 2.0 * C.cell)
            hit = got >= 0
            mp[want[hit]] = cells[cand[got[hit]]]
        flat = np.abs(qside) <= 0.05                # in the plane: plain nearest
        if flat.any():
            got = C._nearest(q[flat], 2.0 * C.cell)
            mp[np.nonzero(flat)[0]] = got
        ok = mp >= 0
        ok[ok] &= P[mp[ok]]
        ok[ok] &= np.einsum("ij,ij->i", C.nor[mp[ok]], qn[ok]) > 0.5
        if a[ok].sum() >= min_score * a.sum():
            m = -np.ones(C.n, np.int64)
            m[cells[ok]] = mp[ok]
            out.append(m)
    return out


def _mirror(mask, maps):
    out = mask.copy()
    for _ in range(2):
        for mp in maps:
            ok = mp >= 0
            pulled = np.zeros_like(out)
            pulled[ok] = out[mp[ok]]
            out |= pulled
    return out


def _pieces_sig(C, mask):
    pcs = _cc(C, mask)
    ids = np.unique(pcs[pcs >= 0])
    out = []
    for p in ids.tolist():
        m = pcs == p
        A, ext = C.mask_signature(m)
        out.append((A, ext))
    return out


def _copies(C, ma, mb, tol=0.3):
    """every piece of mb (by area) has a piece of ma shaped like it
    (area and extents, so rotated or mirrored copies match)"""
    sa, sb = _pieces_sig(C, ma), _pieces_sig(C, mb)
    lo, hi = 1.0 - tol, 1.0 / (1.0 - tol)
    tot = sum(A for A, _ in sb)
    hit = 0.0
    for A, ext in sb:
        for A2, ext2 in sa:
            if A2 * lo <= A <= A2 * hi and np.all((ext >= ext2 * lo - 0.25 * C.cell) &
                                                   (ext <= ext2 * hi + 0.25 * C.cell)):
                hit += A
                break
    return tot > 0 and hit >= 0.8 * tot


def _border_strength(C, sub, a, b):
    """how much of a real crease separates sub-labels a and b (0 = none)"""
    i, j = C.edges[:, 0], C.edges[:, 1]
    e = ((sub[i] == a) & (sub[j] == b)) | ((sub[i] == b) & (sub[j] == a))
    if not e.any():
        return None, 0
    s = np.maximum(C.wall[i[e]], C.wall[j[e]])
    return float(np.median(s)), int(e.sum())


def refine_part(C, P, face_sam, min_frac=0.04, crease=0.35, max_sub=6, purity=0.5, log=None):
    """Sub-labels (0..k-1, -1 outside) of the cells in mask P, or None."""
    log = log or (lambda m: None)
    fc, fw = C.face_cell, C._face_area
    pa = C.area[P].sum()
    fsel = P[fc] & (face_sam >= 0)
    if fsel.sum() == 0:
        return None
    u, inv = np.unique(face_sam[fsel], return_inverse=True)
    K = len(u)
    cells = np.nonzero(P)[0]
    loc = -np.ones(C.n, np.int64)
    loc[cells] = np.arange(len(cells))
    vote = np.zeros((len(cells), K))
    np.add.at(vote, (loc[fc[fsel]], inv), fw[fsel])
    # proposals smaller than min_frac of the part are noise
    big = vote.sum(0) >= min_frac * pa
    vote[:, ~big] = 0.0
    tot = vote.sum(1)
    cov = np.bincount(loc[fc[P[fc]]], weights=fw[P[fc]], minlength=len(cells))
    best = vote.argmax(1)
    conf = np.where(tot > 0, vote.max(1) / np.maximum(tot, 1e-12), 0)
    # seeds: well seen, one clear proposal, away from creases. Carved
    # proposals have creases everywhere, so each one keeps its own flattest
    # 40% as seeds
    ok = (tot >= 0.25 * cov) & (conf >= 0.7)
    wl = C.wall[cells]
    good = ok & (wl < crease)
    for k in np.unique(best[ok]).tolist():
        m = ok & (best == k)
        if m.sum() >= 5:
            good |= m & (wl <= np.percentile(wl[m], 40))
    seeds_c, seeds_l = cells[good], best[good]
    mirrors = _part_mirrors(C, P)
    gmap = np.arange(K)
    for it in range(40):
        if len(np.unique(seeds_l)) < 2:
            return None
        _, _, lab = C.bottleneck_labeled(list(zip(seeds_c.tolist(), seeds_l.tolist())), allowed=P)
        sub = np.where(P, lab, -1)
        if (P & (sub < 0)).any():        # cut off from every seed: main proposal
            sub[P & (sub < 0)] = np.bincount(sub[P & (sub >= 0)], weights=C.area[P & (sub >= 0)]).argmax()
        ids = np.unique(sub[P])
        areas = {int(l): float(C.area[sub == l].sum()) for l in ids}
        drop = None
        # 1) a proposal that only survives as crumbs
        for l in ids:
            if areas[l] < min_frac * pa:
                drop = ("small", sub == l)
                break
        # 2) a border with no crease behind it (lighting, not shape)
        if drop is None:
            weakest = None
            for a in ids:
                for b in ids:
                    if a >= b:
                        continue
                    s, n = _border_strength(C, sub, a, b)
                    if s is not None and s < crease and (weakest is None or s < weakest[0]):
                        weakest = (s, a, b)
            if weakest:
                s, a, b = weakest
                small = a if areas[a] < areas[b] else b
                drop = (f"weak border {s:.2f}", sub == small)
        # 3) mirror copies of each other (left/right arm, the two blade bevels)
        if drop is None and mirrors:
            # a symmetric part has symmetric sub-parts: a proposal whose mirror
            # image lands mostly on other proposals is half of something
            # (one arm, one bevel) and joins what its mirror image lands on
            for b_ in sorted(ids, key=lambda l: areas[l]):
                mb = sub == b_
                M = _mirror(mb, mirrors) & P
                if C.area[M & mb].sum() >= 0.6 * C.area[M].sum():
                    continue
                rest = M & ~mb
                tgt = np.bincount(sub[rest], weights=C.area[rest]).argmax()
                drop = ("mirror", mb)
                merge_to, merge_from = int(tgt), int(b_)
                break
            if drop:
                bsel = drop[1][seeds_c]
                gmap[gmap == merge_from] = merge_to
                seeds_l = np.where(bsel, merge_to, seeds_l)
                log(f"  merge {drop[0]}")
                continue
        # 3b) same shape (repeated details: both hooks, all the spikes)
        if drop is None:
            for a in ids:
                for b in ids:
                    if b == a or areas[b] > areas[a]:
                        continue
                    if _copies(C, sub == a, sub == b):
                        bsel = (sub == b)[seeds_c]
                        gmap[gmap == b] = a
                        seeds_l = np.where(bsel, a, seeds_l)
                        drop = "copy"
                        break
                if drop:
                    break
            if drop:
                log("  merge copy")
                continue
        # 4) the segmenter must agree with itself: most of what was seen of a
        #    sub-part comes from its own proposal (else it is lighting/noise)
        if drop is None:
            worst = None
            main = max(ids, key=lambda l: areas[l])
            sub_loc = sub[cells]
            gtot = vote.sum(0)
            for l in ids:
                if l == main:
                    continue
                v = vote[sub_loc == l]
                tot_v = v.sum()
                if tot_v <= 0:
                    continue
                vin = v.sum(0)
                mine = vin >= 0.6 * np.maximum(gtot, 1e-12)   # groups that live in l
                pur = vin[mine].sum() / tot_v
                if pur < purity and (worst is None or pur < worst[0]):
                    worst = (pur, l)
            if worst:
                drop = (f"unsure {worst[0]:.2f}", sub == worst[1])
        if drop is None and len(ids) > max_sub:
            sm = min(ids, key=lambda l: areas[l])
            drop = ("too many", sub == sm)
        if drop is None:
            break
        log(f"  drop {drop[0]}")
        keep = ~drop[1][seeds_c]
        seeds_c, seeds_l = seeds_c[keep], seeds_l[keep]
    sub = _absorb_crumbs(C, sub, P, 0.5 * min_frac * pa)
    ids = np.unique(sub[P])
    if len(ids) < 2:
        return None
    remap = -np.ones(int(ids.max()) + 1, np.int64)
    order = sorted(ids.tolist(), key=lambda l: -C.area[sub == l].sum())
    remap[order] = np.arange(len(order))
    out = -np.ones(C.n, np.int64)
    out[P] = remap[sub[P]]
    return out


def hybrid_labels(C, labels, names, rules, face_sam, log=None, **kw):
    """Refine shape labels with SAM proposals. Strip regions (rules: edge,
    band) and their borders stay exactly as the shape split made them."""
    log = log or (lambda m: None)
    labels = labels.copy()
    names = list(names)
    strip = {n for n in rules}
    for p in range(len(names)):
        if names[p] in strip:
            continue
        P = labels == p
        if C.area[P].sum() < 0.01 * C.total_area:
            continue
        log(names[p])
        sub = refine_part(C, P, face_sam, log=log, **kw)
        if sub is None:
            continue
        for k in range(1, int(sub.max()) + 1):
            labels[sub == k] = len(names)
            names.append(f"{names[p]} {k + 1}")
        log(f"  -> {int(sub.max()) + 1} parts")
    return labels, names


def part_faces(C, labels, names, rules):
    """Part id per face for the renders: strip regions (edge, band) count as
    their parent, so a part is rendered whole."""
    lab = labels.copy()
    idx = {n: i for i, n in enumerate(names)}
    for n, r in rules.items():
        if n in idx and r.get("parent") in idx:
            lab[lab == idx[n]] = idx[r["parent"]]
    return lab[C.face_cell].astype(np.int32)
