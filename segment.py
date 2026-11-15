# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometry analysis for Smart Regions (pure numpy + bpy reads, nothing written).

The mesh is summarised as surface "cells" (faces binned by position and normal
direction), so it works the same on clean meshes and on AI-generated meshes
made of thousands of loose pieces: neighbourhood comes from proximity in space,
not from mesh connectivity.

Boundaries between parts are found where the surface turns concave (the
"minima rule"): convex bends (a blade's edge, a sphere) stay inside one part,
concave creases (where a guard meets a blade) separate parts.
"""

import heapq
import math

import numpy as np


class Cells:
    """Faces grouped into cells plus a proximity graph between cells."""

    def __init__(self, mesh, target_cells=24000, colors=None):
        n = len(mesh.polygons)
        self.n_faces = n
        cen = np.empty(n * 3, np.float64)
        nor = np.empty(n * 3, np.float64)
        area = np.empty(n, np.float64)
        mesh.polygons.foreach_get("center", cen)
        mesh.polygons.foreach_get("normal", nor)
        mesh.polygons.foreach_get("area", area)
        self._face_area = np.maximum(area.copy(), 1e-12)
        cen = cen.reshape(-1, 3)
        nor = nor.reshape(-1, 3)
        self._fcen = cen.astype(np.float32)
        self._fnor = nor.astype(np.float32)
        area = np.maximum(area, 1e-12)
        self.total_area = float(area.sum())

        lo, hi = cen.min(0), cen.max(0)
        self.size = float(np.linalg.norm(hi - lo)) or 1.0
        cell = math.sqrt(self.total_area / max(target_cells, 100)) * 1.15
        cell = max(cell, self.size * 1e-4)
        self.cell = cell
        vox = np.floor((cen - lo) / cell).astype(np.int64)
        dom = np.argmax(np.abs(nor), axis=1)
        sign = (nor[np.arange(n), dom] > 0).astype(np.int64)
        bucket = dom * 2 + sign                       # 6 normal directions
        dims = vox.max(0) + 2
        vkey = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
        key = vkey * 6 + bucket
        ukey, face_cell = np.unique(key, return_inverse=True)
        self.face_cell = face_cell.astype(np.int64)
        C = len(ukey)
        self.n = C
        w = area
        self.area = np.bincount(face_cell, weights=w, minlength=C)
        pos = np.stack([np.bincount(face_cell, weights=cen[:, i] * w, minlength=C) for i in range(3)], 1)
        self.pos = pos / self.area[:, None]
        nn = np.stack([np.bincount(face_cell, weights=nor[:, i] * w, minlength=C) for i in range(3)], 1)
        ln = np.linalg.norm(nn, axis=1)
        self.nor = nn / np.maximum(ln, 1e-12)[:, None]
        self.col = None
        if colors is not None:
            cc = np.stack([np.bincount(face_cell, weights=colors[:, i] * w, minlength=C) for i in range(3)], 1)
            self.col = cc / self.area[:, None]

        # ---- proximity graph: cells in the same or neighbouring voxels ----
        cvkey = ukey // 6
        cbucket = ukey % 6
        uv, vox_of_cell = np.unique(cvkey, return_inverse=True)
        table = -np.ones((len(uv), 6), np.int64)
        table[vox_of_cell, cbucket] = np.arange(C)
        vx = uv // (dims[1] * dims[2])
        vy = (uv // dims[2]) % dims[1]
        vz = uv % dims[2]
        pairs = []
        # same voxel, different buckets
        for a in range(6):
            for b in range(a + 1, 6):
                m = (table[:, a] >= 0) & (table[:, b] >= 0)
                if m.any():
                    pairs.append(np.stack([table[m, a], table[m, b]], 1))
        # 13 forward neighbour offsets (each unordered voxel pair once)
        offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                if (dx, dy, dz) > (0, 0, 0)]
        for dx, dy, dz in offs:
            nk = ((vx + dx) * dims[1] + (vy + dy)) * dims[2] + (vz + dz)
            idx = np.searchsorted(uv, nk)
            idx = np.minimum(idx, len(uv) - 1)
            ok = uv[idx] == nk
            if not ok.any():
                continue
            src = np.nonzero(ok)[0]
            dst = idx[ok]
            for a in range(6):
                ca = table[src, a]
                for b in range(6):
                    cb = table[dst, b]
                    m = (ca >= 0) & (cb >= 0)
                    if m.any():
                        pairs.append(np.stack([ca[m], cb[m]], 1))
        # mesh connectivity too: faces sharing a vertex are neighbours. Low-poly
        # meshes (big faces far apart) need this; triangle soups get nothing
        # from it and rely on the proximity pairs above.
        topo = _vertex_neighbours(mesh, n)
        self._ftopo = topo.astype(np.int32) if topo is not None else None
        if topo is not None and len(topo):
            tc = self.face_cell[topo]
            tc = tc[tc[:, 0] != tc[:, 1]]
            if len(tc):
                pairs.append(tc)
        E = np.concatenate(pairs, 0) if pairs else np.zeros((0, 2), np.int64)
        if len(E):
            E = np.sort(E, axis=1)
            E = np.unique(E[:, 0] * C + E[:, 1])
            E = np.stack([E // C, E % C], 1)
        self.edges = E
        i, j = E[:, 0], E[:, 1]
        d = self.pos[j] - self.pos[i]
        dist = np.linalg.norm(d, axis=1)
        dist = np.maximum(dist, 1e-9)
        dh = d / dist[:, None]
        ni, nj = self.nor[i], self.nor[j]
        conv = np.einsum("ij,ij->i", nj - ni, dh)        # >0 convex, <0 concave
        concav = np.clip(-conv, 0.0, 2.0)
        dot = np.einsum("ij,ij->i", ni, nj)
        crease = 1.0 - dot
        # parallel surfaces stacked on top of each other (a blade continuing
        # inside a guard, a strap on a box): same direction but offset along it
        # (on a curved surface neighbours sit slightly "above" each other too, by
        # about half the bend angle, so only the excess over that counts)
        sin_t = np.sqrt(np.clip(1.0 - dot * dot, 0.0, 1.0))
        off = np.abs(np.einsum("ij,ij->i", dh, ni)) - (0.6 * sin_t + 0.1)
        stacked = np.clip((dot - 0.85) / 0.15, 0.0, 1.0) * np.clip(off / 0.4, 0.0, 1.0)
        # opposite-facing surfaces touching side by side (two parts pressed together)
        facing = np.clip((-dot - 0.85) / 0.15, 0.0, 1.0) * (1.0 - np.abs(np.einsum("ij,ij->i", dh, ni)))
        self.concav = concav
        self.crease = crease
        self.stacked = stacked + facing
        self.dist = dist
        self.color_diff = None
        if self.col is not None:
            self.color_diff = np.linalg.norm(self.col[j] - self.col[i], axis=1)
        self.set_weights(0.0)

    # ------------------------------------------------------------------ costs
    def set_colors(self, face_colors):
        """Add (or replace) per-face colours as an optional boundary cue."""
        fc = self.face_cell
        w = self.area
        cc = np.stack([np.bincount(fc, weights=face_colors[:, k] * self._face_area, minlength=self.n)
                       for k in range(3)], 1)
        self.col = cc / np.maximum(w, 1e-12)[:, None]
        i, j = self.edges[:, 0], self.edges[:, 1]
        self.color_diff = np.linalg.norm(self.col[j] - self.col[i], axis=1)

    def _wall_score(self):
        """Per-cell boundary strength: the strongest concave crease touching the
        cell, smoothed so a wall has no gaps a few noisy edges could leak through."""
        C = self.n
        i, j = self.edges[:, 0], self.edges[:, 1]
        bw = self.concav + 1.5 * self.stacked
        score = np.zeros(C)
        np.maximum.at(score, i, bw)
        np.maximum.at(score, j, bw)
        deg = np.bincount(i, minlength=C) + np.bincount(j, minlength=C)
        for _ in range(2):
            acc = score.copy()
            np.add.at(acc, i, score[j])
            np.add.at(acc, j, score[i])
            score = np.maximum(score, acc / (deg + 1))
        return score

    def set_weights(self, color_weight=0.0):
        self.wall = self._wall_score()
        w = self.concav + 0.01 * self.crease + 1.5 * self.stacked
        if self.color_diff is not None and color_weight > 0:
            w = w + color_weight * self.color_diff
        self.w = w
        C = self.n
        i, j = self.edges[:, 0], self.edges[:, 1]
        order = np.argsort(np.concatenate([i, j]), kind="stable")
        self.indptr = np.searchsorted(np.concatenate([i, j])[order], np.arange(C + 1))
        self.adj = np.concatenate([j, i])[order]
        self._order = order
        # "toll": length plus a fixed toll per concave crease (part splitting);
        # "crease": mostly crease cost, so a click floods a whole part however big
        # (directed, in CSR order). "crease" climbs the wall score: going up a
        # crease wall costs, coming down is free, so crossing a wall costs its
        # height once and a click that lands on a wall still floods its side
        toll = self.dist + w * (0.05 * self.size)
        src = np.concatenate([i, j])[order]
        base = np.concatenate([w, w])[order] + 0.02 * np.concatenate([self.dist, self.dist])[order] / self.size
        climb = np.maximum(self.wall[self.adj] - self.wall[src], 0.0)
        self._costs = {"toll": np.concatenate([toll, toll])[order], "crease": base + climb}
        self._adj_lists = {}

    def _lists(self, mode="toll"):
        if mode not in self._adj_lists:
            c2 = self._costs[mode]
            ip, a, c = self.indptr.tolist(), self.adj.tolist(), c2.tolist()
            self._adj_lists[mode] = [list(zip(a[ip[k]:ip[k + 1]], c[ip[k]:ip[k + 1]]))
                                     for k in range(self.n)]
        return self._adj_lists[mode]

    # ------------------------------------------------------------- geodesics
    def bottleneck(self, seeds, limit=None):
        """For each cell: the lowest crease "wall" a path from the seeds has to
        climb over (b), tie-broken by path length (l). Cells with b <= t are the
        piece of surface you can reach without crossing a crease higher than t."""
        if not hasattr(self, "_bn_lists"):
            ecost = np.maximum(self.wall[self.adj], np.concatenate([self.w, self.w])[self._order])
            elen = np.concatenate([self.dist, self.dist])[self._order]
            ip, a, c, ln = self.indptr.tolist(), self.adj.tolist(), ecost.tolist(), elen.tolist()
            self._bn_lists = [list(zip(a[ip[k]:ip[k + 1]], c[ip[k]:ip[k + 1]], ln[ip[k]:ip[k + 1]]))
                              for k in range(self.n)]
        adj = self._bn_lists
        inf = float("inf")
        lim = inf if limit is None else float(limit)
        B = [inf] * self.n
        Ls = [inf] * self.n
        h = []
        for s in seeds:
            B[s], Ls[s] = 0.0, 0.0
            h.append((0.0, 0.0, s))
        heapq.heapify(h)
        pop, push = heapq.heappop, heapq.heappush
        while h:
            b, l, u = pop(h)
            if b > B[u] or (b == B[u] and l > Ls[u]):
                continue
            for v, c, ln in adj[u]:
                nb = b if b >= c else c
                if nb > lim:
                    continue
                nl = l + ln
                if nb < B[v] or (nb == B[v] and nl < Ls[v]):
                    B[v], Ls[v] = nb, nl
                    push(h, (nb, nl, v))
        return np.array(B), np.array(Ls)

    def bottleneck_labeled(self, seeds, allowed=None):
        """Multi-source bottleneck flood: seeds are (cell, label) pairs; every
        cell takes the label it can reach over the lowest crease (ties: the
        shorter path). `allowed` (bool per cell) limits the flood to a region."""
        self.bottleneck([], limit=0)            # builds the adjacency lists
        adj = self._bn_lists
        inf = float("inf")
        B = [inf] * self.n
        Ls = [inf] * self.n
        lab = [-1] * self.n
        ok = allowed.tolist() if allowed is not None else None
        h = []
        for c, l in seeds:
            if B[c] > 0.0 or lab[c] < 0:
                B[c], Ls[c], lab[c] = 0.0, 0.0, int(l)
                h.append((0.0, 0.0, c))
        heapq.heapify(h)
        pop, push = heapq.heappop, heapq.heappush
        while h:
            b, l, u = pop(h)
            if b > B[u] or (b == B[u] and l > Ls[u]):
                continue
            lu = lab[u]
            for v, c, ln in adj[u]:
                if ok is not None and not ok[v]:
                    continue
                nb = b if b >= c else c
                nl = l + ln
                if nb < B[v] or (nb == B[v] and nl < Ls[v]):
                    B[v], Ls[v], lab[v] = nb, nl, lu
                    push(h, (nb, nl, v))
        return np.array(B), np.array(Ls), np.array(lab)

    def geodesic(self, seeds, limit=None, labels=None, mode="toll"):
        """Multi-source Dijkstra. seeds: list of cell ids, or (cell, label) pairs
        when `labels` is True. Returns (dist, nearest label)."""
        adj = self._lists(mode)
        INF = float("inf")
        dist = [INF] * self.n
        lab = [-1] * self.n
        heap = []
        for s in seeds:
            c, l = (s if labels else (s, 0))
            dist[c] = 0.0
            lab[c] = l
            heap.append((0.0, c))
        heapq.heapify(heap)
        lim = INF if limit is None else limit
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, u = pop(heap)
            if d > dist[u] or d > lim:
                continue
            lu = lab[u]
            for v, c in adj[u]:
                nd = d + c
                if nd < dist[v]:
                    dist[v] = nd
                    lab[v] = lu
                    push(heap, (nd, v))
        return np.array(dist), np.array(lab)

    # -------------------------------------------------------- segmentation
    def segment(self, k=1.0, min_frac=0.01):
        """Split the surface at concave creases ("walls"), then let every cell
        join the part it reaches most cheaply. k (sensitivity): larger -> fewer,
        bigger regions (only deep creases count as walls)."""
        C = self.n
        i, j = self.edges[:, 0], self.edges[:, 1]
        score = self.wall
        thr = 0.12 * max(k, 0.05)
        wall = score > thr
        # connected components of the non-wall cells (label propagation)
        keep = ~wall[i] & ~wall[j]
        ei, ej = i[keep], j[keep]
        lab = np.arange(C)
        for _ in range(500):
            m = np.minimum(lab[ei], lab[ej])
            new = lab.copy()
            np.minimum.at(new, ei, m)
            np.minimum.at(new, ej, m)
            new = new[new]
            if np.array_equal(new, lab):
                break
            lab = new
        lab[wall] = -1
        ids, inv = np.unique(lab[lab >= 0], return_inverse=True)
        areas = np.bincount(inv, weights=self.area[lab >= 0]) if len(ids) else np.zeros(0)
        big = ids[areas >= min_frac * self.total_area]
        if len(big) == 0:
            return np.zeros(C, np.int64)
        rid = {int(r): n for n, r in enumerate(big.tolist())}
        seeds = [(c, rid[int(l)]) for c, l in enumerate(lab.tolist()) if l in rid]
        _, out = self.geodesic(seeds, labels=True)
        out[out < 0] = 0
        return _relabel(self.smooth_labels(out))

    def smooth_labels(self, labels, iters=4):
        """Clean speckles: a cell takes the label most of its neighbours agree on,
        weighted so votes across a concave crease count for little."""
        lab = np.asarray(labels).copy()
        L = int(lab.max()) + 1
        if L < 2 or not len(self.edges):
            return lab
        i, j = self.edges[:, 0], self.edges[:, 1]
        vote = 1.0 / (1.0 + 40.0 * self.w)
        for _ in range(iters):
            acc = np.zeros((self.n, L))
            np.add.at(acc, (i, lab[j]), vote)
            np.add.at(acc, (j, lab[i]), vote)
            acc[np.arange(self.n), lab] += vote.mean()  # the cell's own vote breaks ties
            new = acc.argmax(1)
            if np.array_equal(new, lab):
                break
            lab = new
        return lab

    def _merge_small(self, labels, min_frac):
        """Fold regions smaller than min_frac of the area into their best neighbour."""
        for _ in range(50):
            ids, inv = np.unique(labels, return_inverse=True)
            areas = np.bincount(inv, weights=self.area)
            small = areas < min_frac * self.total_area
            if not small.any() or len(ids) <= 1:
                break
            li, lj = inv[self.edges[:, 0]], inv[self.edges[:, 1]]
            cross = li != lj
            if not cross.any():
                break
            a, b, w = li[cross], lj[cross], self.w[cross]
            # for each small region: the neighbour with the cheapest boundary
            target = {}
            for x, y, ww in zip(np.concatenate([a, b]).tolist(), np.concatenate([b, a]).tolist(),
                                np.concatenate([w, w]).tolist()):
                if small[x] and (x not in target or ww < target[x][0]):
                    target[x] = (ww, y)
            if not target:
                break
            # merge the smallest first to avoid chains
            x = min(target, key=lambda r: areas[r])
            labels = np.where(inv == x, ids[target[x][1]], labels)
        return labels

    def face_labels(self, labels):
        """Cell labels -> face labels. Faces inside a cell whose neighbours all
        agree just take its label; faces in boundary cells pick the label of the
        nearby cell they match best (position and facing direction), so the
        border follows the surface at face precision rather than cell steps."""
        labels = np.asarray(labels)
        fl = labels[self.face_cell]
        i, j = self.edges[:, 0], self.edges[:, 1]
        diff = labels[i] != labels[j]
        if not diff.any():
            return fl
        border = np.zeros(self.n, bool)
        border[i[diff]] = True
        border[j[diff]] = True
        faces = np.nonzero(border[self.face_cell])[0]
        if not len(faces):
            return fl
        ip, adj = self.indptr, self.adj
        fc = self.face_cell[faces]
        deg = ip[fc + 1] - ip[fc]
        # candidate cells: the face's own cell plus its graph neighbours
        rep = np.repeat(np.arange(len(faces)), deg + 1)
        starts = np.repeat(ip[fc], deg + 1)
        off = np.arange(len(rep)) - np.repeat(np.cumsum(deg + 1) - (deg + 1), deg + 1)
        cand = np.where(off == 0, np.repeat(fc, deg + 1), adj[np.minimum(starts + off - 1, len(adj) - 1)])
        p = self._fcen[faces][rep]
        nrm = self._fnor[faces][rep]
        dpos = np.linalg.norm(p - self.pos[cand], axis=1) / self.cell
        dnor = 1.0 - np.einsum("ij,ij->i", nrm, self.nor[cand])
        score = dpos + 3.0 * dnor
        score[off == 0] -= 0.15                       # slight preference to stay
        order = np.lexsort((score, rep))
        first = np.ones(len(order), bool)
        first[1:] = rep[order][1:] != rep[order][:-1]
        best = cand[order][first]
        fl[faces] = labels[best]
        return fl

    # ------------------------------------------------------- pieces & symmetry
    def components(self, level):
        """Connected pieces of surface not separated by a crease higher than
        `level` (crease cells themselves get -1)."""
        C = self.n
        ok = self.wall <= level
        i, j = self.edges[:, 0], self.edges[:, 1]
        keep = ok[i] & ok[j]
        ei, ej = i[keep], j[keep]
        lab = np.arange(C)
        for _ in range(500):
            m = np.minimum(lab[ei], lab[ej])
            new = lab.copy()
            np.minimum.at(new, ei, m)
            np.minimum.at(new, ej, m)
            new = new[new]
            if np.array_equal(new, lab):
                break
            lab = new
        lab[~ok] = -1
        return lab

    def components_in(self, mask, level):
        """components(level) restricted to the cells in mask (-1 elsewhere)."""
        ok = mask & (self.wall <= level)
        i, j = self.edges[:, 0], self.edges[:, 1]
        keep = ok[i] & ok[j]
        ei, ej = i[keep], j[keep]
        lab = np.arange(self.n)
        for _ in range(500):
            m = np.minimum(lab[ei], lab[ej])
            new = lab.copy()
            np.minimum.at(new, ei, m)
            np.minimum.at(new, ej, m)
            new = new[new]
            if np.array_equal(new, lab):
                break
            lab = new
        lab[~ok] = -1
        return lab

    def pieces(self, level):
        """components(level) with the crease rims given back to the piece they
        border (as a click selection does)."""
        lab = self.components(level)
        i, j = self.edges[:, 0], self.edges[:, 1]
        soft = self.w < 0.12 * max(1.0, level / 0.35)
        for _ in range(3):
            a = soft & (lab[i] >= 0) & (lab[j] < 0)
            b = soft & (lab[j] >= 0) & (lab[i] < 0)
            if not (a.any() or b.any()):
                break
            lab[j[a]] = lab[i[a]]
            lab[i[b]] = lab[j[b]]
        return lab

    def mask_signature(self, mask):
        a = self.area[mask]
        p = self.pos[mask]
        A = float(a.sum())
        mu = (p * a[:, None]).sum(0) / A
        q = p - mu
        cov = (q * a[:, None]).T @ q / A
        return A, np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]

    def shape_signature(self, comp, ids):
        """(area, three sorted extents) per piece id, rotation invariant."""
        m = comp >= 0
        _, inv = np.unique(comp[m], return_inverse=True)
        lut = {int(c): k for k, c in enumerate(np.unique(comp[m]).tolist())}
        a = self.area[m]
        p = self.pos[m]
        K = int(inv.max()) + 1
        A = np.bincount(inv, weights=a, minlength=K)
        mu = np.stack([np.bincount(inv, weights=a * p[:, k], minlength=K) for k in range(3)], 1) / A[:, None]
        cov = np.zeros((K, 3, 3))
        for x in range(3):
            for y in range(x, 3):
                v = np.bincount(inv, weights=a * p[:, x] * p[:, y], minlength=K) / A - mu[:, x] * mu[:, y]
                cov[:, x, y] = v
                cov[:, y, x] = v
        ext = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0.0))[:, ::-1]
        out = {}
        for c in ids:
            k = lut.get(int(c))
            if k is not None:
                out[int(c)] = (float(A[k]), ext[k])
        return out, A, ext, np.array(sorted(lut, key=lut.get))

    def _nearest(self, q, radius):
        """Nearest cell to each query point within `radius` (else -1)."""
        h = max(radius, self.cell)
        lo = self.pos.min(0) - h
        key = lambda v: (v[:, 0] * 1_000_003 + v[:, 1]) * 1_000_033 + v[:, 2]
        cv = np.floor((self.pos - lo) / h).astype(np.int64)
        ck = key(cv)
        order = np.argsort(ck)
        sk = ck[order]
        qv = np.floor((q - lo) / h).astype(np.int64)
        best = np.full(len(q), np.inf)
        arg = -np.ones(len(q), np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    k = key(qv + np.array([dx, dy, dz]))
                    lo_i = np.searchsorted(sk, k, "left")
                    hi_i = np.searchsorted(sk, k, "right")
                    cnt = hi_i - lo_i
                    for r in range(int(cnt.max()) if len(cnt) else 0):
                        has = cnt > r
                        idx = np.nonzero(has)[0]
                        c = order[lo_i[idx] + r]
                        d = np.linalg.norm(self.pos[c] - q[idx], axis=1)
                        better = d < best[idx]
                        best[idx[better]] = d[better]
                        arg[idx[better]] = c[better]
        arg[best > radius] = -1
        return arg

    def mirror_maps(self, min_score=0.8):
        """Mirror planes the object is symmetric about (through its centre,
        across its main axes). Returns [(normal, map)], map[c] = mirror cell."""
        if getattr(self, "_mirrors", None) is not None:
            return self._mirrors
        mean, axes, _ = self.principal_axes()
        out = []
        for ax in axes:
            d = (self.pos - mean) @ ax
            q = self.pos - 2.0 * d[:, None] * ax[None, :]
            qn = self.nor - 2.0 * (self.nor @ ax)[:, None] * ax[None, :]
            mp = self._nearest(q, 2.0 * self.cell)
            ok = mp >= 0
            ok[ok] &= np.einsum("ij,ij->i", self.nor[mp[ok]], qn[ok]) > 0.5
            score = float(self.area[ok].sum() / self.total_area)
            if score >= min_score:
                mp[~ok] = -1
                out.append((ax, mp, score))
        self._mirrors = out
        return out

    def mirror_mask(self, mask):
        """Add the mirror image(s) of a cell mask (pull: no holes)."""
        out = mask.copy()
        for _ in range(2):
            for _, mp, _ in self.mirror_maps():
                ok = mp >= 0
                pulled = np.zeros_like(out)
                pulled[ok] = out[mp[ok]]
                out |= pulled
        return out

    # ------------------------------------------------------- blade edge
    def edge_frame(self, cells, bins=80, outline_cells=None, outline_points=None, spacing=None):
        """Width profile of a (blade) region along its own long axis, plus the
        distance to its outline. outline_cells (default: all cells) give the
        silhouette, so a fuller or other inner region never counts as edge."""
        p, a = self.pos[cells], self.area[cells]
        mu = (p * a[:, None]).sum(0) / a.sum()
        q = p - mu
        cov = (q * a[:, None]).T @ q
        vals, vecs = np.linalg.eigh(cov)
        ax_t, ax_u = vecs[:, 2], vecs[:, 1]
        t = q @ ax_t
        u = q @ ax_u
        t0, t1 = t.min(), t.max()
        b = np.clip(((t - t0) / max(t1 - t0, 1e-9) * bins).astype(int), 0, bins - 1)
        lo = np.full(bins, np.nan)
        hi = np.full(bins, np.nan)
        for k in range(bins):
            m = b == k
            if m.sum() >= 2:
                lo[k], hi[k] = np.percentile(u[m], 1), np.percentile(u[m], 99)
        good = ~np.isnan(lo)
        xs = np.arange(bins)
        lo = np.interp(xs, xs[good], lo[good])
        hi = np.interp(xs, xs[good], hi[good])
        frame = {"mu": mu, "t": ax_t, "u": ax_u, "t0": t0, "t1": t1, "bins": bins, "lo": lo, "hi": hi}
        # outline of the blade seen flat (any shape: curved, serrated); cut-outs
        # are filled so only the outer edge counts as "edge"
        half = max(float(np.median((hi - lo) * 0.5)), 1e-9)
        gap = self.cell if spacing is None else spacing
        px = max(min(gap * 0.5, half * 0.02), half * 0.005)
        close = int(np.ceil(0.8 * gap / px))            # gap between neighbouring samples
        steps = int(np.ceil(0.5 * half / px)) + close + 1   # deepest band we will ask for
        u0, t0g = u.min() - (close + 2) * px, t0 - (close + 2) * px
        W = int((t1 - t0g) / px) + 2 * close + 6
        H = int((u.max() - u0) / px) + 2 * close + 6
        if W * H > 25_000_000:
            return frame
        occ = np.zeros((W, H), bool)
        if outline_points is not None:
            qo = outline_points - mu
        else:
            oc = np.arange(self.n) if outline_cells is None else outline_cells
            qo = self.pos[oc] - mu
        to, uo = qo @ ax_t, qo @ ax_u
        keep = (to >= t0) & (to <= t1) & (uo >= u.min() - px) & (uo <= u.max() + px)
        gi = np.clip(((to[keep] - t0g) / px).astype(int), 0, W - 1)
        gj = np.clip(((uo[keep] - u0) / px).astype(int), 0, H - 1)
        occ[gi, gj] = True
        for _ in range(close):                          # close the gaps between cells
            o2 = occ.copy()
            o2[1:, :] |= occ[:-1, :]; o2[:-1, :] |= occ[1:, :]
            o2[:, 1:] |= occ[:, :-1]; o2[:, :-1] |= occ[:, 1:]
            occ = o2
        outside = ~occ
        reach = np.zeros_like(occ)
        reach[0, :] = reach[-1, :] = reach[:, 0] = reach[:, -1] = True
        reach &= outside
        while True:                                     # flood the outside from the border
            r2 = reach.copy()
            r2[1:, :] |= reach[:-1, :]; r2[:-1, :] |= reach[1:, :]
            r2[:, 1:] |= reach[:, :-1]; r2[:, :-1] |= reach[:, 1:]
            r2 &= outside
            if np.array_equal(r2, reach):
                break
            reach = r2
        inside = ~reach
        dist = np.full((W, H), np.inf)
        dist[~inside] = 0.0
        cur = ~inside
        for k in range(1, steps + 1):                   # distance to the outline, in pixels
            grown = cur.copy()
            grown[1:, :] |= cur[:-1, :]; grown[:-1, :] |= cur[1:, :]
            grown[:, 1:] |= cur[:, :-1]; grown[:, :-1] |= cur[:, 1:]
            if k % 2 == 0:                              # alternate with diagonals: ~round distance
                grown[1:, 1:] |= cur[:-1, :-1]; grown[:-1, :-1] |= cur[1:, 1:]
                grown[1:, :-1] |= cur[:-1, 1:]; grown[:-1, 1:] |= cur[1:, :-1]
            new = grown & ~cur
            dist[new] = k
            cur = grown
        # the closing passes moved the outline outward by `close` px
        frame.update(grid=np.maximum(dist - close, 0.0) * px, g0=(t0g, u0), px=px, half=half)
        return frame

    @staticmethod
    def edge_band(frame, points, frac):
        """True where a point lies within `frac` of the half width from either
        edge of the blade (the sharpened bevel)."""
        q = points - frame["mu"]
        if "grid" in frame:
            t = q @ frame["t"]
            u = q @ frame["u"]
            g = frame["grid"]
            gi = np.clip(((t - frame["g0"][0]) / frame["px"]).astype(int), 0, g.shape[0] - 1)
            gj = np.clip(((u - frame["g0"][1]) / frame["px"]).astype(int), 0, g.shape[1] - 1)
            return g[gi, gj] < frac * frame["half"]
        t = q @ frame["t"]
        u = q @ frame["u"]
        bins = frame["bins"]
        x = np.clip((t - frame["t0"]) / max(frame["t1"] - frame["t0"], 1e-9) * bins - 0.5, 0, bins - 1)
        xs = np.arange(bins)
        lo = np.interp(x, xs, frame["lo"])
        hi = np.interp(x, xs, frame["hi"])
        # constant strip width (a fraction of the blade's typical half width), so
        # the strip keeps its width up to the point and the point itself is edge
        half = max(float(np.median((frame["hi"] - frame["lo"]) * 0.5)), 1e-9)
        return (u > hi - frac * half) | (u < lo + frac * half)

    # ------------------------------------------------------- face precision
    def smooth_face_borders(self, fl, iters=2, thr=0.6):
        """Single faces sticking into another region (a triangle spike on a
        border) take the label most of their neighbours have."""
        t = self._ftopo
        if t is None or not len(t):
            return fl
        fl = fl.copy()
        a, b = t[:, 0].astype(np.int64), t[:, 1].astype(np.int64)
        src = np.concatenate([a, b])
        dst = np.concatenate([b, a])
        nf = len(fl)
        deg = np.bincount(src, minlength=nf)
        L = int(fl.max()) + 1
        for _ in range(iters):
            key = src * L + fl[dst]
            u, cnt = np.unique(key, return_counts=True)
            face, lab = u // L, u % L
            order = np.lexsort((-cnt, face))
            face, lab, cnt = face[order], lab[order], cnt[order]
            first = np.ones(len(face), bool)
            first[1:] = face[1:] != face[:-1]
            f, l, c = face[first], lab[first], cnt[first]
            go = (l != fl[f]) & (c > thr * deg[f])
            if not go.any():
                break
            fl[f[go]] = l[go]
        return fl

    def clean_islands(self, fl, max_frac=0.001, rounds=3):
        """Final pass: a small patch of one region enclosed by exactly one other
        region joins it, when it is a stray bit (its region has a bigger piece
        elsewhere). Loose pieces and small regions of their own stay."""
        t = self._ftopo
        if t is None or not len(t):
            return fl
        fl = fl.copy()
        a, b = t[:, 0].astype(np.int64), t[:, 1].astype(np.int64)
        area = self._face_area
        tot = float(area.sum())
        for _ in range(rounds):
            same = fl[a] == fl[b]
            ea, eb = a[same], b[same]
            comp = np.arange(len(fl))
            for _ in range(500):
                m = np.minimum(comp[ea], comp[eb])
                new = comp.copy()
                np.minimum.at(new, ea, m)
                np.minimum.at(new, eb, m)
                new = new[new]
                if np.array_equal(new, comp):
                    break
                comp = new
            u, inv = np.unique(comp, return_inverse=True)
            ca = np.bincount(inv, weights=area)
            lab_of = np.zeros(len(u), np.int64)
            lab_of[inv] = fl
            d = ~same
            ci = np.concatenate([inv[a[d]], inv[b[d]]])
            nl = np.concatenate([fl[b[d]], fl[a[d]]]).astype(np.int64)
            L = int(fl.max()) + 1
            pairs = np.unique(ci.astype(np.int64) * L + nl)
            pc, pl = pairs // L, pairs % L
            nlab = np.bincount(pc, minlength=len(u))
            only = -np.ones(len(u), np.int64)
            only[pc[nlab[pc] == 1]] = pl[nlab[pc] == 1]
            big = np.zeros(int(fl.max()) + 1)
            np.maximum.at(big, lab_of, ca)
            # a stray bit: its region has a bigger piece somewhere else
            cand = (only >= 0) & (ca < max_frac * tot) & (ca < big[lab_of])
            if not cand.any():
                break
            # neighbours flip one at a time (smallest first) so two slivers
            # never swap labels with each other
            cj = np.concatenate([inv[b[d]], inv[a[d]]])
            cp = np.unique(ci.astype(np.int64) * len(u) + cj)
            src, dst = cp // len(u), cp % len(u)
            keep = cand[src]
            src, dst = src[keep], dst[keep]
            order = np.argsort(src, kind="stable")
            src, dst = src[order], dst[order]
            starts = np.searchsorted(src, np.arange(len(u) + 1))
            go = np.zeros(len(u), bool)
            fixed = np.zeros(len(u), bool)
            for c in np.nonzero(cand)[0][np.argsort(ca[cand])].tolist():
                nb = dst[starts[c]:starts[c + 1]]
                if go[nb].any() or fixed[c]:
                    continue
                go[c] = True
                fixed[nb] = True
            fl = np.where(go[inv], only[inv], fl)
        return fl

    def clean_speckles(self, fl, max_frac=0.0002, rel=0.05, rounds=2):
        """Small patches of one region sitting inside another (connected only to
        that one) join it. Separate loose pieces are never touched."""
        t = self._ftopo
        if t is None or not len(t):
            return fl
        fl = fl.copy()
        a, b = t[:, 0].astype(np.int64), t[:, 1].astype(np.int64)
        area = self._face_area
        tot = float(area.sum())
        for _ in range(rounds):
            same = fl[a] == fl[b]
            ea, eb = a[same], b[same]
            comp = np.arange(len(fl))
            for _ in range(200):
                m = np.minimum(comp[ea], comp[eb])
                new = comp.copy()
                np.minimum.at(new, ea, m)
                np.minimum.at(new, eb, m)
                new = new[new]
                if np.array_equal(new, comp):
                    break
                comp = new
            ca = np.bincount(comp, weights=area, minlength=len(fl))
            L = int(fl.max()) + 1
            ra = np.bincount(fl, weights=area, minlength=L)
            small = (ca > 0) & (ca < max_frac * tot)
            # ... and small next to the rest of its own region
            comp_lab = np.full(len(fl), -1)
            comp_lab[comp] = fl
            small &= ca < rel * np.where(comp_lab >= 0, ra[np.maximum(comp_lab, 0)], np.inf)
            dif = ~same
            pa, pb = a[dif], b[dif]
            ka = small[comp[pa]]
            kb = small[comp[pb]]
            vc = np.concatenate([comp[pa][ka], comp[pb][kb]])
            vl = np.concatenate([fl[pb][ka], fl[pa][kb]])
            if not len(vc):
                break
            key = vc * L + vl
            uk, cnt = np.unique(key, return_counts=True)
            uc, ul = uk // L, uk % L
            order = np.lexsort((-cnt, uc))
            first = np.ones(len(order), bool)
            first[1:] = uc[order][1:] != uc[order][:-1]
            best_c, best_l = uc[order][first], ul[order][first]
            target = -np.ones(len(fl), np.int64)
            target[best_c] = best_l
            mv = target[comp] >= 0
            if not mv.any():
                break
            fl[mv] = target[comp][mv]
        return fl

    # ------------------------------------------------------- visibility
    def hidden_cells(self, labels=None, ndirs=26):
        """Cells no viewer can see from any direction (buried inside another
        part, e.g. a blade's tang inside the guard). With labels given, also
        returns for each hidden cell the label of the part covering it."""
        dirs = [np.array(d, float) for d in
                [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]]
        dirs = [d / np.linalg.norm(d) for d in dirs][:ndirs]
        px = 1.5 * self.cell
        seen = np.zeros(self.n, bool)
        votes = None
        if labels is not None:
            L = int(labels.max()) + 1
            votes = np.zeros((self.n, L))
        for d in dirs:
            a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(d, a)
            u = np.floor(self.pos @ a / px).astype(np.int64)
            v = np.floor(self.pos @ b / px).astype(np.int64)
            depth = self.pos @ d                      # smaller = closer to viewer at -inf
            u -= u.min()
            v -= v.min()
            W, H = int(u.max()) + 3, int(v.max()) + 3
            zb = np.full((W, H), np.inf)
            np.minimum.at(zb, (u + 1, v + 1), depth)
            # front-most depth around each pixel (3x3), so sparse cells do not leave holes
            zz = zb.copy()
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    zz[1:-1, 1:-1] = np.minimum(zz[1:-1, 1:-1], zb[1 + du:W - 1 + du, 1 + dv:H - 1 + dv])
            front = zz[u + 1, v + 1]
            vis = depth <= front + 2.0 * self.cell
            seen |= vis
            if votes is not None:
                idx = np.full((W, H), -1, np.int64)
                order = np.argsort(-depth)                # nearest written last
                idx[u[order] + 1, v[order] + 1] = order
                occ = idx[u + 1, v + 1]
                h = ~vis & (occ >= 0)
                np.add.at(votes, (np.nonzero(h)[0], labels[occ[h]]), 1.0)
        hidden = ~seen
        if votes is None:
            return hidden
        cover = np.where(votes.sum(1) > 0, votes.argmax(1), -1)
        return hidden, cover

    # ------------------------------------------------------- sword details
    def _rel_width(self, cells):
        """Signed position across the blade (-1 .. 1 edge to edge) for all cells."""
        fr = self.edge_frame(cells)
        q = self.pos - fr["mu"]
        t = q @ fr["t"]
        u = q @ fr["u"]
        x = np.clip((t - fr["t0"]) / max(fr["t1"] - fr["t0"], 1e-9) * fr["bins"] - 0.5, 0, fr["bins"] - 1)
        xs = np.arange(fr["bins"])
        lo = np.interp(x, xs, fr["lo"])
        hi = np.interp(x, xs, fr["hi"])
        urel = (u - (lo + hi) / 2) / np.maximum((hi - lo) / 2, 1e-9)
        tl = (t - fr["t0"]) / max(fr["t1"] - fr["t0"], 1e-9)
        return urel, tl

    def find_fuller(self, blade_mask):
        """The groove running down the middle of a blade: a long, narrow,
        central piece bounded by concave creases. Returns a cell mask or None."""
        cells = np.nonzero(blade_mask)[0]
        if len(cells) < 50:
            return None
        urel, tl = self._rel_width(cells)
        A_bl = self.area[blade_mask].sum()
        for lv in (0.1, 0.15, 0.2, 0.25):
            comp = self.components_in(blade_mask, lv)
            m = comp >= 0
            if not m.any():
                continue
            A = np.bincount(comp[m], weights=self.area[m], minlength=self.n)
            found = []
            for c in np.nonzero(A > 0.015 * A_bl)[0].tolist():
                pm = comp == c
                if A[c] > 0.35 * A_bl:
                    continue
                cu = float(np.average(np.abs(urel[pm]), weights=self.area[pm]))
                ln = float(tl[pm].max() - tl[pm].min())
                if cu < 0.3 and ln > 0.3:
                    found.append(c)
            if found:
                mask = np.isin(comp, found)
                # the rest of the blade decides by creases: the band between the
                # bevel lines goes to the fuller even where a weak crease cut it
                # off (near the guard), the bevels and edges stay blade
                outer = blade_mask & ~mask & (np.abs(urel) > 0.6) & (self.wall < 0.3)
                seeds = [(int(c), 1) for c in np.nonzero(mask)[0][::2]]
                seeds += [(int(c), 0) for c in np.nonzero(outer)[0][::2]]
                _, _, lab = self.bottleneck_labeled(seeds, allowed=blade_mask)
                full = blade_mask & (lab == 1)
                # never wider than the groove found where it was found; only where
                # it was cut off (near the guard) may it follow the creases freely
                nb = 40
                tb = np.clip((tl * nb).astype(int), 0, nb - 1)
                cap = np.full(nb, 0.75)
                for k in range(nb):
                    mk = mask & (tb == k)
                    if mk.sum() >= 3:
                        cap[k] = min(0.75, float(np.percentile(np.abs(urel[mk]), 97)) + 0.06)
                full &= np.abs(urel) <= cap[tb]
                return full
        return None

    def find_ferrules(self, grip_mask, axis):
        """Metal collars at the ends of a grip: end stretches clearly wider than
        the grip itself. Returns a cell mask or None."""
        cells = np.nonzero(grip_mask)[0]
        if len(cells) < 50:
            return None
        mu = (self.pos[cells] * self.area[cells, None]).sum(0) / self.area[cells].sum()
        ax = np.asarray(axis, float)
        q = self.pos - mu
        t = q @ ax
        r = np.linalg.norm(q - t[:, None] * ax[None, :], axis=1)
        tg = t[cells]
        t0, t1 = tg.min(), tg.max()
        bins = 40
        b = np.clip(((t - t0) / max(t1 - t0, 1e-9) * bins).astype(int), 0, bins - 1)
        bc = b[cells]
        width = np.zeros(bins)
        for k in range(bins):
            mm = bc == k
            if mm.sum() >= 3:
                width[k] = np.percentile(r[cells][mm], 95)
        good = width > 0
        if good.sum() < 8:
            return None
        width = np.interp(np.arange(bins), np.nonzero(good)[0], width[good])
        med = float(np.median(width))
        fer_bins = np.zeros(bins, bool)
        for rng in (range(bins), range(bins - 1, -1, -1)):
            run = []
            for k in rng:
                if width[k] > 1.3 * med:
                    run.append(k)
                else:
                    break
            if 2 <= len(run) <= int(0.35 * bins):
                fer_bins[run] = True
        if not fer_bins.any():
            return None
        mask = grip_mask & fer_bins[b]
        core = grip_mask & ~mask
        if not core.any():
            return None
        # follow the creases between collar and grip, inside the grip only
        seeds = [(int(c), 1) for c in np.nonzero(mask & (self.wall < 0.3))[0]]
        seeds += [(int(c), 0) for c in np.nonzero(core & (self.wall < 0.3) &
                                                   (np.abs(b - np.clip(b, 0, bins)) == 0) &
                                                   ~np.isin(b, np.nonzero(np.convolve(fer_bins, [1, 1, 1], "same"))[0]))[0]]
        if seeds:
            _, _, lab = self.bottleneck_labeled(seeds, allowed=grip_mask)
            mask = grip_mask & (lab == 1)
        return mask if mask.any() else None

    def guard_subparts(self, guard_mask, level=None, min_frac=0.015, max_parts=4):
        """Split an ornate guard into its pieces (centre, ring, spikes...).
        Pieces of the same shape are grouped (all spikes = one region).
        Returns (labels within guard: 0 = rest of the guard, 1.. = groups, names)."""
        cells = np.nonzero(guard_mask)[0]
        if len(cells) < 50:
            return None
        if level is None:
            # ornate parts are creased almost everywhere: cut at the level that
            # leaves most of the surface in pieces, not at a fixed height
            level = float(np.percentile(self.wall[guard_mask], 60))
        comp = self.components_in(guard_mask, level)
        m = comp >= 0
        A_g = self.area[guard_mask].sum()
        A = np.bincount(comp[m], weights=self.area[m], minlength=self.n)
        big = [int(c) for c in np.nonzero(A >= min_frac * A_g)[0]]
        if len(big) < 2:
            return None
        cen = (self.pos[cells] * self.area[cells, None]).sum(0) / A_g
        sig = {}
        for c in big:
            pm = comp == c
            a, ext = self.mask_signature(pm)
            pc = (self.pos[pm] * self.area[pm, None]).sum(0) / a
            sig[c] = (a, ext, float(np.linalg.norm(pc - cen)))
        # the centre piece: closest to the guard's centre
        centre = min(big, key=lambda c: sig[c][2])
        groups = [[centre]]
        rest = [c for c in big if c != centre]
        used = set()
        for c in sorted(rest, key=lambda c: -sig[c][0]):
            if c in used:
                continue
            g = [c]
            used.add(c)
            for d in rest:
                if d in used:
                    continue
                ra = sig[d][0] / sig[c][0]
                re = sig[d][1] / np.maximum(sig[c][1], 1e-9)
                if 0.75 <= ra <= 1.33 and np.all((re > 0.75) & (re < 1.33)):
                    g.append(d)
                    used.add(d)
            groups.append(g)
        # biggest non-centre group stays "Guard"; repeated small pieces become a group
        out = np.full(self.n, -1, np.int64)
        names = ["Guard"]
        seeds = []
        k = 1
        main = max(groups[1:], key=lambda g: sum(sig[c][0] for c in g)) if len(groups) > 1 else None
        for g in groups:
            if g is main:
                lab = 0
            elif g[0] == centre:
                lab = k
                names.append("Guard Centre")
                k += 1
            elif len(g) >= 2 and k <= max_parts:
                lab = k
                names.append("Guard Detail" if "Guard Detail" not in names else f"Guard Detail {k}")
                k += 1
            else:
                lab = 0
            for c in g:
                seeds += [(int(x), lab) for x in np.nonzero(comp == c)[0][::3]]
        if len(names) < 2:
            return None
        _, _, lab = self.bottleneck_labeled(seeds, allowed=guard_mask)
        lab[~guard_mask] = -1
        lab[guard_mask & (lab < 0)] = 0
        return lab, names

    def face_labels_smooth(self, labels, sigma=0.8, normal_k=4.0):
        """Cell labels -> face labels without cell-sized steps: a face on a
        border takes the label its surrounding cells vote for, weighted by
        distance (smooth, interpolated border) and by facing the same way (so
        the border still snaps to sharp turns)."""
        labels = np.asarray(labels)
        fl = labels[self.face_cell]
        i, j = self.edges[:, 0], self.edges[:, 1]
        diff = labels[i] != labels[j]
        if not diff.any():
            return fl
        border = np.zeros(self.n, bool)
        border[i[diff]] = True
        border[j[diff]] = True
        faces = np.nonzero(border[self.face_cell])[0]
        L = int(labels.max()) + 1
        ip, adj = self.indptr, self.adj
        s2 = (sigma * self.cell) ** 2
        for chunk in range(0, len(faces), 200_000):
            fc_ = faces[chunk:chunk + 200_000]
            cell = self.face_cell[fc_]
            deg = ip[cell + 1] - ip[cell]
            rep = np.repeat(np.arange(len(fc_)), deg + 1)
            starts = np.repeat(ip[cell], deg + 1)
            off = np.arange(len(rep)) - np.repeat(np.cumsum(deg + 1) - (deg + 1), deg + 1)
            cand = np.where(off == 0, np.repeat(cell, deg + 1), adj[np.minimum(starts + off - 1, len(adj) - 1)])
            p = self._fcen[fc_][rep].astype(np.float64)
            nrm = self._fnor[fc_][rep].astype(np.float64)
            d2 = np.einsum("ij,ij->i", p - self.pos[cand], p - self.pos[cand])
            facing = np.einsum("ij,ij->i", nrm, self.nor[cand])
            w = self.area[cand] * np.exp(-d2 / s2) * np.exp(-normal_k * (1.0 - facing))
            votes = np.zeros((len(fc_), L))
            np.add.at(votes, (rep, labels[cand]), w)
            fl[fc_] = votes.argmax(1)
        return fl

    def band_profile(self, frame, band_cells, bins=160, smooth=7):
        """Across-extent [lo, hi] of band_cells per position along the blade,
        gap-filled and smoothed (running median). None if too few cells."""
        q = self.pos[band_cells] - frame["mu"]
        tc, uc = q @ frame["t"], q @ frame["u"]
        t0, t1 = frame["t0"], frame["t1"]
        bc = np.clip(((tc - t0) / max(t1 - t0, 1e-9) * bins).astype(int), 0, bins - 1)
        lo = np.full(bins, np.nan)
        hi = np.full(bins, np.nan)
        for k in range(bins):
            m = bc == k
            if m.sum() >= 3:
                lo[k], hi[k] = np.percentile(uc[m], 3), np.percentile(uc[m], 97)
        have = ~np.isnan(lo)
        if have.sum() < 3:
            return None
        idx = np.arange(bins)
        lo_f = np.interp(idx, idx[have], lo[have])
        hi_f = np.interp(idx, idx[have], hi[have])
        h = smooth // 2
        pad_lo = np.pad(lo_f, h, mode="edge")
        pad_hi = np.pad(hi_f, h, mode="edge")
        lo_s = np.array([np.median(pad_lo[k:k + smooth]) for k in range(bins)])
        hi_s = np.array([np.median(pad_hi[k:k + smooth]) for k in range(bins)])
        # then a moving average, so width changes taper instead of stepping
        ker = np.ones(9) / 9.0
        lo_s = np.convolve(np.pad(lo_s, 4, mode="edge"), ker, "valid")
        hi_s = np.convolve(np.pad(hi_s, 4, mode="edge"), ker, "valid")
        return {"lo": lo_s, "hi": hi_s, "first": idx[have][0], "last": idx[have][-1], "bins": bins}

    @staticmethod
    def band_dist(frame, prof, points):
        """Signed distance (metres) of points from the band: < 0 inside."""
        q = points - frame["mu"]
        t, u = q @ frame["t"], q @ frame["u"]
        bins = prof["bins"]
        idx = np.arange(bins)
        x = (t - frame["t0"]) / max(frame["t1"] - frame["t0"], 1e-9) * bins - 0.5
        xc = np.clip(x, 0, bins - 1)
        lo = np.interp(xc, idx, prof["lo"])
        hi = np.interp(xc, idx, prof["hi"])
        d = np.maximum(lo - u, u - hi)
        # beyond the ends of the band: also outside
        step = (frame["t1"] - frame["t0"]) / bins
        d = np.maximum(d, (prof["first"] - 0.5 - x) * step)
        d = np.maximum(d, (x - prof["last"] - 0.5) * step)
        return d

    def band_faces(self, frame, band_cells, faces, bins=160, smooth=7):
        """Faces lying within the smooth band (used to draw the fuller as a
        clean strip instead of cell-sized steps)."""
        prof = self.band_profile(frame, band_cells, bins, smooth)
        if prof is None:
            return np.zeros(len(faces), bool)
        return self.band_dist(frame, prof, self._fcen[faces].astype(np.float64)) <= 0

    @staticmethod
    def edge_dist(frame, points, frac):
        """Signed distance (metres) from the edge strip border: < 0 inside the
        strip (only with an outline grid). Bilinear, so the border is smooth."""
        q = points - frame["mu"]
        t, u = q @ frame["t"], q @ frame["u"]
        g = frame["grid"]
        cap = frame["half"] * 2.0
        gg = np.minimum(g, cap)
        x = np.clip((t - frame["g0"][0]) / frame["px"] - 0.5, 0, g.shape[0] - 1.001)
        y = np.clip((u - frame["g0"][1]) / frame["px"] - 0.5, 0, g.shape[1] - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        v = (gg[x0, y0] * (1 - fx) * (1 - fy) + gg[x0 + 1, y0] * fx * (1 - fy)
             + gg[x0, y0 + 1] * (1 - fx) * fy + gg[x0 + 1, y0 + 1] * fx * fy)
        return v - frac * frame["half"]

    # ------------------------------------------------------- elongated parts
    def principal_axes(self):
        p = self.pos
        w = self.area / self.area.sum()
        mean = (p * w[:, None]).sum(0)
        q = p - mean
        cov = (q * w[:, None]).T @ q
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        return mean, vecs[:, order].T, np.sqrt(np.maximum(vals[order], 0))

    def elongated_parts(self, bins=160, max_parts=6, penalty=None):
        """Split an elongated object (sword, spear, tool) along its long axis by
        where its width changes. Returns (labels, names) or None if not elongated."""
        mean, axes, spread = self.principal_axes()
        if spread[1] <= 0 or spread[0] / spread[1] < 2.2:
            return None
        t = (self.pos - mean) @ axes[0]
        u = (self.pos - mean) @ axes[1]
        v = (self.pos - mean) @ axes[2]
        t0, t1 = np.percentile(t, 0.2), np.percentile(t, 99.8)
        b = np.clip(((t - t0) / max(t1 - t0, 1e-9) * bins).astype(int), 0, bins - 1)
        width = np.zeros(bins)
        for k in range(bins):
            m = b == k
            if m.sum() >= 3:
                uu, vv = u[m], v[m]
                width[k] = max(np.percentile(uu, 98) - np.percentile(uu, 2),
                               0.6 * (np.percentile(vv, 98) - np.percentile(vv, 2)))
        good = width > 0
        if good.sum() < 10:
            return None
        width = np.interp(np.arange(bins), np.nonzero(good)[0], width[good])
        y = np.log(width + 1e-4 * self.size)
        segs = _changepoints(y, max_parts, penalty)
        # merge very short segments into the neighbour with the closest width
        min_len = max(2, int(bins * 0.03))
        while len(segs) > 1:
            lens = [e - s for s, e in segs]
            i = int(np.argmin(lens))
            if lens[i] >= min_len:
                break
            mi = [y[s:e].mean() for s, e in segs]
            if i == 0:
                j = 1
            elif i == len(segs) - 1:
                j = i - 1
            else:
                j = i - 1 if abs(mi[i - 1] - mi[i]) < abs(mi[i + 1] - mi[i]) else i + 1
            a, c = sorted((i, j))
            segs[a] = (segs[a][0], segs[c][1])
            del segs[c]
        segs = _refine_ends(segs, width)
        if len(segs) < 2:
            return None
        names = _name_parts(segs, width)
        # cells take the segment of their position along the axis; a band around
        # each cut is then re-decided by which part's core a cell reaches over
        # the lowest crease, so cuts follow the real junction (the blade's flare
        # stays blade, guard spikes stay guard) instead of a straight plane
        lab = np.zeros(self.n, np.int64)
        for si, (s, e) in enumerate(segs):
            lab[(b >= s) & (b < e)] = si
        near = np.zeros(self.n, bool)
        for k in range(len(segs) - 1):
            c = segs[k][1]
            shortest = min(segs[k][1] - segs[k][0], segs[k + 1][1] - segs[k + 1][0])
            band = max(2, min(int(bins * 0.08), int(0.35 * shortest)))
            near |= np.abs(b - c) < band
        seeds_m = np.zeros(self.n, bool)
        for si, (s, e) in enumerate(segs):
            mid = (s + e) / 2.0
            half = max(0.6, 0.25 * (e - s))
            mine = (b >= s) & (b < e)
            core = mine & ((np.abs(b + 0.5 - mid) <= half) | ~near)
            if not core.any():
                continue
            # creased everywhere (a wrapped grip)? seed from its smoothest half
            thr = min(0.3, float(np.percentile(self.wall[core], 50)) + 1e-6)
            seeds_m |= core & (self.wall <= thr)
        seeds = list(zip(np.nonzero(seeds_m)[0].tolist(), lab[seeds_m].tolist()))
        if near.any() and seeds:
            bb, _, blab = self.bottleneck_labeled(seeds)
            fix = near & (blab >= 0) & ~seeds_m
            # never move a cell to a part far along the axis from it
            seg_of = np.array([si for si, (s, e) in enumerate(segs) for _ in range(e - s)])
            dseg = np.abs(blab - lab)
            fix &= dseg <= 1
            lab[fix] = blab[fix]
        return lab, names, {"axis": axes[0].tolist(), "segments": segs, "width": width.tolist()}


def _vertex_neighbours(mesh, n):
    """Face pairs that share a vertex (chained around each vertex)."""
    try:
        nl = len(mesh.loops)
        lv = np.empty(nl, np.int64)
        mesh.loops.foreach_get("vertex_index", lv)
        ls = np.empty(n, np.int64)
        lt = np.empty(n, np.int64)
        mesh.polygons.foreach_get("loop_start", ls)
        mesh.polygons.foreach_get("loop_total", lt)
    except (AttributeError, TypeError, RuntimeError):
        return None
    o = np.argsort(ls, kind="stable")
    lp = np.repeat(o, lt[o])
    if len(lp) != nl:
        return None
    ov = np.argsort(lv, kind="stable")
    sv, fv = lv[ov], lp[ov]
    same = sv[1:] == sv[:-1]
    return np.stack([fv[:-1][same], fv[1:][same]], 1)


def _relabel(labels):
    _, inv = np.unique(labels, return_inverse=True)
    return inv


def _changepoints(y, max_parts, penalty):
    """Optimal piecewise-constant segmentation (least squares) with a penalty
    per segment; the penalty is raised until at most max_parts remain."""
    n = len(y)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cs2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(i, j):
        m = j - i
        s = cs[j] - cs[i]
        return (cs2[j] - cs2[i]) - s * s / m

    var = float(np.var(y)) or 1e-6
    pen = penalty if penalty is not None else var * n * 0.02
    for _ in range(30):
        best = [0.0] + [float("inf")] * n
        prev = [0] * (n + 1)
        for j in range(1, n + 1):
            bj, pj = float("inf"), 0
            for i in range(j):
                c = best[i] + cost(i, j) + pen
                if c < bj:
                    bj, pj = c, i
            best[j], prev[j] = bj, pj
        segs = []
        j = n
        while j > 0:
            segs.append((prev[j], j))
            j = prev[j]
        segs.reverse()
        if len(segs) <= max_parts:
            return segs
        pen *= 1.6
    return segs


def _merge_taper(segs, L, width, med):
    """Merge a narrower piece that continues the longest piece toward a point."""
    mL = med(*segs[L])
    for k in (L + 1, L - 1):
        if not 0 <= k < len(segs):
            continue
        s, e = segs[k]
        cut = s if k > L else e
        a = med(max(cut - 3, 0), cut)
        b = med(cut, cut + 3)
        cont = max(a, b) / max(min(a, b), 1e-9) < 1.3
        # a short narrow piece at the very end is the point, whatever the noise
        terminal = k in (0, len(segs) - 1)
        cont = cont or (terminal and (e - s) < 0.12 * len(width))
        tapering = med(s, e) <= 1.05 * mL
        # only if everything beyond it narrows too (a point, not a handle)
        beyond = list(range(k, len(segs))) if k > L else list(range(k, -1, -1))
        meds = [med(*segs[q]) for q in beyond]
        narrowing = all(m <= 1.05 * mL for m in meds)
        # or: widths keep shrinking piece after piece down to a sharp end
        point = (all(meds[i + 1] <= 0.95 * meds[i] for i in range(len(meds) - 1))
                 and meds[0] <= 0.95 * mL and meds[-1] <= 0.5 * mL)
        if tapering and narrowing and (cont or point):
            lo, hi = min(k, L), max(k, L)
            segs[lo] = (segs[lo][0], segs[hi][1])
            del segs[hi]
            return True
    return False


def _refine_ends(segs, width):
    """Fix the two typical mistakes of a step fit on a long object:
    * a tapering end (a blade's point) cut off as its own piece -> merged back
    * a knob at the far end (a pommel, a hammer head) hidden inside a long
      thin piece (the grip) -> split off at the neck."""
    segs = list(segs)
    if len(segs) < 2:
        return segs
    med = lambda a, b: float(np.median(width[a:b])) if b > a else 0.0
    # tapering ends next to the longest piece (repeat: a point can be cut twice)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        lens = [e - s for s, e in segs]
        L = int(np.argmax(lens))
        changed = _merge_taper(segs, L, width, med)
    # a knob at the far end of the terminal pieces
    for end in (len(segs) - 1, 0):           # far end first: indices stay valid
        s, e = segs[end]
        if e - s < 8:
            continue
        mw = med(s, e)
        far = slice(s, s + int(0.45 * (e - s))) if end == 0 else slice(e - int(0.45 * (e - s)), e)
        fw = width[far]
        if len(fw) and fw.max() > 1.6 * mw and mw > 0:
            pk = far.start + int(np.argmax(fw))
            rng = range(pk, e) if end == 0 else range(pk, s - 1, -1)
            neck = None
            for k in rng:
                if width[k] <= 1.15 * mw:
                    neck = k
                    break
            if neck is not None and min(neck - s, e - neck) >= 3:
                if end == 0:
                    segs[0:1] = [(s, neck), (neck, e)]
                else:
                    segs[end:end + 1] = [(s, neck + 1), (neck + 1, e)]
    return segs


def _name_parts(segs, width):
    """Heuristic names for an elongated object's parts (sword-like if it fits)."""
    n = len(segs)
    lens = np.array([e - s for s, e in segs], float)
    wid = np.array([width[s:e].mean() for s, e in segs])
    names = [f"Part {i + 1}" for i in range(n)]
    longest = int(np.argmax(lens))
    if lens[longest] / lens.sum() < 0.4:
        return names
    widest = int(np.argmax(np.where(np.arange(n) == longest, -1, wid)))
    if widest != longest and wid[widest] > 1.15 * wid[longest] and abs(widest - longest) == 1:
        names[longest] = "Blade"
        names[widest] = "Guard"
        step = 1 if widest > longest else -1
        k = widest + step
        if 0 <= k < n:
            names[k] = "Grip"
            k += step
            while 0 <= k < n:
                names[k] = "Pommel" if (k + step < 0 or k + step >= n) else "Grip"
                k += step
        # anything beyond the blade's far end
        k = longest - step
        while 0 <= k < n:
            names[k] = "Tip"
            k -= step
    else:
        names[longest] = "Body"
    # keep names unique
    seen = {}
    for i, nm in enumerate(names):
        if nm in seen:
            seen[nm] += 1
            names[i] = f"{nm} {seen[nm]}"
        else:
            seen[nm] = 1
    return names


def face_colors_from_material(obj, mesh, max_res=512):
    """Per-face base colour sampled from the image driving Base Color (or None)."""
    import bpy
    mat = obj.active_material
    if not mat or not mat.node_tree or not mesh.uv_layers.active:
        return None
    img = None
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED" and n.inputs["Base Color"].is_linked:
            src = n.inputs["Base Color"].links[0].from_node
            if src.type == "TEX_IMAGE" and src.image:
                img = src.image
            break
    if img is None:
        return None
    small = img.copy()
    try:
        w, h = small.size
        s = min(1.0, max_res / max(w, h, 1))
        if s < 1.0:
            small.scale(max(1, int(w * s)), max(1, int(h * s)))
        w, h = small.size
        px = np.empty(w * h * 4, np.float32)
        small.pixels.foreach_get(px)
        px = px.reshape(h, w, 4)[..., :3]
    finally:
        bpy.data.images.remove(small)
    nl = len(mesh.loops)
    uv = np.empty(nl * 2, np.float64)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    n = len(mesh.polygons)
    starts = np.empty(n, np.int64)
    counts = np.empty(n, np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", counts)
    order = np.argsort(starts)
    su = np.add.reduceat(uv[:, 0], starts[order]) / counts[order]
    sv = np.add.reduceat(uv[:, 1], starts[order]) / counts[order]
    fu = np.empty(n)
    fv = np.empty(n)
    fu[order], fv[order] = su, sv
    x = np.clip((fu % 1.0) * (w - 1), 0, w - 1).astype(int)
    y = np.clip((fv % 1.0) * (h - 1), 0, h - 1).astype(int)
    return px[y, x].astype(np.float64)
