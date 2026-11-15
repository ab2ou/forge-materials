# SPDX-License-Identifier: GPL-3.0-or-later
"""Runs in the separate AI environment (PyTorch + SAM 2), not in Blender.

  python sam_parts.py <views_dir> <checkpoint> [--device cuda|cpu] [--points N]

Views may belong to a part (views.json "part"): parts are rendered on their
own, so nothing hides them, and each part is lifted separately.

1. SAM 2 finds masks in every rendered view (normal matcap + shaded).
2. Per view the masks are fused into one label map (smaller masks on top).
3. Labels are lifted to the mesh through the face-ID renders: labels of
   different views that cover the same faces are linked (match graph) and
   grouped by community detection; every face takes the group it was seen in
   most. Writes face_parts.npy (int32 per face, -1 = never seen).
"""
import json, os, sys, time
import numpy as np


def load_ids(path):
    """Face id per pixel from the EXR render (-1 = background)."""
    import OpenEXR
    with OpenEXR.File(path) as f:
        ch = f.channels()
        rgb = None
        for k, v in ch.items():                        # packed "RGB"/"RGBA" channel
            if v.pixels.ndim == 3 and v.pixels.shape[-1] >= 3:
                rgb = v.pixels[..., :3]
                break
        if rgb is None:                                # separate R, G, B (maybe "View.R" ...)
            pick = {}
            for k, v in ch.items():
                tail = k.split(".")[-1]
                if tail in ("R", "G", "B"):
                    pick[tail] = v.pixels
            rgb = np.stack([pick["R"], pick["G"], pick["B"]], -1)
        rgb = np.asarray(rgb, np.float64)
    k = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.int64)
    return (k[..., 0] + (k[..., 1] << 8) + (k[..., 2] << 16)) - 1


def fuse(masks, shape, min_px):
    lab = -np.ones(shape, np.int32)
    for k, m in enumerate(sorted(masks, key=lambda m: -m["area"])):
        lab[m["segmentation"]] = k
    # drop specks
    out = -np.ones(shape, np.int32)
    n = 0
    for k in np.unique(lab[lab >= 0]):
        sel = lab == k
        if sel.sum() >= min_px:
            out[sel] = n
            n += 1
    return out


def lift(views, nf):
    """face -> group (-1 unseen) from the label maps of some views"""
    import scipy.sparse as sp
    import networkx as nx
    rows, cols, vals = [], [], []
    node_view, vis_rows, vis_cols = [], [], []
    base = 0
    for vi, (ids, lab, _) in enumerate(views):
        fgm = ids >= 0
        vis = np.unique(ids[fgm])
        vis_rows.append(np.full(len(vis), vi)); vis_cols.append(vis)
        L = lab.max() + 1
        sel = fgm & (lab >= 0)
        key = (lab[sel].astype(np.int64) + base) * (nf + 1) + ids[sel]
        u, c = np.unique(key, return_counts=True)
        rows.append(u // (nf + 1)); cols.append(u % (nf + 1)); vals.append(c)
        node_view += [vi] * L
        base += L
    N = base
    A = sp.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(N, nf + 1))
    B = A.copy(); B.data[:] = 1.0                                        # node x face (binary)
    V = sp.csr_matrix((np.ones(sum(len(x) for x in vis_cols), np.float32),
                       (np.concatenate(vis_rows), np.concatenate(vis_cols))), shape=(len(views), nf + 1))
    inter = (B @ B.T).tocoo()                                            # shared faces
    seen_in = (B @ V.T).toarray()                                        # node faces visible in view j
    node_view = np.array(node_view)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for a, b, s in zip(inter.row, inter.col, inter.data):
        if a >= b or node_view[a] == node_view[b]:
            continue
        den = min(seen_in[a, node_view[b]], seen_in[b, node_view[a]])
        if den <= 20:
            continue
        r = s / den
        if r > 0.5:
            G.add_edge(int(a), int(b), weight=float(r))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=1.0, seed=1)
    node_comm = np.zeros(N, np.int64)
    for k, cset in enumerate(comms):
        for n in cset:
            node_comm[n] = k
    # face votes: pixel counts per community
    Acoo = A.tocoo()
    K = len(comms)
    key = node_comm[Acoo.row] * (nf + 1) + Acoo.col
    u, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=Acoo.data)
    comm_u, face_u = u // (nf + 1), u % (nf + 1)
    best = np.full(nf + 1, -1, np.int64)
    bestw = np.zeros(nf + 1)
    order = np.argsort(w)
    best[face_u[order]] = comm_u[order]            # the heaviest vote is written last
    bestw[face_u[order]] = w[order]
    return best[:nf].astype(np.int32), K


def main():
    vdir, ckpt = sys.argv[1], sys.argv[2]
    device = "cuda"
    if "--device" in sys.argv:
        device = sys.argv[sys.argv.index("--device") + 1]
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    meta = json.load(open(os.path.join(vdir, "views.json")))
    nf = meta["faces"]
    t0 = time.time()
    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", ckpt, device=device)
    pts = int(sys.argv[sys.argv.index("--points") + 1]) if "--points" in sys.argv else 32
    gen = SAM2AutomaticMaskGenerator(model, points_per_side=pts, points_per_batch=32,
                                     pred_iou_thresh=0.65, stability_score_thresh=0.82,
                                     min_mask_region_area=100)
    print("model", device, round(time.time() - t0, 1), flush=True)
    views = []
    for vi, view in enumerate(meta["views"]):
        name = view.get("name", f"v{vi:02d}")
        ids = load_ids(os.path.join(vdir, f"{name}_id.exr"))
        fg = ids >= 0
        masks = []
        for mode in meta.get("modes", ["normal", "shaded"]):
            img = np.array(Image.open(os.path.join(vdir, f"{name}_{mode}.png")).convert("RGB"))
            with torch.inference_mode():
                ms = gen.generate(img)
            for m in ms:
                seg = m["segmentation"] & fg
                if seg.sum() > 0 and seg.sum() < 0.98 * fg.sum():
                    masks.append({"segmentation": seg, "area": int(seg.sum())})
        lab = fuse(masks, ids.shape, max(64, int(fg.sum() * 0.0015)))
        lab[~fg] = -1
        np.save(os.path.join(vdir, f"{name}_labels.npy"), lab)
        views.append((ids.astype(np.int32), lab, view.get("part", -1)))
        print(f"view {vi}: {len(masks)} masks -> {lab.max() + 1} regions  {time.time() - t0:.0f}s", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    faces = -np.ones(nf, np.int32)
    K = 0
    groups = sorted({p for _, _, p in views})
    for g in groups:
        sub = [v for v in views if v[2] == g]
        f, k = lift(sub, nf)
        own = f >= 0
        if g >= 0 and "parts" in meta:              # a part only labels its own faces
            own &= np.load(os.path.join(vdir, meta["parts"]))[: nf] == g
        faces[own] = f[own] + K
        K += k
        print(f"part {g}: {k} groups, {int(own.sum())} faces", flush=True)
    np.save(os.path.join(vdir, "face_parts.npy"), faces)
    print(json.dumps({"groups": K, "seen_faces": int((faces >= 0).sum()), "faces": nf,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
