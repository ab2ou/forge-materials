# SPDX-License-Identifier: GPL-3.0-or-later
"""Render views (Blender, background) then SAM 2 parts (this environment).

  python run_pipeline.py <blender.exe> <file.blend> <object> <out_dir> [nviews] [parts.npy]

With parts.npy every part is rendered on its own and whole (nothing hides it,
no tile seams); only the shaded image is used and SAM samples a 24x24 grid,
to keep it fast.
"""
import os, subprocess, sys, time

blender, blend, obj, out = sys.argv[1:5]
nv = sys.argv[5] if len(sys.argv) > 5 else "10"
parts = sys.argv[6] if len(sys.argv) > 6 else None
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))           # ...\tools
ckpt = os.environ.get("FM_AI_CKPT") or os.path.join(root, "fm_ai", "checkpoints", "sam2.1_hiera_small.pt")
os.makedirs(out, exist_ok=True)
env = dict(os.environ)
if parts:
    env.setdefault("FM_AI_MODES", "shaded")
    env.setdefault("FM_AI_MAXTILES", "1")
    env.setdefault("FM_AI_TILE_ASPECT", "2.2")
t = time.time()
cmd = [blender, "-b", blend, "--python", os.path.join(here, "render_views.py"), "--", out, obj, "1024", nv]
if parts:
    cmd.append(parts)
r = subprocess.run(cmd, capture_output=True, text=True, env=env)
print("render", r.returncode, round(time.time() - t), "s", flush=True)
if r.returncode != 0 or not os.path.exists(os.path.join(out, "views.json")):
    print(r.stdout[-3000:], r.stderr[-3000:])
    sys.exit(1)
sam = [sys.executable, os.path.join(here, "sam_parts.py"), out, ckpt]
if parts:
    sam += ["--points", "24"]
r = subprocess.run(sam, text=True, env=env)
print("sam", r.returncode, round(time.time() - t), "s", flush=True)
sys.exit(r.returncode)
