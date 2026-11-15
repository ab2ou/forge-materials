# SPDX-License-Identifier: GPL-3.0-or-later
"""AI Hybrid regions: the image segmenter (SAM 2, in its own Python
environment) proposes sub-parts; the shape split keeps every border.
Runs on a saved copy in the background; the open file is only touched when
the result is applied (material slots + face material index, one undo step)."""
import json, os, shutil, signal, subprocess, tempfile, time
import bpy
import numpy as np

from . import hybrid, regions

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AI = os.path.join(os.path.dirname(_HERE), "fm_ai")
_RUN_PREFIX = "fm_ai_"
_STALE_SECONDS = 24 * 3600


def kill_tree(proc):
    """Stop the pipeline and every process it started (Blender render, SAM 2).
    Killing only the wrapper leaves its children running and the GPU busy."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=0x08000000, timeout=30)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
        proc.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def remove_run(path):
    """Delete a temporary run folder (copy of the .blend, renders: 100-400 MB)."""
    if not path or not os.path.basename(os.path.normpath(path)).startswith(_RUN_PREFIX):
        return
    for _ in range(3):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.5)                               # Windows: files may still be closing


def remove_stale_runs(folder):
    """Remove run folders left behind by a crash (older than a day)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return
    now = time.time()
    for n in names:
        p = os.path.join(folder, n)
        if (n.startswith(_RUN_PREFIX) and os.path.isdir(p)
                and os.path.exists(os.path.join(p, "copy.blend"))
                and now - os.path.getmtime(p) > _STALE_SECONDS):
            remove_run(p)


def ai_folder(context):
    try:
        p = context.preferences.addons[__package__].preferences.ai_path
    except (KeyError, AttributeError):
        p = ""
    return bpy.path.abspath(p) if p else _DEFAULT_AI


def ai_python(folder):
    for rel in (("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
        p = os.path.join(folder, *rel)
        if os.path.isfile(p):
            return p
    return None


class FM_OT_regions_ai(bpy.types.Operator):
    bl_idname = "fm.regions_ai"
    bl_label = "AI Hybrid Parts"
    bl_description = ("Find sub-parts with the AI image segmenter (SAM 2) on every part rendered "
                      "on its own; borders still follow the real creases. Runs in the background "
                      "(about 5-30 min on a 4 GB GPU). Esc cancels")

    views: bpy.props.IntProperty(name="Views per Part", default=10, min=6, max=26)

    _proc = None
    _timer = None
    run = None

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == "MESH" and context.mode == "OBJECT"

    def execute(self, context):
        obj = context.active_object
        folder = ai_folder(context)
        py = ai_python(folder)
        if not py:
            self.report({"ERROR"}, f"AI environment not found in {folder} (set it in the add-on preferences)")
            return {"CANCELLED"}
        ckpt = os.path.join(folder, "checkpoints", "sam2.1_hiera_small.pt")
        if not os.path.isfile(ckpt):
            self.report({"ERROR"}, f"SAM 2 checkpoint missing: {ckpt}")
            return {"CANCELLED"}
        C = regions.get_cells(obj)
        S = regions.Session(obj, C)
        if not S.init_from_slots():
            S.auto("SHAPE")
        self.sess = S
        self.n_faces = len(obj.data.polygons)
        base = folder if os.access(folder, os.W_OK) else None
        if base:
            remove_stale_runs(base)
        run = tempfile.mkdtemp(prefix=_RUN_PREFIX, dir=base)
        self.run = run
        np.save(os.path.join(run, "parts.npy"), hybrid.part_faces(C, S.labels, S.names, S.rules))
        blend = os.path.join(run, "copy.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend, copy=True, compress=False)
        self.log_path = os.path.join(run, "log.txt")
        self._log = open(self.log_path, "w")
        cmd = [py, os.path.join(_HERE, "ai", "run_pipeline.py"), bpy.app.binary_path, blend, obj.name,
               os.path.join(run, "views"), str(self.views), os.path.join(run, "parts.npy")]
        env = dict(os.environ, FM_AI_CKPT=ckpt)
        if os.name == "nt":
            self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, cwd=folder,
                                          creationflags=0x08000000, env=env)
        else:                                         # own process group, so kill_tree reaches the children
            self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, cwd=folder,
                                          start_new_session=True, env=env)
        self.t0 = time.time()
        self.obj_name = obj.name
        wm = context.window_manager
        self._timer = wm.event_timer_add(2.0, window=context.window)
        wm.modal_handler_add(self)
        context.workspace.status_text_set("AI Hybrid: rendering parts...  (Esc: cancel)")
        return {"RUNNING_MODAL"}

    def _status(self, context):
        try:
            lines = open(self.log_path, encoding="utf-8", errors="ignore").read().splitlines()
        except OSError:
            lines = []
        views = [l for l in lines if l.startswith("view ")]
        total = 0
        vj = os.path.join(self.run, "views", "views.json")
        if os.path.exists(vj):
            try:
                total = len(json.load(open(vj))["views"])
            except (ValueError, KeyError):
                pass
        m = int(time.time() - self.t0) // 60
        if total:
            msg = f"AI Hybrid: segmenting view {len(views)}/{total}  ({m} min)  (Esc: cancel)"
        else:
            msg = f"AI Hybrid: rendering parts  ({m} min)  (Esc: cancel)"
        context.workspace.status_text_set(msg)

    def _finish(self, context):
        try:
            if self._timer is not None:
                context.window_manager.event_timer_remove(self._timer)
            if context.workspace:
                context.workspace.status_text_set(None)
        except (AttributeError, ReferenceError, RuntimeError):
            pass
        self._timer = None
        try:
            self._log.close()
        except Exception:
            pass

    def cancel(self, context):
        # Blender cancels a running modal operator when the file is closed or reloaded
        kill_tree(self._proc)
        self._finish(context)
        remove_run(self.run)

    def modal(self, context, event):
        if event.type == "ESC":
            kill_tree(self._proc)
            self._finish(context)
            remove_run(self.run)
            self.report({"WARNING"}, "AI Hybrid cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._proc.poll() is None:
            self._status(context)
            return {"PASS_THROUGH"}
        self._finish(context)
        fp = os.path.join(self.run, "views", "face_parts.npy")
        obj = bpy.data.objects.get(self.obj_name)
        if self._proc.returncode != 0 or not os.path.exists(fp) or obj is None:
            # the run folder is kept for its log; it is removed after a day
            self.report({"ERROR"}, f"AI Hybrid failed, see {self.log_path}")
            return {"CANCELLED"}
        face_sam = np.load(fp)
        remove_run(self.run)
        if len(face_sam) != len(obj.data.polygons) or len(obj.data.polygons) != self.n_faces:
            self.report({"ERROR"}, "The mesh changed while the AI was running; result not applied")
            return {"CANCELLED"}
        S = self.sess
        S.push()
        before = len(S.names)
        lab, names = hybrid.hybrid_labels(S.C, S.labels, S.names, S.rules, face_sam)
        S.labels, S.slot_of = lab, list(S.slot_of) + [None] * (len(names) - len(S.names))
        S.names = list(names)
        target, _ = regions.session_apply(S)
        new = S.names[before:]
        mins = int(time.time() - self.t0) // 60
        self.report({"INFO"}, f"AI Hybrid ({mins} min): {len(new)} new parts" + (": " + ", ".join(new) if new else ""))
        return {"FINISHED"}


classes = (FM_OT_regions_ai,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
