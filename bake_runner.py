# SPDX-License-Identifier: GPL-3.0-or-later
"""Run inside a separate `blender -b <copy.blend> --python bake_runner.py -- cfg.json`.

Loads bake.py directly (no add-on registration needed), bakes, writes the
report next to the maps, then exits. The user's open session is untouched.
"""

import importlib.util
import json
import os
import sys
import traceback


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    with open(argv[0], "r", encoding="utf-8") as f:
        cfg = json.load(f)
    log_path = os.path.join(cfg["out_dir"], "fm_bake_log.txt")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(msg):
            log_file.write(msg + "\n")
            log_file.flush()
            print(msg)
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location("fm_bake", os.path.join(here, "bake.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            log(f"Forge Materials bake: {len(cfg['objects'])} object(s) at {cfg['res']}px")
            report = mod.bake_objects(cfg, log)
            for e in report["errors"]:
                log("ERROR: " + e)
            log(f"Done in {report['seconds']}s")
        except Exception:  # report everything to the log for the UI
            log(traceback.format_exc())
            sys.exit(1)


main()
