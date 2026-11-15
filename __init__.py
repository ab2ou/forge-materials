# SPDX-License-Identifier: GPL-3.0-or-later
"""Forge Materials: materials from your PBR library, procedural
metals, procedural weathering (edge wear, dirt, rust) and game-ready baking.

The add-on never modifies mesh geometry or UVs.
"""

import sys

# When the extension is reinstalled/updated in a running Blender, the old
# submodules stay cached in sys.modules and would be used instead of the new
# files. Drop them so this (re)load always imports the current code.
for _k in [k for k in sys.modules if k.startswith(__name__ + ".")]:
    del sys.modules[_k]

import bpy
from bpy.props import EnumProperty, StringProperty

from . import ai_op, dial, regions, ui

_DEFAULT_LIB = ""


class FM_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    library_path: StringProperty(name="Texture Library", subtype="DIR_PATH",
                                 default=_DEFAULT_LIB,
                                 description="Folder of PBR materials (Poly Haven layout or "
                                             "one folder per material)")
    preferred_res: EnumProperty(name="Preferred Resolution", default="HIGHEST", items=[
        ("HIGHEST", "Highest Available", ""), ("1k", "1K", ""), ("2k", "2K", ""),
        ("4k", "4K", ""), ("8k", "8K", "")])
    ai_path: StringProperty(name="AI Environment", subtype="DIR_PATH", default="",
                            description="Folder with the SAM 2 environment (venv + checkpoints) for "
                                        "AI Hybrid Parts. Empty: fm_ai next to the add-on")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "library_path")
        col.prop(self, "preferred_res")
        col.prop(self, "ai_path")
        col.operator("fm.scan_library", icon="FILE_REFRESH")


def register():
    bpy.utils.register_class(FM_Preferences)
    ui.register()
    regions.register()
    ai_op.register()
    dial.register()


def unregister():
    dial.unregister()
    ai_op.unregister()
    regions.unregister()
    ui.unregister()
    bpy.utils.unregister_class(FM_Preferences)
