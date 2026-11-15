# Changelog

## 0.9.1 (2026-11-14)

- AI Hybrid Parts: Esc now stops the whole background job (the Blender render
  and the SAM 2 process), not only the wrapper script. Before, the children
  kept running and the GPU stayed busy until they finished.
- AI Hybrid Parts: the temporary run folder (a copy of the .blend and the
  renders, 100-400 MB) is deleted after the result is applied and after
  Esc. After a failure it is kept for its log and removed after a day.
- AI Hybrid Parts: the job is also stopped when Blender cancels the operator
  (file closed or reloaded).
- No machine-specific default paths: set the texture library and the AI
  environment in the add-on preferences.
- License text and SPDX headers in every file.
- Region splitting is unchanged: Auto Regions gives the same result as 0.9.0,
  face for face.

## 0.9.0

- AI Hybrid Parts (optional): SAM 2 proposes sub-parts on renders of every
  part; the shape split keeps every border.
- Smart Regions: auto split by shape or at creases, guard sub-parts, grip
  ferrules, blade fuller, blade edge strip, buried faces, material
  suggestions, interactive refinement.
- Material library, 24 procedural materials, weathering layers, the dial,
  saved looks, baking and game-engine export packs.
