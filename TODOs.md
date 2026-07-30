TODOs to be done:

- Sync: optional refine of non-anchor intrinsics / VP orientation from landmarks using frames you trust more (bundle-adjust K/R inside matches; keep anchor locked).
- Sync: landmark graph UI polish (rename, reorder, multi-select clear).
- Sync: better absolute-scale UX (known baseline length, ruler, or user slider) when no ground/Known 3D tags.
- Sync: optional snap Known 3D from mesh edit-mode verts without placing Empties first.

Done TODOs:
- let's reduce complexity of this app and remove feature that creates surfaces completely. I can make surfaces myself. I am in blender already. Leave Setting origin.
- Remove Measure scale and known length and all that as well. Just keep setting origin.
- VP Lines don't have any antialiasing and look very bad. IS there anything can be done about it?
- There is no need for project / output section. Leave opening of project fail as import only. No need to save back to it or export json. Rename opening project as import project.
- When exiting line editing, if one of them is selected - delect it.
- Make lines to be aligned with blender colors
- Opacity slider and handle opacity should become one. Also it doesn't affect green orgin indicator seems like.
- Green cross with circle that shows where origin is does not changes opacity when Opacity slider is used.
- Sync matches into one shared world via anchor + landmark graph (off-ground picks, rotate match Empties, shared origin assumed).
- Sync scale/translation from On Ground landmarks instead of requiring the same origin in every still.
- Sync via PnP: 3D from anchor ground picks, 2D only in other matches (no ground required in image 2).
- Sync via SfM-style essential pose from 2D↔2D landmarks; On Ground optional for absolute scale.
- Sync Known 3D from Blender Empties/objects (line verts + off-line 2D↔2D).
- Sync: per-observation pick confidence (High / Normal / Low) weights the solve.
- Sync: 2D↔2D line landmarks (+ optional Known 3D line endpoints).
- Sync: parallel-to relation between line landmarks (shared 3D direction).
