# Sync matches

When several matches show the same scene, register them into one Blender world.

## Overview

1. Match each still on its own (VP lines; Origin optional). Origins do **not** need to match across stills. If every still has locked/imported full K, you can instead omit VP lines and use the calibrated ground-only workflow below.
2. Choose an **Anchor** match — that world is shared space. Each match has **Enable sync for current match** (on by default); turn it off to exclude that still from Solve Sync / Diagnose / Refine Lenses. For a non-anchor match whose current placement you trust, enable **Lock Pose in Sync**: its picks still constrain landmarks and the other cameras, but its current root transform (location, rotation, and scale) is held fixed. The Anchor is already fixed, so the checkbox is disabled there. After **Solve Sync**, the Enable row shows **Synced** or **Not synced** for the active match: whether this still was registered in the last run. A later run that skips it (or **Clear**) removes the check.
3. Add landmarks for features visible in two or more stills (≥5 shared 2D picks), **or** link **Known 3D** Blender objects (≥3) and pick them in the other stills. Optional: pair one-sided features with **Is Mirror Of** and one scene **Mirror Empty**.
4. Pick each landmark in every still where it is visible. With the **Perspective Match** sidebar tab open and the view through the active match camera, **Ctrl+Cmd+A** (macOS; **Ctrl+Win+A** on Windows/Linux) starts **Pick in Active Match**. Optional: enable **Snap to AprilTag** (under **Pick in Active Match**) so a point click on a small or blurry marker snaps to the tag centre — the intersection of the dark quadrilateral's diagonals — without needing the marker to decode.
5. Optional: tag **On Ground** on point landmarks in the anchor, or rely on Known 3D, to pin absolute scale.
6. **Solve Sync** writes a rigid (or similarity) transform onto non-anchor root Empties. If an unlocked, sync-enabled match still has no Pick Origin but has an **On Ground** pick, Sync auto-sets Origin from the earliest such pick (creation order) before solving — status shows `Auto origin: Match←landmark`. Pose-locked matches keep their existing private camera placement instead.

Use **Lock Pose in Sync** after a good solve when you want to add or tune landmarks without letting a trusted match drift. The lock uses the live root Empty transform at the start of each operation and applies to Solve Sync, Diagnose, and the sync solves inside Refine Lenses. It is an exact freeze, not a warm start: an unlocked match is solved from its correspondences, without treating its current root placement as a pose prior. Leave it unlocked when Sync should refine that camera. **Clear Sync** still resets every root transform explicitly.

Why not “any corresponding points”? Photogrammetry / SfM solves relative orientation and baseline *direction* from enough 2D↔2D matches — and so does this sync. Absolute baseline **length** stays free when dropping the second camera into an already-metric Blender world (classic stereo scale ambiguity), so pairwise pose validation does not use an absolute Blender-unit baseline cutoff. **Known 3D** Empties, On Ground picks, or a later ruler pin that one DOF; without them, a depth heuristic chooses a plausible scale.

A match does not need five landmarks in common with the anchor itself. Sync can register it through any already-registered match with at least five well-spread shared point landmarks, then carry that pose into the anchor world. This also covers cameras on the opposite side of a surface — for example, a camera below the ground plane looking upward. The strong bridge chooses the orientation / hemisphere, while all registered views choose its otherwise-ambiguous baseline scale and refinement. If that cheap two-view pose disagrees with the rest of the graph, Sync keeps another candidate that still fits in pixels. Joint adjustment still uses sparse observations elsewhere and can downweight them as outliers.

### Calibrated ground-only workflow (no VP lines)

When the anchor has no usable VP solve, **Solve Sync** and **Diagnose** can initialize its ground frame directly from calibrated views:

1. Give the anchor and at least two supporting matches (three images total) a complete locked K — Manual FOV, imported camera-info YAML, or 1-point mode. Imported `fx`, `fy`, `cx`, `cy`, distortion, and plate dimensions are all used.
2. Mark at least four well-spread, non-collinear point landmarks **On Ground** and pick the same landmarks in each image. Five or six are recommended so a bad pick can be rejected.
3. Use images taken from different positions. A pure camera rotation has no plane-normal cue, and two images alone have a genuine two-solution ground-plane ambiguity.
4. Press **Solve Sync**. Sync infers the anchor Z/vertical and compatible orientations for the other unsolved matches, auto-picks each missing Origin from its first ground landmark, then runs the normal landmark solve.

The ground plane determines Z but has no preferred compass direction, so anchor X/Y yaw is chosen deterministically from the previous camera orientation. All inferred matches share that choice. The usual scale ambiguity still applies; without Known 3D or another metric constraint, the initial camera height is conventional rather than measured.

## Landmarks

Each landmark keeps a stable `item_id` plus a `creation_index` (add order). UI helpers:

- **A–Z** toggle: alphabetical by name vs original add order (display only).
- **Filter** toggle: show only landmarks with a pick defined in the active match.
- **Font** toggle: show landmark names next to picks on the plate.
- Click a pick on the plate to select it in the list (while the **Perspective Match** sidebar tab is open). The selected pick draws in red. **On Ground** picks draw in magenta; Known 3D picks in cyan.
- **Duplicate**: copies type / On Ground / Use in Sync / Sync Weight, clears Known 3D links, parallel links, mirror links, picks, and solved positions.
- **Use in Sync**: exclude a landmark from Solve Sync / Diagnose without deleting picks. With **Landmark Empties** on, that also removes its helper from `PM_Sync_Landmarks`.
- **Sync Weight**: how strongly this landmark pulls Solve Sync (default 1). Raise it on a couple of well-placed picks that sit far from the others so a cluster of easier landmarks cannot ignore them. Combines with per-still **Pick Confidence** (High ×4, Low ×0.25). Boosted landmarks also skip the usual “this pick looks like an outlier” downweight. Weight influences pose refinement and candidate ranking; camera acceptance and mismatched-pick diagnostics remain in raw image pixels.

### Find AprilTags

Scans the active match still for **AprilTag 25h9 and 36h10** markers. Each tag's perspective-correct physical centre (the corner-diagonal intersection, corrected for active lens distortion) becomes a point pick. Landmark names include the family, such as `id005-25h9` and `id005-36h10`, so the same ID in both dictionaries creates distinct landmarks. Tag IDs are zero-padded to at least three digits (four for 36h10 IDs ≥ 1000). If a landmark whose name **starts with** the matching family-qualified name already exists (including older two-digit names such as `id05-25h9`), its pick for this match is updated and the name is rewritten to the current padding; otherwise a new point landmark is created. Needs OpenCV (`opencv-contrib-python-headless`); the button is hidden when neither dictionary is available.

When a printed marker is too small or blurry to decode, pick it by hand with **Snap to AprilTag** enabled. The click does not identify the ID; it only recenters onto the dark blotch (the inner black quad, not the white quiet zone). Needs OpenCV; the checkbox is hidden when the wheel is missing.

### Known 3D workflow

Model or place Empties in the anchor world → select them → Sync list **Landmarks from Selected** (auto-fills 2D on the anchor still) → in each other match, **Pick** those features in 2D. Add a few off-line 2D↔2D landmarks if the known points lie on one edge (kills spin-around-the-line ambiguity). The same Known 3D links plus picks on **this** still also feed **Use Known 3D** in the Camera section — pick on the photo, not the auto-projected positions, when you want that single-camera polish. If the CAD is a strong guide but not exact, raise **Known 3D Slack** so Solve Sync can ease those points a little toward the picks.

### Mirror pairs

A point landmark can name another as **Is Mirror Of** — the same feature on the opposite side of a symmetric object. The pair is stored on both landmarks; selecting either side shows the other, and clearing **None** on one side clears the other. Each side is picked only where it is visible. The magic-wand button next to the dropdown fills the partner when this landmark's name ends with **left** or **right** and another point landmark uses the swapped name. Pairwise registration still needs ordinary shared points; the mirror constraint is used in joint BA.

One scene **Mirror Empty** (below Ground Slack / Known 3D Slack) is the plane for every pair. Place it on the midline. **Plane** chooses which local face is the mirror (YZ by default: local X is the normal). **Mirror Slack** (0 pins the plane) lets that plane slide along its normal if the Empty was slightly off. The Empty is not moved.

If Is Mirror Of is set but Mirror Empty is empty, Solve Sync ignores those pairs and says so in the status line.

### Line landmarks

Add with the mesh icon next to +. Drag the same physical edge in each still — endpoints do **not** need to be the same 3D points, only the same infinite edge. Optional: assign two Empties as **Known 3D** / **Known 3D B** so the edge is metric. **Is Parallel To** can constrain the edge to shared-world **X Axis**, **Y Axis**, or **Z Axis**, or to another Line landmark that shares its 3D direction.

Without Known 3D ends, a free line needs **three or more** stills — two views alone cannot constrain relative pose from lines. Ordinary point landmarks must be picked in **both** stills when Known 3D sit on one line. Expand **Pick Confidence** (collapsed by default, under **Pick in Active Match**) to set the next-pick default or per-still confidence; it multiplies the landmark **Sync Weight**.

**What “px” means:** For **point** landmarks, RMSE is how far the projected 3D Empty lands from your 2D pick. For **line** landmarks, it is how far each drawn **endpoint** sits from the projected infinite 3D line (perpendicular distance in pixels).

## Solve Sync and related tools

**Solve Sync** seeds pairwise pose then runs a joint bundle-adjustment over Empty transforms + landmarks (Huber-weighted, with extra influence on poorly covered regions of each still so a cluster of central picks cannot ignore a few near the edge that pin camera distance). Raise **Sync Weight** on a landmark when those automatic boosts are not enough. Pairwise growth starts from the geometrically strongest still pair (spread, overlap, parallax, pair RMSE) and adds the easiest next camera — never alphabetical names, and not every still vs the Anchor just because it shares five picks. The Anchor remains the shared world. 3D landmarks are triangulated from all registered rays, with near-parallel views downweighted, behind-camera views dropped, and a short reprojection polish. Options between Solve Sync and Refine Lenses:

- **Lock Rotation** — keep each Empty’s rotation on a 90° world-axis jump (identity, ±90°, 180° about X/Y/Z, including an X/Y swap); only solve translation/scale. Use when VP axes already match across stills so a free solve would only add a few degrees of noise.
- **Lock Translation** — keep Empty translation fixed; only solve rotation/scale.
- Both checked — leave cameras unmoved; only adjust 3D landmark / Empty positions.
- **Ground Slack** — how far On Ground landmarks may sit off Z=0 (scene units). 0 pins them to the floor raycast. The default (0.02) is enough for plank cup / tag thickness on a boarded floor.
- **Known 3D Slack** — how far Known 3D point landmarks may sit off their Empty (scene units). 0 (the default) pins them. A small value is a spring toward the Empty while each still's 2D pick pulls the point along that camera's ray, so CAD that is slightly wrong can share the error with the cameras instead of stretching the overlay. Linked Empties stay put; **Landmark Empties** show the eased positions. Known 3D lines stay pinned. **Use Known 3D** (Camera) still treats the Empty as fixed. A point that is both Known 3D and **On Ground** uses the tighter of the two slacks for Z.
- **Mirror Empty / Plane / Mirror Slack** — one object whose chosen local face is the shared mirror for every **Is Mirror Of** pair. Slack 0 pins the plane to the Empty; a small value lets it slide along the normal. The Empty is not moved.

**Diagnose** measures sync quality without moving cameras, then opens a self-contained local HTML report in the default browser. The report leads with actionable problems, shows the camera-overlap graph, names the best available registration route and its shared-point deficit, lists every match and its RMSE, and provides a searchable error-ranked landmark table. Technical solver text and constraint counts stay available in collapsible sections. **Open Report** reopens the newest temporary report; **Export** saves that single portable HTML file permanently. Reports use Blender's configured temporary directory, contain no remote resources, and are not uploaded.

When error is high, Diagnose also runs leave-one-out checks on the worst landmarks. It runs in the background: the status line shows the current solve stage, and **Esc** or **Cancel** stops it. Leave-one-out keeps the camera graph accepted by the base solve, so a rejected still is not globally re-registered five more times. HTML reports belong to **Diagnose** only; **Solve Sync** keeps its normal Blender status and does not create or open a report. **Clear** resets sync transforms and forgets the current report link. Diagnose and Solve Sync cache each still-pair pose so a second run skips the expensive pairwise search when those two cameras' shared picks, Known 3D, and private K/pose are unchanged; **Clear** drops that cache. Independent still pairs on the first run are solved in parallel.

**Refine Lenses** searches focal length to lower sync RMSE. The **Same Lens** checkbox and **%** field sit above the button. **Same Lens** (on by default) applies one scale to every still — use this when they share a physical camera / imported YAML; it does not need VP lines. Off restores a per-still search (re-orients from VP lines, skips 1-point / weak-VP stills). Coupled polish and Solve Sync follow. Runs in a background thread — watch the progress slider, press **Esc** or **Cancel** to stop. The % field is the ± search window around current fx (default 18). Disable unrelated matches or landmarks before refining a subset. Matches in **Adjusted Camera** mode are skipped so the button stays available for the others.

**Iterate Known 3D** (beside Refine Lenses) repeats **Auto from VPs** with **Use Known 3D** then **Solve Sync** until joint RMSE stops falling. That is the hand-click loop that still moves FOV and camera after a match Empty has been registered: Known 3D is expressed in the private frame, so a new root transform changes the pins. Eligible stills need Use Known 3D, four Known 3D picks, and enough VP lines. Pose-locked non-anchor matches and Adjusted Camera stills are skipped. Esc / Cancel keeps the last improvement (at most eight rounds).

The eye icon on **Sync Matches** toggles landmark picks on the plate (same pattern as Vanishing Point Lines, Origin, and Camera); **Landmark Empties** controls the 3D helpers after sync. **Hide Origin Empty** is per match and hides that Origin Empty in the viewport (camera and collection stay visible); it stays in sync with the Origin's Outliner visibility. With the **Perspective Match** sidebar tab open, click a pick on the plate — or select that landmark’s Empty / line helper or Known 3D object in the viewport — to select it in the list (red overlay). Inactive **On Ground** picks are magenta; Known 3D are cyan. **Pick in Active Match** can still place or move the active landmark; clicking a different pick selects it instead of overwriting. **Snap to AprilTag** (point landmarks) looks around the click for a dark four-sided blotch with a brighter border and moves the pick to that quad's diagonal intersection. Per-match pick coordinates, confidence, and last-sync RMSE are under the collapsed **Pick Confidence** header.

## Debugging a bad or rejected sync

- **Rejected (~40+ px)** — no pose fits your picks. Status / **Diagnose** lists the worst landmarks — re-pick those features in *both* stills.
- **Plenty of picks, still rejected** — On Ground is load-bearing. Only landmarks that actually lie on the ground plane should be On Ground. A still that cannot lock is skipped so the others can still sync. After the remaining cameras lock, that still is retried as PnP against their triangulated 3D (floor tags alone if off-plane picks disagree). Diagnose / Solve Sync name the skipped match, and if one pick disagrees with the other stills they name that landmark (uncheck **Use in Sync** or re-pick it). A photo looking straight down at the ground is registered from those On Ground picks (plane homography); generic 2D↔2D pose is a poor fit there even when the tags are correct. If a portrait locked K was copied onto a landscape still of the same pixel count, Import YAML / copy keep the calibrated focal length (axes swap). Solve Sync sets fy=fx when they differ by more than 20%.
- **Accepted but camera looks wrong** with RMSE still a few–tens of px — wrong local minimum or soft constraints. Prefer **Diagnose**, fix the worst landmarks, **Clear**, then **Solve Sync** again. If a few picks far from the main cluster look sacrificed while the rest sit perfectly, raise those landmarks' **Sync Weight** (try 4–8) and solve again. Matches without Origin but with On Ground picks get an auto Origin on Sync; if tilt persists, add an elevated (off-ground) landmark or a 4th ground pick.
- **Flipped the object for underside photos** — On Ground is one shared plane. Table tags from the flipped session must not be On Ground if the original table tags already pin Z=0. Side tags glued to the object *are* the same 3D points and do connect the graphs. After a physical flip, a camera that photographed the underside is placed *below* the original ground looking up (object frame), not above the table in room coordinates.
- **List shows ~1px but the Empty is far from the pick** — the px number is RMSE across the stills that participated in the last solve, not the currently viewed still. Landmarks that only appear on recovered / hanging stills used to keep a stale number; Solve Sync now triangulates those tags and poses the hanging still from them. If the Empty is still off, switch to that still and compare this match's residual under **Pick Confidence**.
- **One landmark huge, others fine** — that pick is mismatched (same ID / feature on a different physical point). Uncheck **Use in Sync** and re-run Diagnose. If a still was skipped, Diagnose names the pick on that still.
- **Many landmarks all high** — FOV or VP solve is likely off on one match; try **Iterate Known 3D** (Use Known 3D on that still) or **Refine Lenses**, or re-refine that camera manually.
- **Sync broke after adding one landmark** — turn off **Use in Sync** on the new one and Diagnose again.
- **Known 3D warn (Empty vs anchor pick)** — the Empty moved or the anchor camera changed; re-run **Landmarks from Selected**.
- **Landmarks jump on the plate when panning one still** — that match Empty was shrunk to a point (typical for a below-ground camera). Switch to the match or re-run **Solve Sync**; the camera should stay below-ground, but overlay picks stay on the photo.
- **A free line should follow a world axis** — set **Is Parallel To** to X Axis, Y Axis, or Z Axis. For two arbitrary parallel edges, link the bad free line to a better-fitting free line or a Known 3D edge.
