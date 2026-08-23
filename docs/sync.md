# Sync matches

When several matches show the same scene, register them into one Blender world.

## Overview

1. Match each still on its own (VP lines; Origin optional). Origins do **not** need to match across stills. If every still has locked/imported full K, you can instead omit VP lines and use the calibrated ground-only workflow below.
2. Choose an **Anchor** match — that world is shared space. Each match has **Enable sync for current match** (on by default); turn it off to exclude that still from Solve Sync / Diagnose / Refine Lenses.
3. Add landmarks for features visible in two or more stills (≥5 shared 2D picks), **or** link **Known 3D** Blender objects (≥3) and pick them in the other stills.
4. Pick each landmark in every still where it is visible. With the **Perspective Match** sidebar tab open and the view through the active match camera, **Ctrl+Cmd+A** (macOS; **Ctrl+Win+A** on Windows/Linux) starts **Pick in Active Match**. Optional: enable **Snap to AprilTag** (under **Pick in Active Match**) so a point click on a small or blurry marker snaps to the tag centre — the intersection of the dark quadrilateral's diagonals — without needing the marker to decode.
5. Optional: tag **On Ground** on point landmarks in the anchor, or rely on Known 3D, to pin absolute scale.
6. **Solve Sync** writes a rigid (or similarity) transform onto non-anchor root Empties. If a sync-enabled match still has no Pick Origin but has an **On Ground** pick, Sync auto-sets Origin from the earliest such pick (creation order) before solving — status shows `Auto origin: Match←landmark`.

Why not “any corresponding points”? Photogrammetry / SfM solves relative orientation and baseline *direction* from enough 2D↔2D matches — and so does this sync. Absolute baseline **length** stays free when dropping the second camera into an already-metric Blender world (classic stereo scale ambiguity), so pairwise pose validation does not use an absolute Blender-unit baseline cutoff. **Known 3D** Empties, On Ground picks, or a later ruler pin that one DOF; without them, a depth heuristic chooses a plausible scale.

A match does not need five landmarks in common with the anchor itself. Sync can register it through any already-registered match with at least five well-spread shared point landmarks, then carry that pose into the anchor world. This also covers cameras on the opposite side of a surface — for example, a camera below the ground plane looking upward. The strong bridge chooses the orientation / hemisphere, while all registered views choose its otherwise-ambiguous baseline scale and refinement. Joint adjustment still uses sparse observations elsewhere and can downweight them as outliers.

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
- **Font** toggle: show landmark names next to picks on the plate.
- Click a pick on the plate to select it in the list (while the **Perspective Match** sidebar tab is open). The selected pick draws in red. **On Ground** picks draw in magenta; Known 3D picks in cyan.
- **Duplicate**: copies type / On Ground / Use in Sync, clears Known 3D links, parallel links, picks, and solved positions.
- **Use in Sync**: exclude a landmark from Solve Sync / Diagnose without deleting picks. With **Landmark Empties** on, that also removes its helper from `PM_Sync_Landmarks`.

### Find AprilTags

Scans the active match still for **AprilTag 25h9 and 36h10** markers. Each tag's perspective-correct physical centre (the corner-diagonal intersection, corrected for active lens distortion) becomes a point pick. Landmark names include the family, such as `id005-25h9` and `id005-36h10`, so the same ID in both dictionaries creates distinct landmarks. Tag IDs are zero-padded to at least three digits (four for 36h10 IDs ≥ 1000). If a landmark whose name **starts with** the matching family-qualified name already exists (including older two-digit names such as `id05-25h9`), its pick for this match is updated and the name is rewritten to the current padding; otherwise a new point landmark is created. Needs OpenCV (`opencv-contrib-python-headless`); the button is hidden when neither dictionary is available.

When a printed marker is too small or blurry to decode, pick it by hand with **Snap to AprilTag** enabled. The click does not identify the ID; it only recenters onto the dark blotch (the inner black quad, not the white quiet zone). Needs OpenCV; the checkbox is hidden when the wheel is missing.

### Known 3D workflow

Model or place Empties in the anchor world → select them → Sync list **Landmarks from Selected** (auto-fills 2D on the anchor still) → in each other match, **Pick** those features in 2D. Add a few off-line 2D↔2D landmarks if the known points lie on one edge (kills spin-around-the-line ambiguity).

### Line landmarks

Add with the mesh icon next to +. Drag the same physical edge in each still — endpoints do **not** need to be the same 3D points, only the same infinite edge. Optional: assign two Empties as **Known 3D** / **Known 3D B** so the edge is metric. Mark **Is Parallel To** another Line landmark when two edges share a 3D direction.

Without Known 3D ends, a free line needs **three or more** stills — two views alone cannot constrain relative pose from lines. Ordinary point landmarks must be picked in **both** stills when Known 3D sit on one line. Expand **Pick Confidence** (collapsed by default, under **Pick in Active Match**) to set the next-pick default or per-still confidence.

**What “px” means:** For **point** landmarks, RMSE is how far the projected 3D Empty lands from your 2D pick. For **line** landmarks, it is how far each drawn **endpoint** sits from the projected infinite 3D line (perpendicular distance in pixels).

## Solve Sync and related tools

**Solve Sync** seeds pairwise pose then runs a joint bundle-adjustment over Empty transforms + landmarks (Cauchy-weighted). Options between Solve Sync and Refine Lenses:

- **Lock Rotation** — keep each Empty’s rotation on a 90° world-axis jump (identity, ±90°, 180° about X/Y/Z, including an X/Y swap); only solve translation/scale. Use when VP axes already match across stills so a free solve would only add a few degrees of noise.
- **Lock Translation** — keep Empty translation fixed; only solve rotation/scale.
- Both checked — leave cameras unmoved; only adjust 3D landmark / Empty positions.

**Diagnose** shows per-landmark RMSE without moving cameras; when error is high it also runs leave-one-out checks on the worst landmarks. **Clear** resets sync transforms.

**Refine Lenses** searches each unlocked match’s focal length (re-orients from VP lines at each trial) with a per-line VP residual prior and hard VP guardrails, then a coupled polish, then Solve Sync. Runs in a background thread — watch the progress slider, press **Esc** or **Cancel** to stop. The % field is the ± search window around current fx (default 18). Skips **1-point** matches and matches without enough VP lines. Disable unrelated matches or landmarks before refining a subset.

The eye icon on **Sync Matches** toggles landmark picks on the plate; **Landmark Empties** controls the 3D helpers after sync. With the **Perspective Match** sidebar tab open, click a pick on the plate — or select that landmark’s Empty / line helper in the viewport — to select it in the list (red overlay). Inactive **On Ground** picks are magenta; Known 3D are cyan. **Pick in Active Match** can still place or move the active landmark; clicking a different pick selects it instead of overwriting. **Snap to AprilTag** (point landmarks) looks around the click for a dark four-sided blotch with a brighter border and moves the pick to that quad's diagonal intersection. Per-match pick coordinates, confidence, and last-sync RMSE are under the collapsed **Pick Confidence** header.

## Debugging a bad or rejected sync

- **Rejected (~40+ px)** — no pose fits your picks. Status / **Diagnose** lists the worst landmarks — re-pick those features in *both* stills.
- **Plenty of picks, still rejected** — On Ground is load-bearing. Only landmarks that actually lie on the ground plane should be On Ground. A still that cannot lock is skipped so the others can still sync. After the remaining cameras lock, that still is retried as PnP against their triangulated 3D (floor tags alone if off-plane picks disagree). Diagnose / Solve Sync name the skipped match, and if one pick disagrees with the other stills they name that landmark (uncheck **Use in Sync** or re-pick it). A photo looking straight down at the ground is registered from those On Ground picks (plane homography); generic 2D↔2D pose is a poor fit there even when the tags are correct. If a portrait locked K was copied onto a landscape still of the same pixel count, Import YAML / copy keep the calibrated focal length (axes swap). Solve Sync sets fy=fx when they differ by more than 20%.
- **Accepted but camera looks wrong** with RMSE still a few–tens of px — wrong local minimum or soft constraints. Prefer **Diagnose**, fix the worst landmarks, **Clear**, then **Solve Sync** again. Matches without Origin but with On Ground picks get an auto Origin on Sync; if tilt persists, add an elevated (off-ground) landmark or a 4th ground pick.
- **One landmark huge, others fine** — that pick is mismatched (same ID / feature on a different physical point). Uncheck **Use in Sync** and re-run Diagnose. If a still was skipped, Diagnose names the pick on that still.
- **Many landmarks all high** — FOV or VP solve is likely off on one match; try **Refine Lenses**, or re-refine that camera manually.
- **Sync broke after adding one landmark** — turn off **Use in Sync** on the new one and Diagnose again.
- **Known 3D warn (Empty vs anchor pick)** — the Empty moved or the anchor camera changed; re-run **Landmarks from Selected**.
- **Landmarks jump on the plate when panning one still** — that match Empty was shrunk to a point (typical for a below-ground camera). Switch to the match or re-run **Solve Sync**; the camera should stay below-ground, but overlay picks stay on the photo.
- **One free line looks skewed vs another that should be parallel** — tag **Is Parallel To**. Prefer linking the bad free line to a better-fitting free line or a Known 3D edge.
