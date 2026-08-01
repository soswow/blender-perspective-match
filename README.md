# Perspective Match

Perspective Match is a Blender 5.1 extension for matching perspective cameras to stills without leaving Blender. Keep several matches in one `.blend`, draw colored vanishing-point line bundles in camera view, solve each camera independently, pick a world origin on the ground plane, and optionally synchronise matches into one shared space with a landmark graph.

The extension is a native port of the manual workflow from Perspective Match Studio. It does not require Electron, Node.js, a Python sidecar, PyTorch, GeoCalib, OpenCV, or network access.

## Features

- Create multiple match cameras in one scene; each still keeps its own calibration session.
- Switch the active match from a dropdown to edit that session and view through its camera.
- Load a still directly as a Blender camera background on the active match only.
- Choose 1-, 2-, or 3-point perspective.
- Draw, select, edit, and delete axis-colored VP segments in camera view.
- Extend VP guides, horizon, and VP markers past the plate to finite vanishing points (capped when lines are nearly parallel).
- Derive a missing horizontal VP from an orthogonal pair so the horizon still draws with only one explicit horizontal bundle.
- Draw VP continuations as axis-colored dashes, and the horizon as red / empty / green / empty dashes.
- Robust length-weighted VP fitting with Huber outlier reduction.
- Solve camera orientation and, when constrained, focal length from orthogonal VPs.
- Lock and edit horizontal FOV manually, including underconstrained 1-point shots.
- Solve an off-center principal point from three finite orthogonal VPs.
- Show per-plane FOV estimates and angular consistency diagnostics.
- Pick a ground origin so the matched camera sits relative to world origin.
- Synchronise multiple matched cameras into one shared world with a landmark graph: 2D↔2D picks and/or **Known 3D** Blender objects (Empties at verts on a modeled edge), optional On Ground, choose an anchor, solve similarities onto match root Empties.
- Estimate Fitzgibbon one-parameter radial distortion from three or more concurrent segments.
- With **Estimate Distortion**, automatically generate and show an expanded undistorted PNG (NumPy only—no OpenCV).
- Apply display-only exposure/contrast to a sibling ``*-pm-view.png`` plate (undistorted uses the same lit image).
- Save match state in the `.blend` and import desktop-compatible `.pmproj` files.

## Requirements

- Blender 5.1 or newer.
- A reference image with clear straight edges for the VP workflow.

The extension uses the NumPy bundled with Blender. It has no third-party runtime packages.

## Installation

### Build and install

1. Build the extension:

   ```sh
   ./build-extension.sh
   ```

   Override Blender's location when necessary:

   ```sh
   BLENDER_BIN="/path/to/blender" ./build-extension.sh
   ```

2. In Blender, open **Edit → Preferences → Get Extensions**.
3. Use **Install from Disk** and select the generated `.zip`.
4. Enable **Perspective Match**.

### Development install (edit → reload)

Point Blender's installed extension at this git checkout with a symlink (no ZIP copy):

```sh
./link-dev.sh
```

That replaces `~/Library/Application Support/Blender/5.1/extensions/user_default/match_perspective` with a link to this repo. Enable **Perspective Match** once if it is not already on.

Daily loop:

1. Edit and save in the editor.
2. In Blender: click **Reload Perspective Match** at the bottom of the sidebar (or **F3 → Reload Perspective Match**). Prefer this over System → Reload Scripts—that often leaves panels and PropertyGroups on stale class objects.
3. Test. Watch the system console for `Perspective Match: reloaded from disk` (the button only queues the reload; the tear-down runs a moment later so Blender does not crash).

Exit any running Draw / Pick Origin modal before reloading. After adding, renaming, or removing RNA properties on `PMSession` / `PMWorkspace`, **restart Blender**—property schema changes are not reliably hot-reloadable and can crash even with a deferred reload. If you still see duplicate panels or `already registered`, disable/re-enable the extension or restart.

Re-run `./link-dev.sh` if you later **Install from Disk** and Blender overwrites the symlink with a ZIP extract.

ZIP builds (`./build-extension.sh`) are only needed for packaging or a clean install test.

## Workflow

The extension lives in the 3D View sidebar under the **Perspective Match** tab.

### Match cameras

Each match is a collection with a root Empty that owns the session data, plus a child camera:

```text
PM_<image>
  PM_<image>_Origin      # session lives here (Object.pm_session)
    PM_<image>_Camera
```

1. Click **New Match Camera** to create an empty match and activate it.
2. Use **Active Match** to switch which session the sidebar edits. Switching also sets the scene camera and enters that camera view.
3. Click **Unload** to detach the sidebar from editing without deleting objects.
4. Delete a match by removing its Empty/camera in the Outliner; the dropdown prunes dead entries automatically.

Selecting objects in the viewport does **not** change the active match—only New, Unload, and the dropdown do.

### 1. Load a reference

With a match active, choose **Open Image** or **Import Project** (`.pmproj`). The still binds to the **active** match only; other matches stay untouched. If no match is active, Open Image / Import Project creates one first.

After bind, the hierarchy is renamed from `PM_Match_###` to `PM_<image stem>` when that name is free. The camera becomes the scene camera, the still is attached with **Stretch** frame mapping, and render dimensions match the image.

### 2. Choose perspective

- **1 Point:** draw Z verticals and Y depth lines. FOV stays manual.
- **2 Point:** draw X and Y horizontal bundles (finite VPs). Z uprights are not used as a vanishing point.
- **3 Point:** draw any two axes with at least two lines each; the missing world axis is derived from orthogonality. All three axes also allow solving principal point.

Axis mapping matches Blender’s gizmos:

- Red → Blender X
- Green → Blender Y
- Blue → Blender Z (up)

### 3. Draw VP lines

1. Choose the colored axis.
2. Click **Draw / Edit Lines**.
3. Drag over straight edges that belong to that axis (clicks must land inside the camera frame).
4. Click a line to select it; drag either endpoint handle to edit.
5. Press **Delete/Backspace** to remove the selected line.
6. Press **Esc** (or right-click) to leave the tool.

While the tool is active, left-clicks in the 3D View belong to Perspective Match (object selection is blocked). Clicking **Draw / Edit Lines** or **Pick Origin** again refreshes the active tool instead of getting stuck. If you orbit out of camera view, the next line/origin click switches back to the match camera.

The camera refines whenever enough required lines exist. **Auto from VPs** allows orthogonal VP pairs to solve FOV. Enable **Manual FOV**, set the horizontal angle, and apply it to lock focal length while continuing to solve orientation.

Middle mouse and the wheel retain normal Blender navigation while a match tool is active. Sidebar sections use Blender’s native collapsible panels (**View** starts collapsed). Overlay guides only draw in **camera view** of the active match — choosing a match from the dropdown (or **View Match Camera**) rehydrates the plate/lens after opening a `.blend` and enters that view.

### 4. Set origin

Use **Pick Origin** to choose a ground point that should become world origin. **Apply Origin** recomputes camera position from the current solve; clear removes the pick.

Build your own floor/wall geometry in Blender once the camera is matched—the extension no longer creates surface meshes or measures known lengths.

### 5. Sync matches

When several matches show the same scene, register them into one Blender world:

1. Match each still on its own (VP + origin). Origins do **not** need to match.
2. Choose an **Anchor** match — that world is shared space.
3. Add landmarks for features visible in two or more stills (≥5 shared 2D picks), **or** link **Known 3D** Blender objects (≥3) and pick them in the other stills. Each landmark keeps a stable `item_id` plus a `creation_index` (add order). The **A–Z** toggle beside the list (below + / line / import / −) is a depressed/undepressed control: on = alphabetical by name, off = original add order. Sort only changes the list display — not storage order.
4. Pick each landmark in every still where it is visible. **Point** landmarks: click the feature. **Line** landmarks: drag a segment along the same edge (no need for a shared point). **Known 3D** Empties: auto-projected on the anchor; pick them in other matches only. For a metric edge, set **Known 3D** + **Known 3D B** on a Line landmark. Free lines (no Empties) need the edge in **≥3 stills** to constrain pose. Ordinary point landmarks must be picked in **both** stills when Known 3D sit on one line. Set **Pick Confidence** before clicking.
5. Optional: tag **On Ground** on point landmarks in the anchor, or rely on Known 3D points/lines, to pin absolute scale. Without metric cues, sync still recovers orientation and baseline *direction* with a depth heuristic.
6. **Solve Sync** writes a rigid transform (`R`, `t`, scale 1) onto non-anchor root Empties. Use **Diagnose** first to see per-landmark RMSE without moving cameras; **Clear** resets sync transforms. The eye icon on **Sync Matches** toggles landmark picks on the plate (same pattern as VP Lines); **Landmark Empties** still controls the 3D helpers after sync.

Why not “any corresponding points”? Photogrammetry / SfM does solve relative orientation and baseline *direction* from enough 2D↔2D matches — and so does this sync. What stays free is absolute baseline **length** when dropping the second camera into an already-metric Blender world from the first match (classic stereo scale ambiguity). **Known 3D** Empties, On Ground picks, or a later ruler pin that one DOF; without them, a depth heuristic chooses a plausible scale.

**Known 3D workflow (line / mesh verts):** Model or place Empties in the anchor world → select them → Sync list **Landmarks from Selected** (auto-fills 2D on the anchor still) → in each other match, **Pick** those features in 2D. Add a few off-line 2D↔2D landmarks if the known points lie on one edge (kills spin-around-the-line ambiguity).

**Line landmarks:** Add with the mesh icon next to +. Drag the same physical edge in each still — endpoints do **not** need to be the same 3D points, only the same infinite edge. Drag existing endpoints to edit (same as VP lines); click empty image to redraw. Optional: assign two Empties as **Known 3D** / **Known 3D B** so the edge is metric (works with two stills; ≥3 Known 3D lines can register pose alone). Mark **Is Parallel To** another Line landmark when two edges share a 3D direction: after pose is solved, free-line meshes in that family are **forced to share one direction** (Known 3D edges anchor the direction when present). Status reports `parallel Δ N° (direction locked)` when enforcement worked. Camera pose / reject uses **point** RMSE only — a Parallel lock that misses the 2D drawings shows as `parallel line miss (…px)` instead of rejecting sync. Without Known 3D ends, a free line needs **three or more** stills — two views alone cannot constrain relative pose from lines.

**What “px” means:** For **point** landmarks, RMSE is how far the projected 3D Empty lands from your 2D pick (image pixels). For **line** landmarks, it is how far each drawn **endpoint** sits from the projected infinite 3D line (perpendicular distance in pixels) — not whether the segment ends match in 3D. High line px after Parallel usually means the drawings disagree with the locked direction; the 3D edges can still look correctly parallel.

**Debugging a bad or rejected sync:**

- **Rejected (~40+ px)** means no pose fits your picks. Status / **Diagnose** lists the worst landmarks — re-pick those features in *both* stills (same physical point).
- **Accepted but camera looks wrong** with RMSE still a few–tens of px: wrong local minimum or soft constraints. Prefer **Diagnose**, fix the worst landmarks, **Clear**, then **Solve Sync** again. A “WARN high error” line means treat the pose as suspicious.
- **One landmark huge, others fine:** that pick is mismatched (wrong corner / wrong still).
- **Many landmarks all high:** FOV or VP solve is likely off on one match — re-refine that camera, then re-project Known 3D (**Landmarks from Selected**).
- **Known 3D warn (Empty vs anchor pick):** the Empty moved or the anchor camera changed after auto-project — re-run **Landmarks from Selected**.
- **Uncertain pick:** set that still's confidence to **Low** (or pick with Pick Confidence = Low). High-confidence picks dominate the pose; Low ones still help but the landmark Empty is allowed to miss them more.
- **One free line looks skewed vs another that should be parallel:** tag **Is Parallel To** — Solve Sync keeps the camera pose and re-fits that line mesh to the shared direction. Prefer linking the bad free line to a better-fitting free line or a Known 3D edge. 2D RMSE on the bad line may stay similar; the 3D edge orientation is what changes.

## Lens distortion

Enable **Estimate Distortion** in the **Camera** section (Manual FOV must be off). After a successful VP refine with ≥3 concurrent segments on one axis, the solver estimates Fitzgibbon λ and **automatically** generates/switches to an undistorted plate. **Original Plate** turns estimation off, clears λ, re-solves, and restores the source still. Plate file paths are logged to the console, not the sidebar status line.

## Project compatibility

The extension imports version-1 Perspective Match Studio projects:

```json
{
  "kind": "perspective-match-project",
  "version": 1,
  "savedAt": "...",
  "session": {}
}
```

The referenced image remains external. Relative image paths are resolved from the `.pmproj` directory. Surfaces and scale fields in imported projects are ignored; origin is still applied when present. Session data for each match is stored on its root Empty in the `.blend`.

## Limitations

- Initial FOV is manual. GeoCalib automatic FOV/gravity estimation is intentionally omitted.
- Vanishing lines are not detected automatically.
- The workflow assumes square pixels and zero skew.
- 1-point perspective cannot determine focal length from VP geometry alone.
- Distortion uses one radial division parameter, not Blender's full tracking-camera lens models.
- Cropped, anamorphic, curved, or CGI plates can require manual FOV and principal-point judgment.
- Sync solves relative pose from 2D↔2D and/or Known 3D Blender objects. Absolute baseline vs the metric anchor world needs Known 3D, On Ground, or the depth heuristic.
- Sync does not yet refine non-anchor lenses or VP orientation from landmarks.
- Display-only exposure and contrast controls from the desktop app are not reproduced; use Blender's image/color-management tools.
- Projects are import-only; there is no Save Project or camera JSON export.

## Project layout

```text
match_perspective/
  blender_manifest.toml  # Blender extension metadata
  __init__.py             # Registration
  properties.py           # Object sessions + scene workspace controller
  core.py                 # VP, focal, distortion, placement, and surface math
  sync.py                 # Multi-match landmark graph registration
  scene.py                # Camera/background integration and coordinate mapping
  overlay.py              # Camera-view GPU drawing
  operators.py            # File, solve, and modal interaction operators
  panel.py                # 3D View sidebar workflow
  distortion.py           # NumPy image remapping
  project_io.py           # .pmproj import
  validate_addon.py       # Headless Blender smoke test
  tests/test_core.py      # Pure geometry regressions
  tests/test_sync.py      # Landmark sync regressions
  link-dev.sh             # Symlink checkout into Blender 5.1 for live reload
  build-extension.sh      # Validate and build installable zip
```

Python modules use snake_case because they are importable packages; user-facing files and generated assets use dash-separated names where practical.

## Development

Live coding uses the symlink from `./link-dev.sh` (see **Development install** above). Prefer that over rebuild/reinstall while iterating.

Compile-check Python:

```sh
python3 -m compileall -q .
```

Run the Blender smoke test:

```sh
"/Applications/Blender 5.1.app/Contents/MacOS/blender" \
  --factory-startup -b --python validate_addon.py
```

Validate and build a distributable ZIP when needed:

```sh
./build-extension.sh
```

The smoke test covers registration, multi-match create/switch/unload/prune, VP solve, camera projection, origin placement, project import, undistorted plates, and cleanup.

## License

GPL-3.0-or-later.
