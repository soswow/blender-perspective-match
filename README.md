# Perspective Match

Perspective Match is a Blender 5.1 extension for matching perspective cameras to stills without leaving Blender. Keep several matches in one `.blend`, draw colored vanishing-point line bundles in camera view, solve each camera independently, pick a world origin on the ground plane, and optionally synchronise matches into one shared space with a landmark graph.

The extension is a native port of the manual workflow from Perspective Match Studio. It does not require Electron, Node.js, a Python sidecar, PyTorch, GeoCalib, or network access. Core matching uses only NumPy (bundled with Blender). **Find AprilTags** ships OpenCV (aruco) as a bundled wheel.

## Features

- Create multiple match cameras in one scene; each still keeps its own calibration session.
- Switch the active match from a dropdown to edit that session and view through its camera.
- Remember each match’s last camera-view zoom and pan when switching or saving.
- Load a still directly as a Blender camera background on the active match only.
- Replace the plate on an existing match without clearing VP lines, origin, or landmarks (same pixel size).
- Choose 1-, 2-, or 3-point perspective.
- Draw, select, edit, and delete axis-colored VP segments in camera view.
- Optionally label each VP segment with its residual (px) to the current camera.
- Extend VP guides, horizon, and VP markers past the plate to finite vanishing points (capped when lines are nearly parallel).
- Derive a missing horizontal VP from an orthogonal pair so the horizon still draws with only one explicit horizontal bundle.
- Draw VP continuations as axis-colored dashes, and the horizon as red / empty / green / empty dashes.
- Robust length-weighted VP fitting with Huber outlier reduction.
- Solve camera orientation and, when constrained, focal length from orthogonal VPs.
- With known full intrinsics (Manual FOV / Import YAML), recover orientation from a single VP line per axis in 3-point mode.
- Lock and edit horizontal FOV manually, including underconstrained 1-point shots.
- Import ROS ``camera_info`` YAML intrinsics (**Import YAML**): locks Manual FOV from ``fx``, sets principal point from ``cx``/``cy``, and applies ``fitzgibbon_lambda`` when present. Plumb-bob ``D`` is skipped.
- Solve an off-center principal point from three finite orthogonal VPs, or set it with **Manual PP Offset** (drag or type offsets; violet crosshair when off-center).
- Show per-plane FOV estimates, angular consistency, and VP-line RMSE diagnostics.
- Pick a ground origin so the matched camera sits relative to world origin.
- Synchronise multiple matched cameras into one shared world with a landmark graph: 2D↔2D picks and/or **Known 3D** Blender objects (Empties at verts on a modeled edge), optional On Ground, choose an anchor, solve similarities onto match root Empties.
- Detect printed **AprilTag 25h9** markers in the active still and auto-create / update landmarks named `idNN-25h9` (OpenCV bundled as a wheel).
- Estimate Fitzgibbon one-parameter radial distortion from three or more concurrent segments.
- Press **Estimate Distortion** to fit λ once and show an expanded undistorted PNG (NumPy only—no OpenCV). Editing VP lines afterward keeps the stored λ; press again to re-fit.
- Apply display-only exposure/contrast to a sibling ``*-pm-view.png`` plate (undistorted uses the same lit image).
- Save match state in the `.blend`.

## Requirements

- Blender 5.1 or newer.
- A reference image with clear straight edges for the VP workflow.

The extension uses the NumPy bundled with Blender. **Find AprilTags** bundles `opencv-contrib-python-headless` as a platform wheel (extracted on install / enable). No manual `pip install` into Blender’s Python.

Wheels are **not** stored in git (~50–65 MB each). Before building or linking for development:

```sh
./fetch-wheels.sh
```

`./build-extension.sh` and `./link-dev.sh` call that for you. Release zips are built with `--split-platforms` so each OS package only embeds its own OpenCV binary.


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

That replaces `~/Library/Application Support/Blender/5.1/extensions/user_default/match_perspective` with a link to this repo (and downloads OpenCV wheels if missing). Enable **Perspective Match** once if it is not already on — **disable then re-enable** after the first link so Blender extracts the OpenCV wheel.

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

1. Click **New Match Camera** to create an empty match, activate it, and open the reference image file dialog.
2. Use **Active Match** to switch which session the sidebar edits. Switching also sets the scene camera and enters that camera view.
3. With the **Perspective Match** sidebar tab selected, **Ctrl+Alt+NumPad 1–9** jumps to the Nth match (name-sorted). Any Draw / Pick Origin / PP / Landmark tool is cancelled first.
4. Click the rename button (font icon) next to **New** to rename the active match after it was created. Opening a still still defaults the name from the image stem; rename anytime afterward. The hierarchy stays `PM_<name>` / `PM_<name>_Origin` / `PM_<name>_Camera`.
5. Click **Unload** to detach the sidebar from editing without deleting objects.
6. Click the trash button to **Delete** the active match (asks for confirmation). This removes its collection / Origin / Camera and that match’s landmark picks. You can still delete via the Outliner; the dropdown prunes dead entries automatically.

Selecting objects in the viewport does **not** change the active match—only New, Rename, Unload, Delete, the dropdown, and the NumPad shortcuts do.

Each match remembers its last camera-view zoom and pan. Switching matches (or Unload / save) stores the current framing; activating a match restores it.

### 1. Load a reference

With a match active, choose **Open Image** (or create a match with **New Match Camera**, which opens the same dialog). The still binds to the **active** match only; other matches stay untouched. If no match is active, Open Image creates one first. Hover the filename under Reference Image to see the full path.

After bind, the hierarchy is renamed from `PM_Match_###` to `PM_<image stem>` when that name is free. You can rename again anytime with the rename button in **Match Cameras**. The camera becomes the scene camera, the still is attached with **Stretch** frame mapping, and render dimensions match the image.

Once a still is loaded, **Replace Image** appears: same pixel size only, keeps VP lines, origin, calibration, and landmarks (drops view-lighting / undistorted caches). **Open Image** still does a full rebind and clears that match’s edit state.

### 2. Vanishing point lines

At the top of this section, choose perspective mode:

- **1 Point:** draw Z verticals and Y depth lines. FOV stays manual.
- **2 Point:** draw X and Y horizontal bundles (finite VPs). Z uprights are not used as a vanishing point.
- **3 Point:** draw any two axes with at least two lines each; the missing world axis is derived from orthogonality. All three axes also allow solving principal point. With **Manual FOV** or **Import YAML** (known full K), a single line on each of X/Y/Z is enough to recover orientation.

Axis mapping matches Blender’s gizmos:

- Red → Blender X
- Green → Blender Y
- Blue → Blender Z (up)

Then draw lines:

1. Choose the colored axis.
2. Click **Draw / Edit Lines**. Optional: enable **Snap to Edges** to refine each stroke on release onto a nearby image edge or thin dark/bright line (grout, painted strokes) along the whole segment — both endpoints move onto the fitted feature. Undo with Blender’s normal undo. Optional: enable **Show Error Label** to see each segment’s residual (px) beside the stroke.
3. Drag over straight edges that belong to that axis (clicks must land inside the camera frame).
4. Click a line to select it; drag either endpoint handle to edit (release also re-snaps when Snap to Edges is on).
5. Press **Delete/Backspace** to remove the selected line.
6. Press **Esc** (or right-click) to leave the tool.

While the tool is active, left-clicks on the visible plate belong to Perspective Match (object selection is blocked). Sidebar / toolbar / header clicks still go to Blender UI — including when **Region Overlap** draws those panels over the viewport. Mode feedback: a color-coded viewport banner + side/bottom frame, and a depressed tool button. Exit with **Esc** (or finish the pick / drag). Cursors also switch per tool (paint-cross for VP lines and origin, scroll for principal point, knife/dot for landmarks). In VP line mode the cursor refines on hover: hand-point over an unselected line, default over a selected segment, open hand over endpoint handles, and closed hand while dragging; empty plate stays paint-cross. Clicking **Draw / Edit Lines** or **Pick Origin** again refreshes the active tool instead of getting stuck. If you orbit out of camera view, the next line/origin click switches back to the match camera.

The camera refines whenever enough required lines exist. **Auto from VPs** allows orthogonal VP pairs to solve FOV. Enable **Manual FOV**, set the horizontal angle, and apply it to lock focal length while continuing to solve orientation. With known intrinsics locked that way (or via **Import YAML**), **3-point** mode can orient from one line per axis — pairs of parallels are still preferred when available, but not required. **Import YAML** loads a ROS `camera_info` file (same layout as OpenCV / `camera_calibration_parsers`): it locks Manual FOV from `fx`, sets the principal point from `cx`/`cy`, applies optional project-extension `fitzgibbon_lambda` as Division λ (and builds/shows the undistorted plate when λ ≠ 0, without re-fitting λ from VP lines), and scales K if the YAML resolution differs from the loaded still. Brown–Conrady / `plumb_bob` coefficients are still skipped.

**Manual PP Offset** (Camera section): drag the principal point on the plate, or click the pencil icon beside it to type **Offset X / Offset Y** in pixels from image center (same values as the PP offset readout; OK applies, Cancel discards). A **violet** crosshair marks it whenever it is off-center (and always while the drag tool is active). While dragging, orientation is rebuilt from VP lines on a short throttle (~12 Hz) so the mesh tracks the same “snap” you get on release; release does a final rebuild. **Esc** exits the drag tool like other tools (cancels an in-progress drag). **Reset Camera** recenters PP.

Middle mouse and the wheel retain normal Blender navigation while a match tool is active. Sidebar sections use Blender’s native collapsible panels (**View** starts collapsed). Overlay guides only draw in **camera view** of the active match — choosing a match from the dropdown (or **View Match Camera**) rehydrates the plate/lens after opening a `.blend` and enters that view.

### 3. Set origin

Use **Pick Origin** to choose a ground point that should become world origin (placement updates automatically after VP solves). Clear (X) removes the pick.

Build your own floor/wall geometry in Blender once the camera is matched—the extension no longer creates surface meshes or measures known lengths.

### 4. Sync matches

When several matches show the same scene, register them into one Blender world:

1. Match each still on its own (VP + origin). Origins do **not** need to match.
2. Choose an **Anchor** match — that world is shared space. Each match has **Enable sync for current match** (on by default) at the top of Sync Matches; turn it off to exclude that still from Solve Sync / Diagnose / Refine Lenses (the rest of the sync UI hides while that match is active).
3. Add landmarks for features visible in two or more stills (≥5 shared 2D picks), **or** link **Known 3D** Blender objects (≥3) and pick them in the other stills. Each landmark keeps a stable `item_id` plus a `creation_index` (add order). The **A–Z** toggle beside the list (below + / line / import / tracking / −) is a depressed/undepressed control: on = alphabetical by name, off = original add order. Sort only changes the list display — not storage order. The **font** toggle under A–Z shows each landmark's name next to its pick on the plate (off by default). The **Duplicate** button under that copies type / On Ground / Use in Sync (and sets Pick Confidence from existing picks) but clears Known 3D links, parallel links, picks, and solved positions. Uncheck **Use in Sync** (list checkbox or detail prop) to exclude a landmark from Solve Sync / Diagnose without deleting its picks — useful when sync starts failing after adding one point. With **Landmark Empties** on, that also removes its helper from `PM_Sync_Landmarks` (and restores it when re-enabled).

   **Find AprilTags** (AprilTag icon in the first button column): scans the active match still for **AprilTag 25h9** markers (same family as `tools/print-apriltags`). Each tag centre becomes a point pick on that match. Landmark names are `idNN-25h9` (NN = 00–99): if a landmark whose name **starts with** that prefix already exists, its pick for this match is updated; otherwise a new point landmark is created. OpenCV ships with the extension as a bundled wheel.
4. Pick each landmark in every still where it is visible (or re-run **Find AprilTags** per still). **Point** landmarks: click the feature. **Line** landmarks: drag a segment along the same edge (no need for a shared point). **Known 3D** Empties: auto-projected on the anchor; pick them in other matches only. For a metric edge, set **Known 3D** + **Known 3D B** on a Line landmark. Free lines (no Empties) need the edge in **≥3 stills** to constrain pose. Ordinary point landmarks must be picked in **both** stills when Known 3D sit on one line. Set **Pick Confidence** before clicking.
5. Optional: tag **On Ground** on point landmarks in the anchor, or rely on Known 3D points/lines, to pin absolute scale. Without metric cues, sync still recovers orientation and baseline *direction* with a depth heuristic.
6. **Solve Sync** writes a rigid transform (`R`, `t`, scale 1) onto non-anchor root Empties — or a similarity with free scale if a rigid pose cannot lock (different private-world metrics). Between Solve Sync and Refine Lenses: **Lock Rotation** keeps each Empty’s rotation at identity and only solves translation/scale (when private VP worlds already share axes); **Lock Translation** keeps Empty translation fixed and only solves rotation/scale; check **both** to leave cameras unmoved and only adjust 3D landmark / Empty positions to fit the picks. After the pairwise seed, a **joint bundle-adjustment** pass couples every free Empty pose with shared landmark positions (Cauchy-weighted point residuals + free-line midpoints / Known 3D line constraints) and soft-downweights severe landmark outliers. Use **Diagnose** first to see per-landmark RMSE without moving cameras — when error is high it also runs leave-one-out checks on the worst landmarks; **Clear** resets sync transforms. **Refine Lenses** searches each unlocked match’s focal length (re-orients from VP lines at each trial) with a **per-line VP residual prior** and hard VP guardrails, then a **coupled polish** jointly moves landmark-sharing pairs, then runs Solve Sync. It runs in a background thread so the UI stays responsive — watch the progress slider / status line, and press **Esc** or **Cancel** to stop (partial results are discarded). The % field beside it is the ± search window around current fx (default 18). Skips **1-point** matches and matches without enough VP lines (Manual FOV matches are included). The eye icon on **Sync Matches** toggles landmark picks on the plate (same pattern as VP Lines); **Landmark Empties** still controls the 3D helpers after sync.

   Refine Lenses batches landmark projection and ray math in NumPy, but its work still grows with the number of sync-enabled matches and focal trials. Disable unrelated matches or landmarks before refining a subset.

Why not “any corresponding points”? Photogrammetry / SfM does solve relative orientation and baseline *direction* from enough 2D↔2D matches — and so does this sync. What stays free is absolute baseline **length** when dropping the second camera into an already-metric Blender world from the first match (classic stereo scale ambiguity). **Known 3D** Empties, On Ground picks, or a later ruler pin that one DOF; without them, a depth heuristic chooses a plausible scale.

**Known 3D workflow (line / mesh verts):** Model or place Empties in the anchor world → select them → Sync list **Landmarks from Selected** (auto-fills 2D on the anchor still) → in each other match, **Pick** those features in 2D. Add a few off-line 2D↔2D landmarks if the known points lie on one edge (kills spin-around-the-line ambiguity).

**Line landmarks:** Add with the mesh icon next to +. Drag the same physical edge in each still — endpoints do **not** need to be the same 3D points, only the same infinite edge. Drag existing endpoints to edit (same as VP lines); click empty image to redraw. Optional: assign two Empties as **Known 3D** / **Known 3D B** so the edge is metric (works with two stills; ≥3 Known 3D lines can register pose alone). Mark **Is Parallel To** another Line landmark when two edges share a 3D direction: after pose is solved, free-line meshes in that family are **forced to share one direction** (Known 3D edges anchor the direction when present). Status reports `parallel Δ N° (direction locked)` when enforcement worked. Camera pose / reject uses **point** RMSE only — a Parallel lock that misses the 2D drawings shows as `parallel line miss (…px)` instead of rejecting sync. Without Known 3D ends, a free line needs **three or more** stills — two views alone cannot constrain relative pose from lines.

**What “px” means:** For **point** landmarks, RMSE is how far the projected 3D Empty lands from your 2D pick (image pixels). For **line** landmarks, it is how far each drawn **endpoint** sits from the projected infinite 3D line (perpendicular distance in pixels) — not whether the segment ends match in 3D. High line px after Parallel usually means the drawings disagree with the locked direction; the 3D edges can still look correctly parallel.

**Debugging a bad or rejected sync:**

- **Rejected (~40+ px)** means no pose fits your picks. Status / **Diagnose** lists the worst landmarks — re-pick those features in *both* stills (same physical point).
- **Accepted but camera looks wrong** with RMSE still a few–tens of px: wrong local minimum or soft constraints. Prefer **Diagnose**, fix the worst landmarks, **Clear**, then **Solve Sync** again. A “WARN high error” line means treat the pose as suspicious.
- **One landmark huge, others fine:** that pick is mismatched (wrong corner / wrong still). Uncheck **Use in Sync** on it and re-run Diagnose to confirm the rest solve.
- **Many landmarks all high:** FOV or VP solve is likely off on one match — try **Refine Lenses**, or re-refine that camera manually, then re-project Known 3D (**Landmarks from Selected**).
- **Sync broke after adding one landmark:** turn off **Use in Sync** on the new one (list checkbox) and Diagnose again. If it succeeds, re-pick that feature on both stills.
- **Known 3D warn (Empty vs anchor pick):** the Empty moved or the anchor camera changed after auto-project — re-run **Landmarks from Selected**.
- **Uncertain pick:** set that still's confidence to **Low** (or pick with Pick Confidence = Low). High-confidence picks dominate the pose; Low ones still help but the landmark Empty is allowed to miss them more.
- **One free line looks skewed vs another that should be parallel:** tag **Is Parallel To** — Solve Sync keeps the camera pose and re-fits that line mesh to the shared direction. Prefer linking the bad free line to a better-fitting free line or a Known 3D edge. 2D RMSE on the bad line may stay similar; the 3D edge orientation is what changes.

## Lens distortion

Press **Estimate Distortion** in the **Camera** section. With a successful VP solve and ≥3 concurrent segments on one axis, the solver fits Fitzgibbon λ once and generates/switches to an undistorted plate. Editing VP lines afterward keeps that λ; the undistorted plate is rebuilt only when intrinsics or λ change (orientation-only refines reuse the cache). Press the button again to re-fit. Works with **Manual FOV** (λ is estimated at the locked focal so you can compare VP-line RMSE with/without distortion). **Original Plate** clears λ, re-solves, and restores the source still. Plate file paths are logged to the console, not the sidebar status line.

## Limitations

- Initial FOV is manual. GeoCalib automatic FOV/gravity estimation is intentionally omitted.
- Vanishing lines are not detected automatically.
- The workflow assumes square pixels and zero skew.
- 1-point perspective cannot determine focal length from VP geometry alone.
- Distortion uses one radial division parameter, not Blender's full tracking-camera lens models.
- Cropped, anamorphic, curved, or CGI plates can require manual FOV and principal-point judgment.
- Sync solves relative pose from 2D↔2D and/or Known 3D Blender objects. Absolute baseline vs the metric anchor world needs Known 3D, On Ground, or the depth heuristic.
- Sync **Solve Sync** seeds pairwise pose then runs joint BA over Empty transforms + landmarks + free-line midpoints (Cauchy + line constraints) with block-analytic Jacobians; use **Refine Lenses** afterward to adjust unlocked focals against landmarks + a per-line VP prior with hard guardrails and a coupled multi-camera polish.
- **Diagnose** can leave-one-out the worst landmarks when sync error is high.
- View lighting bakes per match into ``<stem>-<camera>-pm-view.png``; matches that share one still do not share one plate. Reload the add-on after updating so activate restores a lost plate instead of the bright original.

## Project layout

```text
match_perspective/
  blender_manifest.toml  # Blender extension metadata
  __init__.py             # Registration
  properties.py           # Object sessions + scene workspace controller
  core.py                 # VP, focal, distortion, placement, and surface math
  sync.py                 # Multi-match landmark graph registration
  lens_refine.py          # Outer fx search (VP prior + sync RMSE)
  apriltag_detect.py      # AprilTag 25h9 → landmark picks (bundled OpenCV wheel)
  scene.py                # Camera/background integration and coordinate mapping
  overlay.py              # Camera-view GPU drawing
  operators.py            # File, solve, and modal interaction operators
  panel.py                # 3D View sidebar workflow
  icons.py                # Custom sidebar icons (bpy.utils.previews)
  icons/                  # PNG icon assets (vp-lines, april-tag 32/64, …)
  distortion.py           # NumPy image remapping
  wheels/                 # OpenCV wheels (gitignored; ./fetch-wheels.sh)
  fetch-wheels.sh         # Download platform wheels before build / link-dev
  validate_addon.py       # Headless Blender smoke test
  tests/test_core.py      # Pure geometry regressions
  tests/test_sync.py      # Landmark sync regressions
  tests/test_lens_refine.py  # VP residual + locked-focal helper
  tests/test_apriltag_detect.py  # AprilTag naming / matching helpers
  tools/print-apriltags/  # Generate A4/A3 AprilTag print sheets (standalone)
  tools/explore-vp-intrinsics/  # FOV×PP diagnostic plotter (standalone)
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

Validate and build distributable ZIPs (one per platform, each with its OpenCV wheel):

```sh
./build-extension.sh
```

The smoke test covers registration, multi-match create/switch/unload/prune, VP solve, camera projection, origin placement, project import, undistorted plates, and cleanup.

## License

GPL-3.0-or-later.
