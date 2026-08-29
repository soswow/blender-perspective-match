# User guide

The extension lives in the 3D View sidebar under the **Perspective Match** tab.

## Match cameras

Each match is a collection with a root Empty that owns the session data, plus a child camera. **Hide Origin Empty** (Sync section, under Landmark Empties) is stored per match: it hides that Origin Empty in the viewport while the camera and collection stay visible. Hiding the Origin in the Outliner checks the box the next time you open that match. After **Solve Sync**, the **Enable sync for current match** row shows **Synced** or **Not synced** for the active match in that last run.

```text
PM_<image>
  PM_<image>_Origin      # session lives here (Object.pm_session)
    PM_<image>_Camera
```

1. Click **New Match Camera** to create an empty match, activate it, and open the reference image file dialog. If the previously active match has **Manual FOV** (including YAML import) or is in **1 Point** mode, its complete locked intrinsics (`fx`, `fy`, `cx`, `cy`, plus imported or estimated distortion) are copied onto the new match; K is remapped when you load a differently sized still. A portrait plate copied onto a landscape still of the same pixel count (3000×4000 ↔ 4000×3000) swaps axes so focal length stays the calibrated value; any other size change scales `fx` and `fy` together so square pixels are kept. **Bulk Create** (same row) picks a folder of stills and makes a match for each image that does not already have one, copying that same camera model onto every new match.
2. Use **Active Match** to switch which session the sidebar edits. Switching also sets the scene camera and enters that camera view.
3. With the **Perspective Match** sidebar tab selected, **Ctrl+Alt+NumPad 1–9** jumps to the Nth match (name-sorted). **Ctrl+Alt+←/↑** and **Ctrl+Alt+→/↓** step to the previous or next match (wraps). Any Draw / Pick Origin / PP / Landmark tool is cancelled first. Pressing the shortcut for the match that is already active keeps the current zoom and pan. In that match's **camera view**, **Ctrl+Cmd+A** (macOS; **Ctrl+Win+A** on Windows/Linux) starts **Pick in Active Match** for the selected landmark.
4. Click the rename button (font icon) next to **New** to rename the active match. Opening a still defaults the name from the image stem; rename anytime afterward. The hierarchy stays `PM_<name>` / `PM_<name>_Origin` / `PM_<name>_Camera`.
5. Click **Unload** to detach the sidebar from editing without deleting objects.
6. Click the trash button to **Delete** the active match (asks for confirmation). This removes its collection / Origin / Camera and that match’s landmark picks. You can still delete via the Outliner; the dropdown prunes dead entries automatically.

Selecting objects in the viewport does **not** change the active match—only New, Rename, Unload, Delete, the dropdown, and the match shortcuts do.

Each match remembers its last camera-view zoom and pan. Switching matches (or Unload / save) stores the current framing; activating a different match restores it. Re-activating the current match does not reset zoom/pan.

## Load a reference

With a match active, choose **Open Image** (or create a match with **New Match Camera**, which opens the same dialog). The still binds to the **active** match only. If no match is active, Open Image creates one first. Hover the filename under Reference Image to see the full path. **Bulk Create** loads every still in a folder (not subfolders), skips images that already have a match, and copies the active match’s complete locked camera model when Manual FOV / YAML / 1-point applies. A portrait locked K copied onto a landscape still of the same pixel count swaps axes (focal length unchanged); otherwise `fx` and `fy` share one scale so square pixels are kept. Normalized distortion coefficients are reused and each distorted still gets its own undistorted plate.

After bind, the hierarchy is renamed from `PM_Match_###` to `PM_<image stem>` when that name is free. The camera becomes the scene camera, the still is attached with **Stretch** frame mapping, and render dimensions match the image.

Once a still is loaded, **Replace Image** appears: same pixel size only, keeps VP lines, origin, calibration, and landmarks (drops view-lighting / undistorted caches). **Open Image** still does a full rebind (clears lines/origin); if **Manual FOV** is on, K is kept and scaled to the new plate size.

## Vanishing point lines

At the top of this section, choose perspective mode:

- **1 Point:** draw Z verticals and Y depth lines. FOV stays manual.
- **2 Point:** draw X and Y horizontal bundles (finite VPs). Z uprights are not used as a vanishing point.
- **3 Point:** draw any two axes with at least two lines each; the missing world axis is derived from orthogonality. All three axes also allow solving principal point. With **Manual FOV** or **Import YAML** (known full K), a single line on each of X/Y/Z is enough to recover orientation.

Axis mapping matches Blender’s gizmos:

- Red → Blender X (the axis button also shows its current line count)
- Green → Blender Y (the axis button also shows its current line count)
- Blue → Blender Z (up; the axis button also shows its current line count)

VP lines determine an axis but cannot distinguish its two infinity directions. **Flip X / Y Polarity**, below the axis buttons, reverses the red X and green Y directions together while keeping blue Z unchanged. Use it when the matched camera faces the right way vertically but Blender’s horizontal axes point the wrong way.

Then draw lines (or auto-detect in 3-point mode):

1. Choose the colored axis.
2. Click **Draw / Edit Lines**. Optional: enable **Snap to Edges** to refine each stroke on release onto a nearby image edge. Optional: enable **Show Error Label** to see each segment’s local direction error beside the stroke. The value is an endpoint-equivalent pixel miss over that stroke, so it remains comparable to pointing accuracy when a vanishing point is far off-screen or at infinity. Those numbers follow the Vanishing Point Lines eye toggle; hiding the VP guides also hides the labels without clearing the checkbox.
3. Drag over straight edges that belong to that axis (clicks must land inside the camera frame).
4. Click a line to select it; drag either endpoint handle to edit (release also re-snaps when Snap to Edges is on).
5. Press **Delete/Backspace** to remove the selected line.
6. Press **Esc** (or right-click) to leave the tool.

**Detect VP Lines** (3 Point only, needs OpenCV): runs in the background (Esc to cancel). Replaces the current strokes with automatically found bundles — well-spread inliers per axis that prefer well-separated vanishing points — then the usual camera refine when enough lines exist. Axis colors: **blue/Z** = uprights; **green/Y** = left-hand vanishing point; **red/X** = right-hand vanishing point. An X/Y vanishing point may land inside the still (near-2VP look); 3-point mode still solves it. **Edge Sensitivity** (0–1) controls how eagerly faint lines are kept. **Debug auto detected edges** toggles a black plate with every surviving edge in white. Needs clear man-made edges; cluttered foliage or curved architecture will miss or mis-label axes. Disabled in 1- and 2-point modes. Hidden entirely when OpenCV is not available.

While the tool is active, left-clicks on the visible plate belong to Perspective Match. Sidebar / toolbar / header clicks still go to Blender UI. Exit with **Esc**. Middle mouse and the wheel retain normal Blender navigation. Overlay guides only draw in **camera view** of the active match and while that 3D View has the **Perspective Match** sidebar tab selected with its main panel expanded. Closing the sidebar, collapsing the Perspective Match panel, or selecting another tab hides VP, origin, camera, landmark, label, and error overlays without changing their eye-toggle settings. The eye on **Vanishing Point Lines**, **Origin**, **Camera**, and **Sync Matches** hides that section's plate overlay (origin pick, principal-point marker, VP strokes, landmark picks). An active Draw / Pick tool keeps its required overlays visible until the tool exits. This does not hide the camera background image or solved landmark Empties / line helpers. **Hide Origin Empty** is separate: it hides the 3D Origin Empty, not the green origin mark on the still.

The camera refines whenever enough required lines exist. **Auto from VPs** allows orthogonal VP pairs to solve FOV. Enable **Manual FOV**, set the horizontal angle, and apply it to lock focal length while continuing to solve orientation. **Import YAML** loads a ROS `camera_info` file: locks Manual FOV from `fx`, sets the principal point from `cx`/`cy`, and remaps K if the YAML resolution differs from the loaded still (portrait ↔ landscape of the same pixel count swaps axes; other size changes scale `fx` and `fy` together). OpenCV `plumb_bob` / `rational_polynomial` coefficients (`k1, k2, p1, p2, k3[, k4, k5, k6]`) undistort the plate when present. Optional `fitzgibbon_lambda` is used only when those coefficients are zero. Fisheye / `equidistant` models are skipped. Distortion coefficients are not estimated from VP lines.

**Use Known 3D** (Camera, under Auto from VPs) folds landmark picks into that same solve instead of a one-shot button. Link **Known 3D** Empties on point landmarks and **Pick** those features on the photo — not **Landmarks from Selected**, which writes where the current camera thinks they are. Needs at least four well-spread picks with some depth. Name left/right from the same side as the Empties: a swapped pair (object projects onto the other pick) looks like a ~hundred-pixel miss and will skip the polish. When the checkbox is on, Auto from VPs, Estimate Distortion, Detect VP Lines, and later VP-line edits still solve from the lines, then polish FOV, principal point, and camera position so the Empties project onto the picks. Orientation is rebuilt from the VP strokes at that new K, so short verticals are not left pointing at a stale vanishing point. Distortion stays locked per trial (stored lens versus pinhole). If a VP-only λ would wreck an axis just to shave a little pin error, the pinhole wins. Manual FOV is not switched on. If the pin fit is rejected, the VP camera is kept and the status line says so.

At the top of **Camera**, **Camera Control** chooses who owns the camera. **Perspective Match** is the normal VP/origin workflow. Choose **Adjusted Camera** before continuing the camera in Blender or another add-on: the live camera transform and FOV become authoritative, including edits made after the mode was enabled. Switching matches, reopening **View Match Camera**, and rehydrating a saved match keep the current camera rather than restoring the old VP solve. Pose is read relative to the match Origin Empty, so **Solve Sync** may still move that Empty without changing the camera's private pose. Camera-solving, Origin, principal-point, distortion-estimation, and lens-refinement controls are disabled while this mode is active. VP lines remain editable as diagnostics. Return to **Perspective Match** only when you want its stored solve to control the camera again.

**Manual PP Offset** (Camera section): drag the principal point on the plate, or click the pencil icon to type **Offset X / Offset Y** in pixels from image center. A **light-blue** crosshair marks it whenever it is off-center. The Camera section eye hides that marker. **Reset Camera** recenters PP.

## Set origin

Use **Pick Origin** to choose a ground point that should become world origin (placement updates automatically after VP solves). The Origin section eye hides the green origin mark on the plate. Clear (X) removes the pick. **Solve Sync** / **Diagnose** / **Refine Lenses** also auto-set Origin from the first **On Ground** landmark pick when a match still has none. **Ground Slack** (next to Lock Rotation) is how far those On Ground tags may leave Z=0 — leave the default for a boarded floor; 0 for a machined plate. **Known 3D Slack** (beside it) is how far Known 3D points may leave their Empty during Solve Sync — 0 keeps them pinned; a small value lets CAD / mesh error ease toward the picks. A point that is both On Ground and Known 3D uses the tighter slack for Z. The linked Empty is not moved; Landmark Empties show the eased position. **Use Known 3D** in Camera still treats those Empties as fixed for the single-still polish. **Mirror Empty** (with **Plane** and **Mirror Slack**) is the shared midline for **Is Mirror Of** landmark pairs — the magic wand next to that dropdown fills the partner when the name ends with left/right. See [Sync matches](sync.md#mirror-pairs).

With complete locked/imported K on an anchor plus at least two supporting matches, four or more shared, well-spread **On Ground** landmarks can also initialize the camera orientations when the anchor has no VP lines. Five or six landmarks are recommended. Sync recovers Z from their common plane; X/Y yaw remains an arbitrary shared choice. See [Sync matches](sync.md#calibrated-ground-only-workflow-no-vp-lines).

Build your own floor/wall geometry in Blender once the camera is matched—the extension does not create surface meshes or measure known lengths.

## Lens distortion

**Import YAML** with a ROS `camera_info` file that includes OpenCV `plumb_bob` (or `rational_polynomial`) coefficients applies that model and generates/switches to an undistorted plate. **Estimate Distortion** is separate: with a successful VP solve and ≥3 concurrent segments on one axis, it fits Fitzgibbon λ once (replacing imported D) and shows an undistorted plate. Editing VP lines afterward keeps the current model; press the button again to re-fit λ. Works with **Manual FOV**. When **Use Known 3D** is on, the same Known 3D polish runs after λ is estimated and will keep λ = 0 if that VP-only λ makes an axis (especially short uprights) much worse. **Undistorted Plate** remaps the current still with that model. **Original Plate** shows the source still (imported D is kept; estimated λ is cleared and the camera is re-solved). Plate file paths are logged to the console, not the sidebar status line.

View lighting applies display-only exposure/contrast to `post-processed/<stem>-pm-view.png` next to the source still (undistorted and VP-detect debug plates go in the same folder). The folder is created if needed.

## Limitations

- Initial FOV is manual. GeoCalib automatic FOV/gravity estimation is intentionally omitted.
- Automatic VP lines are 3-point only; 1- and 2-point still need hand-drawn strokes. Detection can mis-label axes on ambiguous stills.
- Assumes square pixels and zero skew.
- 1-point perspective cannot determine focal length from VP geometry alone.
- Distortion: imported OpenCV Brown–Conrady from ROS YAML, or one Fitzgibbon radial parameter estimated from VP lines — not Blender's full tracking-camera lens models.
- Cropped, anamorphic, curved, or CGI plates can require manual FOV and principal-point judgment.
- Sync absolute baseline vs the metric anchor world needs Known 3D, On Ground, or the depth heuristic. See [sync.md](sync.md).
