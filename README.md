# Perspective Match

Perspective Match is a Blender 5.1 extension for matching perspective cameras to stills without leaving Blender. Keep several matches in one `.blend`, draw colored vanishing-point line bundles in camera view, solve each camera independently, and pick a world origin on the ground plane.

The extension is a native port of the manual workflow from Perspective Match Studio. It does not require Electron, Node.js, a Python sidecar, PyTorch, GeoCalib, OpenCV, or network access.

## Features

- Create multiple match cameras in one scene; each still keeps its own calibration session.
- Switch the active match from a dropdown to edit that session and view through its camera.
- Load a still directly as a Blender camera background on the active match only.
- Choose 1-, 2-, or 3-point perspective.
- Draw, select, edit, and delete axis-colored VP segments in camera view.
- Extend VP guides, horizon, and VP markers past the plate to finite vanishing points (capped when lines are nearly parallel).
- Derive a missing horizontal VP from an orthogonal pair so the horizon still draws with only one explicit horizontal bundle.
- Robust length-weighted VP fitting with Huber outlier reduction.
- Solve camera orientation and, when constrained, focal length from orthogonal VPs.
- Lock and edit horizontal FOV manually, including underconstrained 1-point shots.
- Solve an off-center principal point from three finite orthogonal VPs.
- Show per-plane FOV estimates and angular consistency diagnostics.
- Pick a ground origin so the matched camera sits relative to world origin.
- Estimate Fitzgibbon one-parameter radial distortion from three or more concurrent segments.
- Generate a transparent, expanded undistorted PNG using Blender's NumPy—no OpenCV required.
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

Middle mouse and the wheel retain normal Blender navigation while a match tool is active.

### 4. Set origin

Use **Pick Origin** to choose a ground point that should become world origin. **Apply Origin** recomputes camera position from the current solve; clear removes the pick.

Build your own floor/wall geometry in Blender once the camera is matched—the extension no longer creates surface meshes or measures known lengths.

## Lens distortion

Enable **Estimate Distortion** and refine after drawing at least three concurrent segments on one axis. The solver estimates the Fitzgibbon division parameter λ.

**Generate Undistorted Plate**:

1. Expands the output canvas when necessary so source content is not cropped.
2. Writes a transparent PNG beside the source image.
3. Switches the camera background and projection to the pinhole plate.
4. Warps overlay coordinates between stored source pixels and visible plate pixels.

Extreme or weakly constrained estimates that hit the λ search boundary are rejected and the pinhole model is retained.

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
- Display-only exposure and contrast controls from the desktop app are not reproduced; use Blender's image/color-management tools.
- Projects are import-only; there is no Save Project or camera JSON export.

## Project layout

```text
match_perspective/
  blender_manifest.toml  # Blender extension metadata
  __init__.py             # Registration
  properties.py           # Object sessions + scene workspace controller
  core.py                 # VP, focal, distortion, placement, and surface math
  scene.py                # Camera/background integration and coordinate mapping
  overlay.py              # Camera-view GPU drawing
  operators.py            # File, solve, and modal interaction operators
  panel.py                # 3D View sidebar workflow
  distortion.py           # NumPy image remapping
  project_io.py           # .pmproj import
  validate_addon.py       # Headless Blender smoke test
  tests/test_core.py      # Pure geometry regressions
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
