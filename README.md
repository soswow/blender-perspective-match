# Perspective Match

Perspective Match is a Blender 5.1 extension for matching perspective cameras to stills without leaving Blender. Keep several matches in one `.blend`, draw colored vanishing-point line bundles in camera view, solve each camera independently, place floor/wall reference surfaces, and set a world origin and known scale.

The extension is a native port of the manual workflow from Perspective Match Studio. It does not require Electron, Node.js, a Python sidecar, PyTorch, GeoCalib, OpenCV, or network access.

## Features

- Create multiple match cameras in one scene; each still keeps its own calibration session.
- Switch the active match from a dropdown to edit that session and view through its camera.
- Load a still directly as a Blender camera background on the active match only.
- Choose 1-, 2-, or 3-point perspective.
- Draw, select, edit, and delete axis-colored VP segments in camera view.
- Robust length-weighted VP fitting with Huber outlier reduction.
- Solve camera orientation and, when constrained, focal length from orthogonal VPs.
- Lock and edit horizontal FOV manually, including underconstrained 1-point shots.
- Solve an off-center principal point from three finite orthogonal VPs.
- Show per-plane FOV estimates and angular consistency diagnostics.
- Draw perspective-correct floor and wall surfaces with equal-world-spacing grids.
- Create live axis-aligned Blender meshes from those surfaces.
- Pick a ground origin and two points with a known real-world distance.
- Estimate Fitzgibbon one-parameter radial distortion from three or more concurrent segments.
- Generate a transparent, expanded undistorted PNG using Blender's NumPy—no OpenCV required.
- Save state in the `.blend` and import/export desktop-compatible `.pmproj` files.
- Export generic camera intrinsics and pose JSON.

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

### Development install

Build and reinstall the archive after source changes, or use **Refresh Local** in Blender's Extensions preferences.

## Workflow

The extension lives in the 3D View sidebar under the **Perspective Match** tab.

### Match cameras

Each match is a collection with a root Empty that owns the session data, plus a child camera (and any surfaces):

```text
PM_<image>
  PM_<image>_Origin      # session lives here (Object.pm_session)
    PM_<image>_Camera
    PM_<image>_Surface_*
```

1. Click **New Match Camera** to create an empty match and activate it.
2. Use **Active Match** to switch which session the sidebar edits. Switching also sets the scene camera and enters that camera view.
3. Click **Unload** to detach the sidebar from editing without deleting objects.
4. Delete a match by removing its Empty/camera in the Outliner; the dropdown prunes dead entries automatically.

Selecting objects in the viewport does **not** change the active match—only New, Unload, and the dropdown do.

### 1. Load a reference

With a match active, choose **Open Image** or open an existing `.pmproj`. The still binds to the **active** match only; other matches stay untouched. If no match is active, Open Image / Open Project creates one first.

After bind, the hierarchy is renamed from `PM_Match_###` to `PM_<image stem>` when that name is free. The camera becomes the scene camera, the still is attached with **Stretch** frame mapping, and render dimensions match the image.

### 2. Choose perspective

- **1 Point:** draw yellow depth lines and blue verticals. The shot does not determine FOV by itself, so FOV stays manual.
- **2 Point:** draw red and yellow line bundles. Their orthogonal VPs solve focal length and orientation.
- **3 Point:** draw yellow and blue bundles; red is optional but improves the fit. Three finite VPs can also solve principal point.

Axis mapping is fixed:

- Red → Blender X
- Yellow → Blender Y
- Blue → Blender Z (up)

### 3. Draw VP lines

1. Choose the colored axis.
2. Click **Draw / Edit Lines**.
3. Drag over straight edges that belong to that axis.
4. Click a line to select it; drag either endpoint handle to edit.
5. Press **Delete/Backspace** to remove the selected item.
6. Press **Esc** to leave the tool.

The camera refines whenever enough required lines exist. **Auto from VPs** allows orthogonal VP pairs to solve FOV. Enable **Manual FOV**, set the horizontal angle, and apply it to lock focal length while continuing to solve orientation.

Middle mouse and the wheel retain normal Blender camera-view navigation while a match tool is active.

### 4. Add surfaces, origin, and scale

Choose a surface plane:

- **Floor — XY:** red × yellow, mesh on Z=0.
- **Wall — YZ:** blue × yellow, mesh on X=0.
- **Wall — ZX:** blue × red, mesh on Y=0.

Run **Draw / Edit Surfaces** and drag opposite corners. Select a surface to adjust its overlay grid count. The grid is perspective-correct; the generated Blender mesh remains a four-corner rectangle.

Use **Pick Origin** to choose a ground point that should become world origin. Use **Measure Scale** to click two ground points, enter their known distance, and apply placement. Both scale points and measured length participate in every solve.

### 5. Save or export

Session data for each match is stored on its root Empty in the `.blend`. A `.pmproj` remains useful for exchanging editable matches with the standalone desktop app and loads into the active match.

- **Save Project** writes version-1 `.pmproj` for the active match.
- **Export Camera JSON** writes `K`, `R`, camera center, FOV, distortion, and image metadata.
- No Blender Python export is needed: the cameras and surfaces already exist in the current scene.

## Lens distortion

Enable **Estimate Distortion** and refine after drawing at least three concurrent segments on one axis. The solver estimates the Fitzgibbon division parameter λ.

**Generate Undistorted Plate**:

1. Expands the output canvas when necessary so source content is not cropped.
2. Writes a transparent PNG beside the source image.
3. Switches the camera background and projection to the pinhole plate.
4. Warps overlay coordinates between stored source pixels and visible plate pixels.

Extreme or weakly constrained estimates that hit the λ search boundary are rejected and the pinhole model is retained.

## Project compatibility

The extension reads and writes version-1 Perspective Match Studio projects:

```json
{
  "kind": "perspective-match-project",
  "version": 1,
  "savedAt": "...",
  "session": {}
}
```

The referenced image remains external. Relative image paths are resolved from the `.pmproj` directory.

## Limitations

- Initial FOV is manual. GeoCalib automatic FOV/gravity estimation is intentionally omitted.
- Vanishing lines are not detected automatically.
- The workflow assumes square pixels and zero skew.
- 1-point perspective cannot determine focal length from VP geometry alone.
- Distortion uses one radial division parameter, not Blender's full tracking-camera lens models.
- Surface reconstruction assumes Manhattan-world floor/wall planes.
- Cropped, anamorphic, curved, or CGI plates can require manual FOV and principal-point judgment.
- Display-only exposure and contrast controls from the desktop app are not reproduced; use Blender's image/color-management tools.

## Project layout

```text
match_perspective/
  blender_manifest.toml  # Blender extension metadata
  __init__.py             # Registration
  properties.py           # Object sessions + scene workspace controller
  core.py                 # VP, focal, distortion, placement, and surface math
  scene.py                # Camera/background/mesh integration and coordinate mapping
  overlay.py              # Camera-view GPU drawing
  operators.py            # File, solve, and modal interaction operators
  panel.py                # 3D View sidebar workflow
  distortion.py           # NumPy image remapping
  project_io.py           # .pmproj and camera JSON I/O
  validate_addon.py       # Headless Blender smoke test
  tests/test_core.py      # Pure geometry regressions
  build-extension.sh      # Validate and build installable zip
```

Python modules use snake_case because they are importable packages; user-facing files and generated assets use dash-separated names where practical.

## Development

Compile-check Python:

```sh
python3 -m compileall -q .
```

Run the Blender smoke test:

```sh
"/Applications/Blender 5.1.app/Contents/MacOS/blender" \
  --factory-startup -b --python validate_addon.py
```

Validate and build:

```sh
./build-extension.sh
```

The smoke test covers registration, multi-match create/switch/unload/prune, VP solve, camera projection, surface creation, origin/scale, project save/load, and cleanup.

## License

GPL-3.0-or-later.
