# Perspective Match

Blender 5.1 extension for matching perspective cameras to stills. Keep several matches in one `.blend`, draw vanishing-point line bundles, solve each camera, pick a world origin, and optionally sync matches into one shared space with a landmark graph.

Native port of the manual workflow from Perspective Match Studio — no Electron, sidecar, or network. Core matching uses NumPy (bundled with Blender). **Find AprilTags**, **Snap to AprilTag**, and **Detect VP Lines** need OpenCV (`opencv-contrib-python-headless`); the extension still loads without it and hides those controls.

## Features

- Multiple match cameras per scene, each with its own still and calibration (locked `fx`, `fy`, `cx`, `cy`, and distortion copied onto new matches when the previous one used Manual FOV / YAML / 1-point; **Bulk Create** does the same for a folder)
- 1-, 2-, or 3-point perspective; draw or auto-detect (3-point) axis-colored VP lines
- Solve orientation and FOV from orthogonal VPs; **Use Known 3D** polishes FOV / principal point / camera position from landmark picks without leaving Auto from VPs; manual FOV / ROS `camera_info` YAML import (including OpenCV distortion coefficients)
- Adjusted Camera mode keeps live Blender or third-party pose/FOV edits when revisiting a match
- Principal point from three VPs or manual offset; undistort from imported OpenCV D or estimated Fitzgibbon λ
- Ground origin pick; multi-match sync via landmarks, Known 3D Empties (optional Known 3D Slack), Is Mirror Of pairs across one Mirror Empty, and AprilTag 25h9 / 36h10 (locked/imported K plus shared ground landmarks across 3+ images can replace VP lines)
- Diagnose produces self-contained local HTML diagnostics with camera connectivity, match status, ranked landmark errors, and permanent report export
- Match state saved in the `.blend`

## Requirements

- Blender 5.1 or newer
- A reference image with clear straight edges for the VP workflow

No manual `pip install` into Blender’s Python. Release zips and `./scripts/fetch-wheels.sh` (also called from build / link-dev) bundle OpenCV for auto-detect. Without that wheel, matching / drawing / sync still work; **Detect VP Lines**, **Find AprilTags**, and **Snap to AprilTag** stay hidden, and the Info editor logs that OpenCV is missing on enable.

## Installation

Download the zip for your OS from [Releases](https://github.com/soswow/blender-perspective-match/releases). In Blender: **Edit → Preferences → Get Extensions → Install from Disk**, then enable **Perspective Match**.

For live editing against this checkout, see [docs/development.md](docs/development.md).

## Usage

Sidebar: **3D View → Perspective Match**.

1. **New Match Camera** → load a still (or **Bulk Create** a folder of stills)
2. Draw (or detect) VP lines → camera solves when enough lines exist; alternatively, use locked/imported K and shared **On Ground** landmarks across 3+ stills
3. **Pick Origin** on the ground plane
4. Optionally sync several matches — see [docs/sync.md](docs/sync.md)

Full walkthrough: [docs/user-guide.md](docs/user-guide.md).

## License

[GPL-3.0-or-later](LICENSE) — see the `LICENSE` file for the full text.
