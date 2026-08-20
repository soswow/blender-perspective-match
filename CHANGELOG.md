# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Calibrated sync can infer the shared ground frame from four or more matching On Ground landmarks across three or more images, without requiring VP lines
- AprilTag sheets can embed subtle, detection-safe numeric labels in each tag's bottom-right border
- Find AprilTags now detects both 25h9 and 36h10 families with distinct family-qualified landmark names

### Changed
- AprilTag sheet printing now accepts only official OpenCV dictionary names and reports out-of-range marker IDs clearly
- Printed AprilTag labels contain only the marker number, without an `ID` prefix

### Fixed
- Sync can register upside-down or below-ground cameras through five or more landmarks shared with any already-solved view
- Sync resolves a bridged camera's ambiguous baseline scale from the whole landmark graph instead of an arbitrary two-view seed

## [0.4.0] - 2026-08-19

### Added
- Ctrl+Alt+Arrow keys cycle to the previous/next match (name-sorted, wraps)
- Bulk Create next to New Match Camera: one match per still in a folder, skipping images that already have a match and copying the active camera’s K
- Import YAML applies ROS `plumb_bob` / `rational_polynomial` distortion coefficients to undistort the still

### Changed
- Enable no longer waits on OpenCV; Detect VP Lines / Find AprilTags appear after a short probe
- Re-activating the current match (slot shortcut or cycle wrap) keeps live camera-view zoom/pan
- Selected landmark picks draw in red (still larger than other picks)
- View lighting, undistorted plates, and VP-detect debug images are written to a `post-processed` folder next to the source still
- Lock Rotation allows 90° world-axis jumps (including an X/Y swap) instead of forcing identity
- Reload Perspective Match only appears when the extension is a linked git checkout, not a zip install
- Rename Match focuses the name field with the current name selected, so typing replaces it
- AprilTag detection: increase sensitivity by 2x

### Fixed
- Bulk Create now copies the complete locked camera intrinsics and distortion model to every new match
- AprilTag landmarks now use the perspective-correct tag center instead of the average of its projected corners

## [0.3.7] - 2026-08-13

### Changed
- Installation: download the zip for your OS from GitHub Releases (Install from Disk)

### Fixed
- Disable then re-enable no longer fails with `already registered as a subclass 'PMLineSegment'`

## [0.3.6] - 2026-08-13

First public release for Blender 5.1+.

### Added
- Multiple match cameras per `.blend`, each with its own still and calibration
- 1-, 2-, and 3-point vanishing-point matching (draw, snap to edges, or auto-detect in 3-point)
- Solve orientation and FOV from orthogonal VPs; Manual FOV and ROS `camera_info` YAML import
- Principal point from three VPs or a manual offset; optional Fitzgibbon radial undistort
- Ground origin pick; match state saved in the `.blend`
- Multi-match sync via landmarks, Known 3D Empties, and AprilTag 25h9
- Optional OpenCV (`opencv-contrib-python-headless`): Detect VP Lines and Find AprilTags; core matching still loads without it
