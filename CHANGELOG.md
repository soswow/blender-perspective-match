# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Import YAML applies ROS `plumb_bob` / `rational_polynomial` distortion coefficients to undistort the still
- Colorspace dropdown under Reference Image, bound to the loaded still (items come from the active OCIO config).

### Changed
- Lock Rotation allows 90° world-axis jumps (including an X/Y swap) instead of forcing identity
- Reload Perspective Match only appears when the extension is a linked git checkout, not a zip install
- Rename Match focuses the name field with the current name selected, so typing replaces it
- AprilTag detection: increase sensitivity by 2x

### Fixed
- HDR / EXR stills (e.g. ACEScg): Detect VP Lines and Find AprilTags no longer hard-clip highlights at 1.0 when reading Blender's pixel buffer (the fallback used when OpenCV cannot read the file, e.g. EXR). Float plates are percentile-normalized and gamma-encoded before quantization, so highlight edges survive detection.

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
