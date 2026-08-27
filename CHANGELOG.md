# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Adjusted Camera control keeps reading the live Blender camera so external pose and FOV edits survive match switching and View Match Camera
- Line landmarks can be constrained parallel to the shared-world X, Y, or Z axis from the Is Parallel To dropdown
- Hide Origin Empty (per match) hides that Origin Empty; Outliner hide/show stays in sync, camera and collection stay visible
- Ctrl+Cmd+A (Ctrl+Win+A on Windows/Linux) starts Pick in Active Match while the Perspective Match sidebar is open in that match's camera view
- Snap to AprilTag under Pick in Active Match recenters a point pick on a nearby dark tag-like quadrilateral (hidden when OpenCV is missing)
- Click a landmark pick on the plate to select it in the Sync list (Perspective Match sidebar tab open)
- Selecting a solved landmark Empty (or line helper) in the viewport selects that landmark in the Sync list when the Perspective Match sidebar tab is open
- Calibrated sync can infer the shared ground frame from four or more matching On Ground landmarks across three or more images, without requiring VP lines
- AprilTag sheets can embed subtle, detection-safe numeric labels in each tag's bottom-right border
- AprilTag sheets can export matching page-sized SVG cut outlines for vinyl cutters
- Find AprilTags now detects both 25h9 and 36h10 families with distinct family-qualified landmark names
- Diagnose / Solve Sync name a mismatched landmark pick when that still is skipped because one correspondence disagrees with the other views
- Refine Lenses **Same Lens** checkbox (on by default, left of the % window above the button) searches one shared focal scale for every still, including YAML-only matches with no VP lines
- **Undistorted Plate** (left of Original Plate) remaps the still with imported D / estimated λ and shows that pinhole plate
- **Ground Slack** lets On Ground landmarks sit a little off Z=0 so a boarded floor can flex without bending cameras

### Changed
- Solve Sync balances landmarks across the frame so a cluster of well-fitting central picks cannot ignore a few near the edge that pin camera distance
- Pick Confidence and per-match confidence dropdowns sit in a collapsed section under Pick in Active Match
- On Ground landmark picks draw in magenta on the plate (selected stays red)
- Manual PP Offset marker is light blue (was violet)
- AprilTag landmark names zero-pad IDs to at least three digits (`id005-25h9`; four digits when the ID is ≥ 1000)
- AprilTag sheet printing now accepts only official OpenCV dictionary names and reports out-of-range marker IDs clearly
- Printed AprilTag labels contain only the marker number, without an `ID` prefix
- AprilTag sheet padding now wraps all four outer edges of the packed tag group
- Diagnose leave-one-out reuses the last locked poses instead of re-running pairwise registration for each worst landmark
- Diagnose and Solve Sync reuse pairwise poses when that still pair's picks and cameras have not changed (Clear Sync drops the cache)
- Diagnose and Solve Sync solve independent still pairs in parallel on the first run
- Solve Sync with many landmarks now thaws 3D after cameras settle so a bent triangulation can unbend
- Solve Sync grows the camera graph from the strongest still pair and the easiest next camera, instead of registering every well-overlapped still only against the Anchor in name order
- Solve Sync triangulation downweights near-parallel views, drops views that put a point behind the camera, and polishes 3D to the picks
- Original Plate keeps imported Brown–Conrady D and only switches the background; estimated λ is still cleared

### Fixed
- Manual PP Offset keeps the undistorted plate visible while dragging and rebuilds it after applying the new principal point
- Diagnose / Solve Sync no longer error when Is-Parallel-To lines are in the graph
- Diagnose no longer errors when leave-one-out has to re-solve without a previous per-landmark RMSE
- Snap to AprilTag recenters on the full inner black body of small blurry tags instead of a dark fragment
- Refine Lenses no longer drops the undistorted plate, which made a pinhole 3D view sit on the original barrel still and look like a horizon at infinity
- Solve Sync triangulates landmarks that only appear on recovered stills and poses hanging cameras from that 3D, so their Empties match the picks instead of keeping a stale 1px RMSE
- Solve Sync no longer shrinks a below-ground camera to a point, which made landmark overlays jump when panning
- Solve Sync still registers the cameras that fit when one still cannot, instead of failing the whole solve
- Solve Sync still registers a photo looking straight down at the ground from On Ground landmarks; if off-ground picks still disagree, that still is placed from the floor after the others lock
- Landmarks wrongly marked On Ground no longer block a pose that already fits the 2D picks
- Import YAML or copying a locked camera onto a landscape still whose size is the portrait calibration swapped (3000×4000 ↔ 4000×3000) keeps the calibrated focal length. Solve Sync still repairs matches that already have stretched fy
- Switching matches no longer errors when the Sync Anchor dropdown is out of date
- SVG cut shapes match the rasterized grid spacing of existing printed PDF sheets
- Seven-point sync accepts 180° camera yaw and scale-ambiguous short-baseline solutions instead of rejecting valid poses
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
