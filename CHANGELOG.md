# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Solve Sync keeps two-view camera baselines from collapsing while registering noisy shared point picks.

## [0.6.0] - 2026-09-12

### Fixed
- Solve Sync can build a consistent camera graph from only 2D point picks when a sparse anchor overlap disagrees with stronger view bridges.
- For undistorted stills, Solve Sync keeps visible line constraints active when their 3D midpoint slides far along the same edge.
- Free lines in a hard shared plane retain a compatible world-axis or Known 3D parallel direction.
- Solve Sync no longer crashes when a recovered camera triggers a geometry update in a scene containing free lines.
- Refine Lenses can recover from an initially rejected Sync by registering cameras afresh at new focal candidates while preserving explicit pose locks.
- Loading another file cancels Diagnose and Refine Lenses jobs; late callbacks from an old job cannot cancel or overwrite a newer one.
- Refine Lenses includes recovered cameras in its focal score and preserves supported cameras and geometry when selecting improvements.
- Mirrored lines retain compatible Is Parallel To directions from world axes or Known 3D edges, including when an independently supported hard plane also constrains them.
- Refine Lenses restores cameras, landmarks and cached plates after an application error, while retaining improved lenses after a numerical Sync refusal.
- Refine Lenses discards outdated results after input or camera edits, preserving the current calibration, poses and landmark diagnostics.
- Diagnose discards results when Sync inputs or the active scene change during its background job, preserving current landmark errors instead of publishing an outdated report.
- Background Refine Lenses now retains Is in Plane groups and Plane Slack, matching the blocking search and subsequent Sync solve.
- Hard shared planes now constrain mirrored line reconstruction and count toward line support when independently established by other geometry.
- Is in Plane no longer moves hard On Ground landmarks off the floor while initializing a Free plane; Ground Slack still permits later refinement.
- Solve Sync keeps previously solved geometry when a recovered camera's proposed 3D update would spoil the existing camera fits, while retaining soft Known 3D and plane constraints during refinement.
- Shift-selecting extra viewport objects no longer changes the Sync landmark or replaces the selection with that landmark’s Empty.
- Sync carries ground scale through overlapping cameras even when later ground landmarks are absent from the Anchor; Fit Only cameras can fit this ground without extending reconstruction.
- Fit Only cameras no longer seed or reshape mirrored landmarks, or suppress weak-line warnings with strokes excluded from reconstruction.
- Solve Sync keeps **Is Mirror Of** pairs even when those landmarks also share an **Is in Plane** bucket

### Added
- With zero Plane Slack, a supported Is in Plane bucket can place a point from one Solve or Lock Pose pick; diagnostics identify when the plane supplies its depth.
- Ctrl+Alt+Shift+Left/Right steps through the last 10 selected matches (back/forward; wraps).
- Solve Sync and Diagnose flag free and mirrored lines whose 3D position or direction is sensitive to small stroke edits, even when their pixel fit is good.
- Point and line landmarks can share a plane: **Is in Plane** chooses X, Y, Z, or Free and a #1–#10 bucket; **Plane Slack** is how far they may leave that plane

### Changed
- Clarified how to refine a shared lens from point matches without VP lines or Known 3D, including camera-motion and calibration limits.
- Solve Sync lists its geometric constraints without presenting unknown planes, free lines, or parallel directions as sources of absolute scale.
- Sync Matches status and this-match RMSE sit in a collapsed **Info** section under Hide Origin Empty.
- Duplicating a landmark named left/right or top/bottom (or ending in a space and a number) flips that side or increments the number when the new name is free; otherwise it still appends copy.
- Diagnose HTML match table columns are sortable, and the camera graph keeps the overlap tree in view with a green/gray key and hover details on shared-point counts.
- Sync Matches **This Camera** (Solve, Lock Pose, Fit Only) replaces the Lock Pose checkbox: Solve lets Sync move this camera and 3D; Lock Pose freezes the camera while its picks still move 3D; Fit Only only places the camera. Disabled on the Anchor.
- **Mirror Slack** sits beside Plane Slack, under Ground Slack / Known 3D Slack

## [0.5.0] - 2026-09-10

### Added
- When a Perspective Match camera is moved off its stored pose, landmark px errors hide and Camera offers Restore Stored or Capture Live
- Landmark list toggle to sort by last sync reprojection error (highest first)
- Diagnose opens a self-contained local HTML report with actionable issues, camera connectivity, match status, sortable landmark errors, and export controls
- Per-match **Lock Pose in Sync** keeps a trusted camera fixed while its landmark picks constrain the rest of the sync graph
- Per-landmark **Sync Weight** so a few important picks can pull harder during Solve Sync
- Sync Matches shows whether the current match was registered in the last Solve Sync
- Origin and Camera section eye toggles hide the origin pick and principal-point marker on the plate
- Diagnose now runs in the background with progress plus Esc / Cancel support
- The landmark list can be filtered to picks defined in the active match
- Vanishing Point Lines can flip X/Y infinity polarity per match, and each axis button shows its line count
- Adjusted Camera control keeps reading the live Blender camera so external pose and FOV edits survive match switching and View Match Camera
- Line landmarks can be constrained parallel to the shared-world X, Y, or Z axis from the Is Parallel To dropdown
- Line landmarks can use **Is Mirror Of** the same way as points (same-kind partner across the Mirror Empty)
- Hide Origin Empty (per match) hides that Origin Empty; Outliner hide/show stays in sync, camera and collection stay visible
- Ctrl+Cmd+A (Ctrl+Win+A on Windows/Linux) starts Pick in Active Match while the Perspective Match sidebar is open in that match's camera view
- Snap to AprilTag under Pick in Active Match recenters a point pick on a nearby dark tag-like quadrilateral (hidden when OpenCV is missing)
- Click a landmark pick on the plate to select it in the Sync list (Perspective Match sidebar tab open)
- Selecting a solved landmark Empty (or line helper) or its Known 3D object in the viewport selects that landmark in the Sync list when the Perspective Match sidebar tab is open
- Calibrated sync can infer the shared ground frame from four or more matching On Ground landmarks across three or more images, without requiring VP lines
- AprilTag sheets can embed subtle, detection-safe numeric labels in each tag's bottom-right border
- AprilTag sheets can export matching page-sized SVG cut outlines for vinyl cutters
- Find AprilTags now detects both 25h9 and 36h10 families with distinct family-qualified landmark names
- Diagnose / Solve Sync name a mismatched landmark pick when that still is skipped because one correspondence disagrees with the other views
- Refine Lenses **Same Lens** checkbox (on by default, left of the % window above the button) searches one shared focal scale for every still, including YAML-only matches with no VP lines
- **Undistorted Plate** (left of Original Plate) remaps the still with imported D / estimated λ and shows that pinhole plate
- **Ground Slack** lets On Ground landmarks sit a little off Z=0 so a boarded floor can flex without bending cameras
- **Known 3D Slack** lets Known 3D point landmarks ease a little off their Empty during Solve Sync so CAD error can follow the 2D picks without stretching cameras
- Point landmarks can be **Is Mirror Of** a partner across one scene **Mirror Empty**; **Mirror Slack** lets that plane ease along its normal if the Empty was slightly off; the magic wand next to Is Mirror Of fills the partner from a left/right name suffix
- **Use Known 3D** (Camera) polishes FOV, principal point, and camera position from landmark picks after Auto from VPs, Estimate Distortion, or VP-line edits, without locking Manual FOV; orientation is rebuilt from the VP strokes at the new K
- **Iterate Known 3D** repeats Auto from VPs (Use Known 3D) and Solve Sync until joint RMSE stops improving, so a moved match Empty can still pull FOV and camera

### Changed
- Filter to Current Match lists this still's overlay px error (and Sort by Error uses it) instead of the all-views RMSE
- Selecting a landmark in the Sync list selects its Known 3D object (or solved helper) in the viewport, including when that object is parented under another object
- Drawing a landmark line rubber-band uses the selected-pick red, not the current VP axis color
- Solve Sync uses a one-view **Is Mirror Of** line when placing a recovered camera, then pose-only BA of those stills
- **Is Mirror Of** is stored on both landmarks: selecting either side shows the partner, and **None** clears the pair
- Hiding Vanishing Point Lines also hides Show Error Label numbers on the plate
- Viewport overlays now show only while the Perspective Match sidebar panel is expanded, except while a Draw / Pick tool is active
- Refine Lenses skips Adjusted Camera stills instead of disabling the button for the whole graph
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
- Viewport overlay reuse of GPU batches so orbiting and drawing in camera view no longer leaks gigabytes of Metal vertex buffers
- Overlay hide-on-sidebar no longer redraws the 3D View ten times a second (that leaked tens of GB of GPU memory); **N** or another N-panel tab hides it, collapsing the Perspective Match accordion does not
- **Is Mirror Of** stores the partner by landmark id, so adding landmarks no longer retargets existing pairs
- Renaming a match re-sorts the Perspective Match and Anchor dropdowns by the new name
- Sync pose weights no longer inflate the pixel acceptance threshold and falsely report valid high-weight picks as mismatched
- Sync no longer prefers a cheap two-view pose when another candidate still fits the camera graph in pixels
- Sync registration failures now say that five shared 2D landmarks may connect through any registered match and clarify the usable 2D↔3D alternatives
- Undo and Redo no longer leave the match list empty after restoring dynamic dropdown state
- Pose-locked cameras now anchor free line 3D from their picks so stills being tuned (instead of the locked mesh cameras) move to the line instead of stretching a wrong 3D edge
- Solve Sync places a free 3D line from the best-conditioned views, so two locked near-duplicate stills cannot pin it at the wrong depth
- Solve Sync rebuilds free 3D lines after placing a recovered camera, and no longer refits those lines from locked near-duplicates only, so that still can pin line depth
- Recovered-camera pose polish uses a looser Huber so a dense cluster of well-fitting picks cannot ignore isolated landmarks that pin orientation
- Solve Sync keeps a still that already fits frozen 3D even when a line overlay is still off, and scores line error as stroke offset plus heading
- Orbiting with Sync Matches open no longer hitchs when the landmark list is large
- On Ground Known 3D points now use the tighter of Ground Slack and Known 3D Slack for Z, so a looser Known 3D leash cannot lift a floor pin
- Use Known 3D no longer leaves short uprights behind when FOV / principal point move, and no longer keeps a VP-only λ that wrecks an axis just to shave pin RMS
- Use Known 3D no longer recenters the principal point when λ is already 0, and names swapped/mismatched picks when VP lines block the fit
- Diagnose no longer spends minutes re-registering rejected cameras during leave-one-out and mismatched-pick checks
- VP error labels and lens refinement keep nearly parallel line bundles at local pixel scale instead of exploding for far-away vanishing points
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
