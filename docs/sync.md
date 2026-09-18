# Sync matches

When several matches show the same scene, register them into one Blender world.

## Overview

1. Match each still on its own (VP lines; Origin optional), or start with camera calibration and shared point picks using one of the no-VP workflows below. Origins do **not** need to match across stills.
2. Choose an **Anchor** match — that world is shared space. Each match has **Enable sync for current match** (on by default); turn it off to exclude that still from Solve Sync / Investigate Problems / Refine Lenses. **This Camera** chooses how a non-anchor match participates: **Solve** (default) lets Sync move the camera and 3D; **Lock Pose** keeps the current root transform (location, rotation, and scale) while its picks still constrain landmarks and the other cameras; **Fit Only** skips pairwise and only fits this camera against 3D from the other matches. The Anchor is already fixed, so the row is disabled there. After **Solve Sync**, the Enable row shows **Synced** or **Not synced** for the active match: whether this still was registered in the last run. A later run that skips it (or **Clear**) removes the check.
3. Add landmarks for features visible in two or more stills (≥5 shared 2D picks), **or** link **Known 3D** Blender objects (≥3) and pick them in the other stills. Optional: pair one-sided features with **Is Mirror Of** and one scene **Mirror Empty**.
4. Pick each landmark in every still where it is visible. With the **Perspective Match** sidebar tab open and the view through the active match camera, **Ctrl+Cmd+A** (macOS; **Ctrl+Win+A** on Windows/Linux) starts **Pick in Active Match**. Optional: enable **Snap to AprilTag** (under **Pick in Active Match**) so a point click on a small or blurry marker snaps to the tag centre — the intersection of the dark quadrilateral's diagonals — without needing the marker to decode.
5. Optional: tag **On Ground** on point landmarks in the anchor, or rely on Known 3D, to pin absolute scale.
6. **Solve Sync** writes a rigid (or similarity) transform onto non-anchor root Empties. Landmark px errors are vs each still's stored camera. If the Blender camera object was moved off that pose, the Camera section warns and hides those numbers until **Restore Stored** or **Capture Live**.

Use **Lock Pose** after a good solve when you want to add or tune landmarks without letting a trusted match drift. The lock uses the live root Empty transform at the start of each operation and applies to Solve Sync, Investigate Problems, and Refine Lenses. It is an exact freeze, not a warm start: **Solve** and **Fit Only** matches may start from their current solution, but their current root placement is not a pose constraint. Leave **Solve** selected when Sync should refine that camera. **Fit Only** skips pairwise and resects against the cloud built from the remaining matches. For **line** landmarks, two or more pose-locked picks on the same edge still pin helper length so a far still cannot stretch the mesh; the infinite 3D line itself is the best-conditioned intersection of the strokes (a locked near-duplicate view cannot pin it at the wrong depth). After Solve Sync places a recovered still, free lines are re-intersected so that still’s stroke can pin depth instead of keeping the locked-pair miss — only when This Camera is Solve or Lock Pose. A mirrored partner is refit from every posed stroke that may move 3D (near-duplicate locked views are dropped), so Is Mirror Of cannot put the helper back on the locked-only depth. Stills you are still tuning then move to that line. **Clear Sync** still resets every root transform explicitly.

Why not “any corresponding points”? Photogrammetry / SfM solves relative orientation and baseline *direction* from enough 2D↔2D matches — and so does this sync. Absolute baseline **length** stays free when dropping the second camera into an already-metric Blender world (classic stereo scale ambiguity), so pairwise pose validation does not use an absolute Blender-unit baseline cutoff. During each two-view pose fit, Sync holds one arbitrary baseline length fixed so noisy picks cannot make the ray-distance fit look better by shrinking the cameras together; this does not supply metric scale. **Known 3D** Empties, On Ground picks, or a later ruler pin that one DOF; without them, a depth heuristic chooses a plausible scale. The result’s **constraints** list describes the available geometric evidence. Unknown shared planes, free image lines and parallel directions can improve shape or pose while leaving absolute size undetermined. Constraint counts do not prove metric accuracy.

A match does not need five landmarks in common with the anchor itself. Sync can register it through any already-registered match with at least five well-spread shared point landmarks, then carry that pose into the anchor world. This also covers cameras on the opposite side of a surface — for example, a camera below the ground plane looking upward. The strong bridge chooses the orientation / hemisphere, while all registered views choose its otherwise-ambiguous baseline scale and refinement. If that cheap two-view pose disagrees with the rest of the graph, Sync keeps another candidate that still fits in pixels. Joint adjustment still uses sparse observations elsewhere and can downweight them as outliers.

Ground landmarks need not all appear in the Anchor. Once a camera is registered,
its **On Ground** picks can meet the shared Z=0 plane and supply scale to the next
overlapping camera. This requires **Solve** or **Lock Pose** participation. A
**Fit Only** camera can use ground reconstructed by those cameras to find its
pose, but its own picks cannot extend the 3D graph to another still. The ground
must represent the same physical plane throughout the set. These raycasts seed
the solve; Known 3D references keep precedence, and triangulation agreement and
Ground Slack still control how the ground positions are adjusted.

### Shared point matches with an approximate shared FOV (no VP lines)

For images with the same lens/zoom and consistent crops, shared point picks can
support camera placement and a common focal correction without VP lines,
On Ground, or Known 3D:

1. Create a match for each image. Start with at least three views taken from
   different positions, with overlapping features at different depths. Rotating
   the camera in place does not supply depth evidence.
2. Set **Manual FOV** to a reasonable horizontal-angle estimate. Use the same
   estimate for comparable images with the same lens and crop.
3. Choose an **Anchor**, enable Sync for the matches, and leave the other cameras
   in **Solve** mode. Add shared point landmarks spread across the images and
   object depth; leave them free. More than the minimum five shared picks gives
   the solve useful redundancy. An Origin is not required.
4. Run **Solve Sync**, then **Refine Lenses** with **Same Lens** enabled. The
   search range is a percentage of focal length, not degrees of FOV. A wider
   range can help when the initial estimate is poor, but does not guarantee a
   reliable solution.
5. Check alignment on object features you did **not** pick. A low landmark pixel
   error alone does not establish correct camera distance, focal length, or depth.

Same Lens applies one multiplier to the starting focal lengths; it preserves
their ratios rather than making arbitrary starting calibrations identical.
Different lenses, zooms, or crops need a different treatment. Without a metric
reference, the recovered world still has arbitrary scale. Points promoted to
Known 3D from this same reconstruction are working estimates, not independent
evidence that the geometry is correct.

### Independent FOV estimates from shared points (no VP lines)

For different lenses or zoom settings, turn **Same Lens** off and enable
**Estimate FOV from Landmarks** before running **Refine Lenses**. This opt-in mode
jointly fits each focal length, camera pose, reconstructed 3D point and supported line. Start with at
least three overlapping views taken from different positions and picks spread
across the images and object depth. Use Manual FOV for approximate starting
values; the **%** field bounds the focal length around those values (40% by
default in this mode, separate from the existing 18% lens search). It is not
a range in FOV degrees: 75% allows 0.25–1.75 times the starting focal length.
For a 50° starting FOV, that is approximately 123.6°–29.8°. A result at the
search boundary is not automatically applied; it does not establish where the
true lens lies. An improved, geometrically usable candidate can instead be
chosen explicitly with **Use Best Fit**, as described below.

The goal is to minimize the combined disagreement with your point picks, line
strokes and geometric constraints, using their weights and slack settings. The
solver does not know the true FOVs or a final error value in advance. Point RMSE
is one part of that score: satisfying a plane or mirror relation can slightly
increase point error while improving the combined fit. Startup registration
provides an initial estimate; its headline point error does not measure the
complete point, line and constraint objective.

“Converged” means nearby adjustments have become sufficiently unhelpful under
the numerical stopping checks. It is separate from image accuracy and does not
prove a global optimum or correct depth. Fitting currently has a 400-iteration
and 60-second limit after startup. A nonconvergence message identifies whether
the iteration limit was reached or no improving step was found. Running longer
is not guaranteed to produce a useful change, and reaching a lower point RMSE
is not itself a convergence condition.
The improvement tolerance is now `1e-7` instead of `1e-9`: for combined scores
above one, that means a successful step improves the score by less than
**0.00001%**. Below one, the same numerical tolerance is absolute. This stops
the search earlier when further progress is negligible; all depth, constraint,
noise and uncertainty checks still run before automatic application.

If initial Sync omits a camera despite enough picks, this mode can now try a
provisional pose against the reconstructed point cloud and then fit all cameras,
lenses and geometry together. The provisional cloud is not treated as Known 3D.
The final fit must still pass the same noise, geometry and uncertainty checks.
Plenty of picks supplies redundancy, but does not guarantee that the current
FOV, provisional 3D or correspondences agree.

A search-limit message names affected cameras and whether they need a wider
or narrower FOV. Try a wider **%** range or revise that camera's **Manual FOV**
starting estimate. A repeatedly saturated limit is not a measured calibration:
inconsistent picks, incorrect geometric constraints or unsupported image
calibration can also push fitting to a boundary. Do not reduce Assumed Pick
Error or delete picks just to force an accepted result.

A suggestion to check a camera pair is a tentative diagnostic, not the reason
for a focal-bound refusal. The check now suppresses that suggestion when a
model fitted to all shared picks agrees with the assumed noise: omitting one
isolated pick can make its prediction unreliable even when the correspondence
is correct. Some actual mismatches will also escape this conservative check.
Neither a warning nor its absence certifies the picks or reconstruction.

Throughout independent FOV estimation the sidebar shows activity, elapsed time
and completed fitting iterations. There is no completion percentage: registration
and numerical fitting have variable workloads. The other lens-search modes
retain their trial progress bar.

An off-center crop also moves the **principal point**, the pixel where the optical
axis meets the image. This mode keeps that point fixed; changing FOV alone cannot
generally compensate for a wrong crop offset. If the original principal point
was `(cx, cy)` and a crop removes `left` and `top` pixels, the cropped point is
`(cx - left, cy - top)`. Uniform resizing then multiplies both these coordinates
and focal length by the resize scale. The original frame or known crop rectangle
supplies this information without adding unknowns to the fit. **Manual PP Offset**
can enter an offset within the image; a crop's true principal point can also lie
outside it. Guessing the offset from the object's position is not a calibration.
Unknown crop offsets are still not estimated by this mode.

Check **Manual PP Offset** on each match before fitting, including offsets
inherited when creating matches. The numeric editor's `(0, 0)` means image
center; use that only when centered calibration is appropriate. Keep offsets
supported by calibration or crop metadata. Refine Lenses does not reset or fit
these offsets for you. You can run it directly after correcting PP or FOV:
it rebuilds its startup from the current inputs, so a separate Solve Sync is
not required and old landmark Empties are not treated as Known 3D.

**Assumed Pick Error (px)** is your assumed standard deviation of error in each picked
image coordinate. It is used to assess weak evidence and estimate local focal
ranges; it is not a measured accuracy score. Start conservatively when picks
are blurry. The default 1 px assumes roughly one pixel of typical error in
each X/Y coordinate at the original image resolution, independent of viewport
zoom. Larger values tolerate more noise but widen the estimated FOV ranges and
can make depth evidence insufficient. This is not a maximum allowed residual or
a control to force acceptance. A small fitted pixel error does not justify
reducing this setting.
It does not set the optimizer's convergence tolerance. A “did not converge”
message or focal search-bound message is separate from the pick-noise test.
The reported FOV ranges describe sensitivity near the fitted solution under
that assumption, not a guarantee of correct geometry or a search for every
possible solution. Inspect features you did not pick, and add translated views
or better-spread picks when the ranges remain broad.

Joint fitting keeps the principal point and supplied distortion coefficients
fixed. It preserves the focal aspect ratio when changing focal scale, including
imported nonsquare calibration. Known 3D points and lines, On Ground, camera
roles and pose locks retain their Sync meanings. VP-derived calibrations can
supply the starting camera orientation. These features are not discarded to
make a focal fit eligible.

Free self-calibration still needs sufficient independent information about
camera motion and object depth. Known geometry and pose constraints can supply
information that free image correspondences lack; acceptance checks the active
fit rather than assuming every point must be a free two-view landmark. A good
pixel fit alone does not establish focal certainty. The fitter does not estimate
distortion or unknown crop offsets. An accepted result applies its fitted
calibrations, cameras, points, and lines together; a refusal leaves the current
joint solution intact until an eligible best fit is explicitly chosen.
Without an external reference, scale remains arbitrary.

Free-focal fitting is bounded to 32 cameras and 80 reconstructed points, plus
the line limits below. Fixed-focal Solve can fit larger graphs in blocks, with
a memory and time limit. Resource refusals and excluded evidence are reported;
they do not silently remove constraints to obtain a successful fit.

**Use Best Fit** appears after a completed fit if its combined error improved and
it passed the physical geometry and per-camera deterioration checks, but
convergence, a focal bound, the noise model or local uncertainty prevented
automatic acceptance. It applies the fitted FOVs, camera poses, points and
lines together, without another Sync. The status retains the refusal reason
and labels the result **Provisional fit applied; calibration not validated**;
it does not present confidence intervals. Use Blender Undo to reverse the apply.
The status explicitly says when the option is available below Refine Lenses,
shows startup and fitted point RMSE, and notes if point RMSE increased while
the combined fit improved. The physical and per-camera deterioration checks
still apply; a lower combined score cannot bypass them.
Check alignment on features you did not pick: this option gives you a usable
modeling candidate, not evidence that its lens lengths or depth are correct.

If the fitting time limit is reached, an improved endpoint can still be offered
through Use Best Fit after the physical checks. It is not automatically accepted.
Cancelled jobs, fits that made no improvement, failed geometry checks and
unsupported/weak startup setups do not offer this option. The candidate is
temporary: it is cleared by a new lens job, file load or use. Changing numerical
inputs or cameras makes it stale; clicking then asks you to refit without
overwriting your edits. After using it, Refine Lenses can start a fresh search
around the new focal lengths; this is not a guarantee that another run improves
the result or resolves a repeatedly saturated bound.

Line strokes contribute alongside shared point picks and known geometry. The
fit uses distance to an infinite projected line: mark any clearly visible
portion of the same straight edge, in either drawing direction. Endpoints do
not need to identify the same physical locations across images. Assumed Pick
Error also applies to the perpendicular error of each stroke endpoint.
Ordinary free lines and line-to-line **Is Mirror Of** pairs are supported.
Free-focal fitting allows up to 24 lines and 96 strokes per fit; these are
resource limits, not requirements for a well-determined reconstruction.
A line needs two-view strokes, or a reconstructed mirror partner to supply its
geometry. Known 3D lines supply fixed geometry; mixed point/line mirror pairs are invalid.
Degenerate or inconsistent line evidence can
cause a refusal even when the point-only fit would pass.

Line **Is in Plane** and **Is Parallel To** also participate in this joint fit:

- A line plane needs independently located members in the same group: one
  member for **X/Y/Z #**, or three non-collinear members for **Free #**. Fitted
  points and fixed Known 3D line references can provide this support. Free
  lines cannot establish their own supporting plane. A plane supported by
  fitted points follows them during fitting; they do not become Known 3D.
- A plane constrains both the line's position and direction. **Plane Slack**
  softens position, while the line must still run along the plane. Plane Slack
  is not an angular tolerance.
- **Is Parallel To** may target another included line or **X/Y/Z Axis**. Opposite
  drawing directions are equivalent, and the lines may be at different positions.
  A parallel relation alone does not provide missing line depth or replace the
  two-view-stroke/mirror-partner startup requirement.
- These relations can be combined with line mirrors. They affect the fit and
  conditional focal uncertainty, but do not count as additional independent
  image picks. Conflicting relations can make the joint fit refuse.

Use relations that describe the physical object, not edges that merely look
parallel in one image. Valid constraints can rule out distorted reconstructions;
they cannot guarantee recovery from a wrong principal point or lens model.
Use Free planes and line-to-line parallelism when world orientation is unknown;
X/Y/Z planes and axes assert that the object is already aligned to those world
directions in the anchor frame.
A Free plane expresses coplanarity without asserting world alignment. Use it
when that matches the intended evidence. Changing focal length alone cannot
generally repair a mismatch between a guessed starting frame and a known world
direction; the joint fit can adjust the observable orientation freedoms below.

Point landmarks may use **Is in Plane** (X/Y/Z or Free) and **Is Mirror Of**
with a supplied **Mirror Empty** or an on-plane **Mirror Landmark**. Plane Slack and Mirror Slack keep their
existing meanings: plane membership can be softened, and Mirror Slack lets
the effective mirror plane slide along its normal relative to the selected object or live landmark.
A one-view point can participate when current Known 3D, ground, or a supported
hard plane determines its depth. Fit Only views cannot supply that depth.
Point-only Free groups need four members to constrain coplanarity; point-only
axis groups need two. Point-supported line planes use the requirements above.
Plane and mirror relations are enforced in the joint fit after preliminary
camera registration from the image picks. This helps weak camera arrangements
whose guessed starting FOV previously left the constrained fit stuck.

The fit can rotate the cameras and reconstructed landmarks together to satisfy
X/Y/Z planes, a supplied mirror normal, or world-axis line parallelism. This
lets those relations orient a reconstruction whose starting anchor orientation
was only a guess. The anchor camera center stays fixed. Rotation freedoms that
the relations do not measure retain a starting-frame convention; two points in
one axis bucket constrain less rotation than a fully supported plane. Free
coplanarity and line-to-line parallelism alone keep the original anchor orientation.
The fitted anchor orientation is stored in its camera calibration, so the next
Sync uses the corrected frame. The common final fit uses these orientation
freedoms with either fixed or free focals. Explicit camera locks and world
references can remove those freedoms.

The joint fit allows up to 400 iterations within a 60-second time
limit; preliminary camera registration has separate work limits. Weak setups
can need more iterations when orientation is fitted too.

The reported uncertainty includes fitted orientation freedoms but remains
conditional on the supplied constraints and fixed camera center. It does not
check that those references are true.
A Mirror Empty can fix scale relative to the stored anchor placement, but
that establishes real dimensions only if that placement is trustworthy.
An incorrect mirror offset can change reconstructed scale without worsening
image alignment. A landmark can supply the plane position, but its orientation must still be
supplied. Inferring that orientation from mirror pairs remains a separate
workflow question.

Passing the eligibility and uncertainty checks is not an accuracy guarantee.
Frozen tests include successful exact and noisy 12- and 16-landmark sets with
partial overlap, but noisy focal errors can still approach 10% while lying
inside the reported intervals. More points on the same weak part of the object
do not necessarily determine its depth better. Check the rest of the object.
When a fit fails to converge or disagrees with the stated pick error, it may
suggest a pair of views whose shared picks deserve review. This optional hint
needs at least 12 shared picks in that pair; it does not identify a particular
wrong landmark, prove a mismatch, or change whether a result is accepted.

### Calibrated ground-only workflow (no VP lines)

When the anchor has no usable VP solve, cold **Solve Sync** and **Refine Lenses** can initialize its ground frame directly from calibrated views. A usable applied solution retains its fitted frame:

1. Give the anchor and at least two supporting matches (three images total) a complete locked K — Manual FOV, imported camera-info YAML, or 1-point mode. Imported `fx`, `fy`, `cx`, `cy`, distortion, and plate dimensions are all used.
2. Mark at least four well-spread, non-collinear point landmarks **On Ground** and pick the same landmarks in each image. Five or six are recommended so a bad pick can be rejected.
3. Use images taken from different positions. A pure camera rotation has no plane-normal cue, and two images alone have a genuine two-solution ground-plane ambiguity.
4. Press **Solve Sync**. Sync infers the anchor Z/vertical and compatible orientations for the other unsolved matches, auto-picks each missing Origin from its first ground landmark, then runs the normal landmark solve.

The ground plane determines Z but has no preferred compass direction, so anchor X/Y yaw is chosen deterministically from the previous camera orientation. All inferred matches share that choice. The usual scale ambiguity still applies; without Known 3D or another metric constraint, the initial camera height is conventional rather than measured.

## Landmarks

Each landmark keeps a stable `item_id` plus a `creation_index` (add order). UI helpers:

- **A–Z** toggle: alphabetical by name vs original add order (display only).
- **Filter** toggle: show only landmarks with a pick defined in the active match, including From Points lines whose two endpoints both have picks there. List px and **Sort by Error** then use this still's overlay miss instead of the all-views RMSE. The selected landmark always shows both numbers when this still can be scored.
- **Font** toggle: show landmark names next to picks on the plate.
- Click a pick on the plate to select it in the list (while the **Perspective Match** sidebar tab is open). The selected pick draws in red. **On Ground** picks draw in magenta; Known 3D picks in cyan.
- **Duplicate**: copies type / On Ground / Is in Plane / Use in Sync / Sync Weight, clears Known 3D links, parallel links, mirror links, picks, and solved positions. A From Points duplicate keeps its source mode but clears Point A/B for fresh selection. The new name flips a trailing left/right or top/bottom, or increments a trailing ` 3`-style number, when that name is free; otherwise it appends `copy`.
- **Use in Sync**: exclude a landmark from solving and investigation without deleting picks. With **Landmark Empties** on, that also removes its helper from `PM_Sync_Landmarks`.
- **Sync Weight**: how strongly this landmark pulls Solve Sync (default 1). Raise it on a couple of well-placed picks that sit far from the others so a cluster of easier landmarks cannot ignore them. Combines with per-still **Pick Confidence** (High ×4, Low ×0.25). Boosted landmarks also skip the usual “this pick looks like an outlier” downweight. Weight influences pose refinement and candidate ranking; camera acceptance and mismatched-pick diagnostics remain in raw image pixels.

### Find AprilTags

Scans the active match still for **AprilTag 25h9 and 36h10** markers. Each tag's perspective-correct physical centre (the corner-diagonal intersection, corrected for active lens distortion) becomes a point pick. Landmark names include the family, such as `id005-25h9` and `id005-36h10`, so the same ID in both dictionaries creates distinct landmarks. Tag IDs are zero-padded to at least three digits (four for 36h10 IDs ≥ 1000). If a landmark whose name **starts with** the matching family-qualified name already exists (including older two-digit names such as `id05-25h9`), its pick for this match is updated and the name is rewritten to the current padding; otherwise a new point landmark is created. Needs OpenCV (`opencv-contrib-python-headless`); the button is hidden when neither dictionary is available.

When a printed marker is too small or blurry to decode, pick it by hand with **Snap to AprilTag** enabled. The click does not identify the ID; it only recenters onto the dark blotch (the inner black quad, not the white quiet zone). Needs OpenCV; the checkbox is hidden when the wheel is missing.

### Known 3D workflow

Model or place Empties in the anchor world → select them → Sync list **Landmarks from Selected** (auto-fills 2D on the anchor still) → in each other match, **Pick** those features in 2D. Add a few off-line 2D↔2D landmarks if the known points lie on one edge (kills spin-around-the-line ambiguity). The same Known 3D links plus picks on **this** still also feed **Use Known 3D** in the Camera section — pick on the photo, not the auto-projected positions, when you want that single-camera polish. If the CAD is a strong guide but not exact, raise **Known 3D Slack** so Solve Sync can ease those points a little toward the picks.

### Mirror pairs

A point or line landmark can name another of the same kind as **Is Mirror Of** — the same feature on the opposite side of a symmetric object. The pair is stored on both landmarks; selecting either side shows the other, and clearing **None** on one side clears the other. Each side is picked only where it is visible. The magic-wand button next to the dropdown fills the partner when this landmark's name ends with **left** or **right** and another landmark of the same kind uses the swapped name. Pairwise registration still needs ordinary shared points. The mirror constraint is used in joint BA, and a line picked in only a recovered still is mixed against the partner's reflected 3D when Solve Sync places that camera.

One scene **Mirror Empty** (below the slack rows) is the plane for every pair. Place it on the midline. **Local Plane** chooses which local face is the mirror (YZ by default: local X is the normal). **Mirror Slack** sits beside **Plane Slack** (0 pins the plane) and lets that plane slide along its normal if the Empty was slightly off. The Empty is not moved.

For a plane that should follow a reconstructed point, set **Mirror Position**
to **Landmark**, then choose the **On-plane Point**. It must lie on the symmetry
plane, but need not be the object's center. Keep it separate from the paired
left/right landmarks. **World Plane** chooses YZ, XZ or XY when no orientation
object is assigned. For another direction, choose an **Orientation (optional)**
object and its **Local Plane**; only that object's orientation is used.

The reference is an ordinary reconstructed point. Its current estimate and the
mirror relations are fitted together, so better picks and additional views can
change the plane position during later solves. Its cached position and any
visible landmark Empty are not treated as exact Known 3D. **Mirror Slack = 0**
keeps the plane through the fitted reference; positive slack allows a normal
offset from it. Moving an orientation object does not move the plane; rotating
it changes the supplied normal. The solve never moves that object.

The reference needs picks in at least two cameras allowed to contribute 3D,
or an explicit Known 3D reference. The same reference identity and support
rules apply to joint focal fitting. Missing,
excluded, line or paired references are refused rather than replaced with a
static plane. Rename/reorder keeps the selection by landmark identity; after
deleting it, choose a replacement. Investigation keeps the reference in place during
leave-one-out checks because removing it would change the mirror model.

A free reconstructed landmark supplies position, not orientation or measured scale. The normal
must be meaningful in the anchor frame, and weak picks can still produce weak
geometry. Check features outside your fitted picks. Existing files keep
**Mirror Position = Object** and their previous behavior.

When the Anchor has no picked Origin or On Ground landmarks, a world **YZ** or
**XZ** mirror plane with **Mirror Position = Landmark** also places the 3D
Anchor Origin below the completed reconstruction. The chosen point sits directly
above world zero, and every reconstructed point and line endpoint is above
Z=0. This placement runs after a successful Solve Sync or accepted Refine Lenses
fit; it leaves image projections unchanged. It needs a free world position, so
it does not run with Known 3D objects, a locked camera pose, Lock Translation,
or an orientation object. The 2D **Pick Origin** marker stays unset because that
control represents a picked ground point. Clear Sync removes reconstructed
landmarks and Sync transforms; the Anchor camera keeps its last placement until
you recalibrate it.

In Object mode, if Is Mirror Of is set but Mirror Empty is empty, Solve Sync
ignores those pairs and says so in the status line.

### Shared planes

A point or line landmark can join an **Is in Plane** bucket: **X**, **Y**, or **Z** (share that world coordinate with others in the same **#1–#10** bucket) or **Free** (lie on the same unknown plane). The group menu shows assigned landmark counts, such as **#1 (7)** or **#4 (empty)**. Counts include point and line landmarks of the selected plane type, including those disabled for Sync; **X #1** and **Z #1** have separate counts. Two landmarks tagged **Z #2** share some height that is not necessarily Z=0; **Z #1** is a different height. **On Ground** remains the Z=0 floor — putting On Ground points in a Z bucket pulls the rest of that bucket toward the floor. A Free plane can also meet that floor: initialization preserves On Ground seeds, and nonzero Ground Slack controls whether they can move during refinement. **Free** only constrains once four or more members are reconstructed (three points always define a plane). Lines are pulled so their midpoint lies in the plane; a zero **Plane Slack** also keeps the stroke direction in that plane. The same landmark can also be **Is Mirror Of** a partner; Solve Sync applies both.

With **Plane Slack = 0**, a point picked in one **Solve** or **Lock Pose** camera can also be placed using its plane bucket. X/Y/Z need at least one other reconstructed member to establish the shared coordinate; Free needs at least three other non-collinear members to establish the plane. The camera must already be placed. A grazing ray, an intersection behind the camera, insufficient support, or a Fit Only pick does not supply a new point this way. This route currently handles points and hard planes only.

The report labels these points **Plane + one view**. Their depth follows your plane constraint; a small pick error does not independently verify it. A pick in another camera allowed to contribute 3D adds a separate geometric check. Removing the plane or making the only contributing camera Fit Only removes the point unless other evidence determines it.

A hard plane established independently by other reconstructed geometry also helps fit existing free lines, including mirrored pairs. The line is fitted within the plane using its strokes. A compatible world-axis or Known 3D **Is Parallel To** direction remains fixed while fitting a free line inside that plane; incompatible directions cannot be enforced together. Compatible planes on mirrored partners preserve both plane membership and reflection. Free lines and points whose depth comes from that plane alone do not count as independent plane support for this check. This does not add reconstruction of an ordinary line with only one stroke.

The common final fit keeps the whole infinite line parallel to its hard plane.
Changing which portion is displayed cannot introduce a new plane violation.

Mirrored lines also retain a compatible **Is Parallel To** direction supplied by a world axis or a **Known 3D** edge. When a supported hard plane applies, reconstruction keeps that direction, plane membership and reflection together. A fixed direction alone does not determine the line's position: nearly coincident reflected strokes can still produce a **Weak 3D line support** warning. Nonzero Plane Slack does not turn the plane into a hard depth reference.

### Line landmarks

Add with the mesh icon next to +. With **Source → Drawn**, drag the same physical edge in each still — endpoints do **not** need to be the same 3D points, only the same infinite edge. Optional: assign two Empties as **Known 3D** / **Known 3D B** so the edge is metric. **Is Parallel To** can constrain the edge to shared-world **X Axis**, **Y Axis**, or **Z Axis**, or to another Line landmark that shares its 3D direction. **Is Mirror Of** pairs a drawn line with its counterpart across the scene Mirror Empty, the same as for points. **Is in Plane** can keep a line in a wall, table, or other shared plane with point landmarks.

With **Source → From Points**, select two existing point landmarks as **Point A**
and **Point B**. Their solved positions define the infinite line; the displayed
segment ends exactly at those points. No line strokes are needed. On the plate,
the segment connects the endpoint picks using the usual line colors and a dashed
stroke. It has no line handles and cannot be drawn or dragged; edit the endpoint
point picks instead.

**Is Parallel To** and **Is in Plane** on a From Points line influence its
endpoints during Solve Sync and Refine Lenses. For example, parallel to **X Axis**
requires the points to share Y and Z, while leaving their X separation free.
A line's plane membership acts alongside each endpoint's own plane membership,
Ground and Known 3D settings. It does not overwrite them: endpoints can belong
to different plane buckets while the line has another plane relation. Zero
slack keeps the corresponding constraint hard; positive slack allows its usual
movement. Incompatible hard constraints cannot produce an accepted fit.

Both endpoints must be included point landmarks with enough evidence to locate
them. Missing references, repeated endpoints and coincident 3D positions cannot
define a usable line. A derived line adds no image picks or independent plane
support; its endpoints retain their own pick weights and errors. A line plane
still needs independently located support in its bucket: one member for X/Y/Z,
or three non-collinear members for Free. Parallel targets may be world axes,
drawn lines or other From Points lines. From Points lines do not support
**Is Mirror Of** or separate Known 3D line endpoints.

Without Known 3D ends, a **drawn** free line needs **three or more** stills — two views alone cannot constrain relative pose from lines. Ordinary point landmarks must be picked in **both** stills when Known 3D sit on one line. Expand **Pick Confidence** (collapsed by default, under **Pick in Active Match**) to set the next-pick default or per-still confidence; it multiplies the landmark **Sync Weight**.

**What “px” means:** For **point** landmarks, RMSE is how far the projected 3D Empty lands from your 2D pick. For **drawn line** landmarks, each drawn endpoint’s perpendicular distance to the projected infinite 3D line is measured. Those two distances combine offset (the stroke sitting beside the projected edge) and heading (angle miss scaled by half the stroke length): RMS = hypot(midpoint offset, ½ length × sin(angle)). A short stroke therefore reports a parallel miss more than a heading miss; a long stroke also punishes a twist. Pose accept still uses **point** RMSE, so a line that is not yet sitting on the overlay cannot skip a still that already fits the 3D cloud. After that still is placed, recovered-camera polish still uses the line (and spatially isolated picks) to rotate it — a dense cluster of well-fitting picks cannot Huber-ignore isolated landmarks that pin orientation.

For an undistorted still, sliding the 3D line helper along the same infinite edge does not change that edge's pixel error. A line that extends into the space in front of the camera can still be fitted when its helper midpoint is behind the camera. A line entirely behind the camera and parallel to the image plane has no visible projection.

## Solve Sync and related tools

After placing a recovered **Solve** camera, Sync may refine the shared 3D again.
That update retains Ground, Known 3D, Mirror and **Is in Plane** constraints,
including their slack. Soft Known 3D points can continue moving toward the picks.
If the proposed update worsens a previously solved camera beyond the existing
fit allowance, Sync retains the previous geometry and reports **kept existing
geometry after camera recovery**. The recovered camera remains placed; inspect
its own pick errors if its evidence conflicts with the rest. This check protects
existing point fits; it does not establish that every camera or 3D feature is
accurate.

**Solve Sync** uses a usable current solution as its starting point. With unchanged captured evidence, it retains that solution if a proposed replacement loses support, fails geometric checks, or worsens the common point/line/constraint objective. Changed evidence can use the old poses as a starting guess, but is evaluated against the new picks and constraints.

For a cold or incomplete start, Sync seeds pairwise pose then runs a joint bundle-adjustment over Empty transforms + landmarks (Huber-weighted, with extra influence on poorly covered regions of each still so a cluster of central picks cannot ignore a few near the edge that pin camera distance). Raise **Sync Weight** on a landmark when those automatic boosts are not enough. Pairwise growth starts from the geometrically strongest still pair (spread, overlap, parallax, pair RMSE) and adds the easiest next camera — never alphabetical names, and not every still vs the Anchor just because it shares five picks. With only free 2D points and no pose locks, it connects the pair's better-spread member to the Anchor first, then compares each later direct-Anchor pose with bridges through registered views so a few matching pixels cannot pull the graph apart. The Anchor remains the shared world. 3D landmarks are triangulated from all registered rays, with near-parallel views downweighted, behind-camera views dropped, and a short reprojection polish. Options between Solve Sync and Refine Lenses:

- **Lock Rotation** — keep each Empty’s rotation on a 90° world-axis jump (identity, ±90°, 180° about X/Y/Z, including an X/Y swap); only solve translation/scale. Use when VP axes already match across stills so a free solve would only add a few degrees of noise.
- **Lock Translation** — keep Empty translation fixed; only solve rotation/scale.
- Both checked — leave cameras unmoved; only adjust 3D landmark / Empty positions.
- **Ground Slack** — how far On Ground landmarks may sit off Z=0 (scene units). 0 pins them to the floor raycast. The default (0.02) is enough for plank cup / tag thickness on a boarded floor.
- **Known 3D Slack** — how far Known 3D point landmarks may sit off their Empty (scene units). 0 (the default) pins them. A small value is a spring toward the Empty while each still's 2D pick pulls the point along that camera's ray, so CAD that is slightly wrong can share the error with the cameras instead of stretching the overlay. Linked Empties stay put; **Landmark Empties** show the eased positions. Known 3D lines stay pinned. **Use Known 3D** (Camera) still treats the Empty as fixed. A point that is both Known 3D and **On Ground** uses the tighter of the two slacks for Z.
- **Plane Slack** — how far **Is in Plane** landmarks may leave their shared plane (scene units). 0 (the default) pins them. A small value lets a slightly warped wall or table flex.
- **Mirror Empty / Plane / Mirror Slack** — one object whose chosen local face is the shared mirror for every **Is Mirror Of** pair. Slack 0 pins the plane to the Empty; a small value lets it slide along the normal. The Empty is not moved. Mirror Slack sits beside Plane Slack.

**Solve Sync** and **Refine Lenses** save a self-contained local HTML report
when an operation finishes. They do not open the browser automatically.
**Open Last Report** opens the latest report without running another solve;
**Export** saves a permanent copy. Reports contain no remote resources and are
not uploaded.

The report identifies the operation and whether its result was applied. Point
and line errors are separate, with per-camera details, camera and landmark
coverage, constraint warnings, an interactive camera-overlap graph, and a
searchable landmark table. Error numbers on the visible landmarks belong to
the applied result. A failed fit or a diagnostic trial does not replace those
numbers with errors from different cameras or geometry. After input edits, the
saved report is marked outdated; it still describes the captured result.

**Investigate Problems** is an optional diagnostic operation. It uses the same
final fitter, then can temporarily omit individual landmarks to assess their
influence. Each comparison scores the same surviving evidence before and after
the trial, with separate point, line and combined scores. The report lists any
relations removed with a feature; improvement does not prove the feature is
wrong. These checks are limited to five candidates and 60 seconds, and report
incomplete work explicitly. The trial is not applied. This operation is not
needed to obtain the report from the last Solve or Refine operation.

Solve, Refine, and investigation run in the background with activity and elapsed
time in the sidebar. **Esc** or **Cancel** stops the active operation. Changed
picks, constraints, camera roles, or other numerical inputs invalidate a
pending result; unrelated modeling edits do not. Loading another file retires
the old jobs so late callbacks cannot overwrite the new scene. **Clear** resets
Sync transforms and forgets the current report link. Scripted execution remains
available without modal window-manager plumbing.

**Refine Lenses** searches focal length to lower reprojection error across supported point picks. The **Same Lens** checkbox and **%** field sit above the button. **Same Lens** (on by default) applies one scale to every still — use this when they share a physical camera / imported YAML; it does not need VP lines. Off normally uses a per-still VP search (re-orients from VP lines, skips 1-point / weak-VP stills). Coupled polish and Solve Sync follow, retaining the same plane groups, Plane Slack and other geometric constraints as Solve Sync. Alternatively, enable **Estimate FOV from Landmarks** for the independent no-VP workflow above; that mode applies the common joint camera, point and line fit directly, with the supported constraints and resource limits above. Both run in a background thread — watch the progress slider, press **Esc** or **Cancel** to stop. The % field is the ± search window around current fx (default 18 for the existing searches, 40 for point FOV estimation). Disable unrelated matches or landmarks before refining a subset. Matches in **Adjusted Camera** mode are skipped so the button stays available for the others.

In **Estimate FOV from Landmarks** mode, **Refine Distortion** adds a conservative one-parameter radial-distortion pass after an accepted FOV fit. It keeps the fitted focal lengths, camera poses and 3D geometry fixed. Each still needs at least 16 supported point picks spread from the image centre toward the edges. A radial-stratified validation subset must confirm a meaningful improvement, and the complete point/line/constraint score must also improve; otherwise that still keeps its existing distortion unchanged. Imported Brown–Conrady coefficients are preserved and skipped. The reported FOV intervals remain conditional focal-only intervals from the preceding fit. The option is off by default and is unavailable to the Same Lens / VP searches.

For the existing shared/VP searches, the lens score includes recovered stills whose errors may be excluded from Solve Sync’s joint headline. Once a successful candidate exists, later candidates must keep its registered cameras and reconstructed points/lines and remain successful. An initially refused solve can still improve its lenses, even if Sync continues to refuse. Line-only searches retain their existing line score. Point FOV estimation automatically applies only an accepted joint result; **Use Best Fit** explicitly applies an eligible provisional candidate with its warning.

If the initial Sync is rejected, Refine Lenses registers cameras afresh for new focal candidates until it obtains a successful solve. It then reuses the solved poses to speed subsequent trials. Explicit **Lock Pose** settings apply throughout; a rejected starting focal length does not prevent recovery when a suitable candidate lies within the search window.

Refine Lenses also checks its inputs and target cameras before applying a background result. Changing picks, constraints, origin or VP evidence, search settings, or camera transforms while it runs discards that result and preserves your edits. Switching the active match or editing an unrelated object does not invalidate it. If applying a valid result encounters an internal error, it restores the previous cameras, landmark state and cached plates and reports the error. A numerical Sync refusal still retains the refined lenses and reports the refusal.

**Iterate Known 3D** (beside Refine Lenses) repeats **Auto from VPs** with **Use Known 3D** then **Solve Sync** until joint RMSE stops falling. That is the hand-click loop that still moves FOV and camera after a match Empty has been registered: Known 3D is expressed in the private frame, so a new root transform changes the pins. Eligible stills need Use Known 3D, four Known 3D picks, and enough VP lines. Pose-locked non-anchor matches and Adjusted Camera stills are skipped. Esc / Cancel keeps the last improvement (at most eight rounds).

The eye icon on **Sync Matches** toggles landmark picks on the plate (same pattern as Vanishing Point Lines, Origin, and Camera); **Landmark Empties** controls the 3D helpers after sync. **Hide Origin Empty** is per match and hides that Origin Empty in the viewport (camera and collection stay visible); it stays in sync with the Origin's Outliner visibility. With the **Perspective Match** sidebar tab open, click a pick on the plate — or select that landmark’s Empty / line helper or Known 3D object in the viewport — to select it in the list (red overlay). Shift-selecting additional objects leaves the Sync landmark unchanged so the viewport multi-select is kept. Selecting a landmark in the list selects its Known 3D object or solved helper in the viewport (including when that object is parented). Inactive **On Ground** picks are magenta; Known 3D are cyan. **Pick in Active Match** can still place or move the active landmark; clicking a different pick selects it instead of overwriting. Dragging a new line rubber-band uses the same red as the selected pick, not the VP axis colors. **Snap to AprilTag** (point landmarks) looks around the click for a dark four-sided blotch with a brighter border and moves the pick to that quad's diagonal intersection. Per-match pick coordinates, confidence, and last-sync RMSE are under the collapsed **Pick Confidence** header.

## Debugging a bad or rejected sync

- **Rejected (~40+ px)** — no pose fits your picks. Status / **Open Last Report** lists the worst landmarks — re-pick those features in *both* stills.
- **Plenty of picks, still rejected** — On Ground is load-bearing. Only landmarks that actually lie on the ground plane should be On Ground. A still that cannot lock is skipped so the others can still sync. After the remaining cameras lock, that still is retried as PnP against their triangulated 3D (floor tags alone if off-plane picks disagree). The report and Solve Sync name the skipped match, and if one pick disagrees with the other stills they name that landmark (uncheck **Use in Sync** or re-pick it). A photo looking straight down at the ground is registered from those On Ground picks (plane homography); generic 2D↔2D pose is a poor fit there even when the tags are correct. If a portrait locked K was copied onto a landscape still of the same pixel count, Import YAML / copy keep the calibrated focal length (axes swap). Cold registration repairs a stored fx/fy stretch above 20%; continuation keeps the accepted calibration.
- **Accepted but camera looks wrong** with RMSE still a few–tens of px — wrong local minimum or soft constraints. Open the last report and review the largest residuals. Use **Investigate Problems** for additional trials; correct the evidence and solve again. **Clear** forces a fresh registration if the current geometry is unusable. If a few picks far from the main cluster look sacrificed while the rest sit perfectly, raise those landmarks' **Sync Weight** (try 4–8) and solve again. Matches without Origin but with On Ground picks get an auto Origin on Sync; if tilt persists, add an elevated (off-ground) landmark or a 4th ground pick.
- **Flipped the object for underside photos** — On Ground is one shared plane. Table tags from the flipped session must not be On Ground if the original table tags already pin Z=0. Side tags glued to the object *are* the same 3D points and do connect the graphs. After a physical flip, a camera that photographed the underside is placed *below* the original ground looking up (object frame), not above the table in room coordinates.
- **List shows ~1px but the Empty is far from the pick** — the px number is RMSE across the stills that participated in the last solve, not the currently viewed still. Landmarks that only appear on recovered / hanging stills used to keep a stale number; Solve Sync now triangulates those tags and poses the hanging still from them. If the Empty is still off, switch to that still and compare this match's residual under **Pick Confidence**.
- **One landmark huge, others fine** — check whether the same ID identifies a different physical point in one still. Review its picks and use **Investigate Problems** to assess its influence. A large residual can also reflect conflicting constraints or calibration; it does not prove that one click is wrong.
- **Many landmarks all high** — FOV or VP solve is likely off on one match; try **Iterate Known 3D** (Use Known 3D on that still) or **Refine Lenses**, or re-refine that camera manually.
- **Sync broke after adding one landmark** — inspect the new feature in **Open Last Report**, or use **Investigate Problems** to compare a trial without it.
- **Known 3D warn (Empty vs anchor pick)** — the Empty moved or the anchor camera changed; re-run **Landmarks from Selected**.
- **Landmarks jump on the plate when panning one still** — that match Empty was shrunk to a point (typical for a below-ground camera). Switch to the match or re-run **Solve Sync**; the camera should stay below-ground, but overlay picks stay on the photo.
- **A free line should follow a world axis** — set **Is Parallel To** to X Axis, Y Axis, or Z Axis. For two arbitrary parallel edges, link the bad free line to a better-fitting free line or a Known 3D edge.
- **Weak 3D line support** — the strokes and geometric constraints supply nearly coincident supporting planes, so small pick errors can move or rotate the reconstructed line substantially despite a good pixel fit. Solve Sync names these lines; the report marks their rows and reports the best supporting-plane angle. Mirror pairs include reflected views in this check. An independently established hard shared plane counts when the reconstructed line actually lies in it. Try longer strokes or a stroke from a more distinct viewing angle, using a camera in **Solve** or **Lock Pose**. Fit Only strokes do not supply reconstruction support or clear this warning. Known 3D lines and their reflected partners get geometry directly and are exempt. This warning preserves the solved geometry; it is a sensitivity indicator, not a confidence interval or proof that unflagged lines are correct.
