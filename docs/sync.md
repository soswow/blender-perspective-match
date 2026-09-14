# Sync matches

When several matches show the same scene, register them into one Blender world.

## Overview

1. Match each still on its own (VP lines; Origin optional), or start with camera calibration and shared point picks using one of the no-VP workflows below. Origins do **not** need to match across stills.
2. Choose an **Anchor** match — that world is shared space. Each match has **Enable sync for current match** (on by default); turn it off to exclude that still from Solve Sync / Diagnose / Refine Lenses. **This Camera** chooses how a non-anchor match participates: **Solve** (default) lets Sync move the camera and 3D; **Lock Pose** keeps the current root transform (location, rotation, and scale) while its picks still constrain landmarks and the other cameras; **Fit Only** skips pairwise and only fits this camera against 3D from the other matches. The Anchor is already fixed, so the row is disabled there. After **Solve Sync**, the Enable row shows **Synced** or **Not synced** for the active match: whether this still was registered in the last run. A later run that skips it (or **Clear**) removes the check.
3. Add landmarks for features visible in two or more stills (≥5 shared 2D picks), **or** link **Known 3D** Blender objects (≥3) and pick them in the other stills. Optional: pair one-sided features with **Is Mirror Of** and one scene **Mirror Empty**.
4. Pick each landmark in every still where it is visible. With the **Perspective Match** sidebar tab open and the view through the active match camera, **Ctrl+Cmd+A** (macOS; **Ctrl+Win+A** on Windows/Linux) starts **Pick in Active Match**. Optional: enable **Snap to AprilTag** (under **Pick in Active Match**) so a point click on a small or blurry marker snaps to the tag centre — the intersection of the dark quadrilateral's diagonals — without needing the marker to decode.
5. Optional: tag **On Ground** on point landmarks in the anchor, or rely on Known 3D, to pin absolute scale.
6. **Solve Sync** writes a rigid (or similarity) transform onto non-anchor root Empties. Landmark px errors are vs each still's stored camera. If the Blender camera object was moved off that pose, the Camera section warns and hides those numbers until **Restore Stored** or **Capture Live**.

Use **Lock Pose** after a good solve when you want to add or tune landmarks without letting a trusted match drift. The lock uses the live root Empty transform at the start of each operation and applies to Solve Sync, Diagnose, and the sync solves inside Refine Lenses. It is an exact freeze, not a warm start: **Solve** and **Fit Only** matches are solved from their correspondences, without treating the current root placement as a pose prior. Leave **Solve** selected when Sync should refine that camera. **Fit Only** skips pairwise and resects against the cloud built from the remaining matches. For **line** landmarks, two or more pose-locked picks on the same edge still pin helper length so a far still cannot stretch the mesh; the infinite 3D line itself is the best-conditioned intersection of the strokes (a locked near-duplicate view cannot pin it at the wrong depth). After Solve Sync places a recovered still, free lines are re-intersected so that still’s stroke can pin depth instead of keeping the locked-pair miss — only when This Camera is Solve or Lock Pose. A mirrored partner is refit from every posed stroke that may move 3D (near-duplicate locked views are dropped), so Is Mirror Of cannot put the helper back on the locked-only depth. Stills you are still tuning then move to that line. **Clear Sync** still resets every root transform explicitly.

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
uses points alone, so its smaller point error can accompany worse constraints.

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

This mode supports 3–32 cameras and 8–80 point landmarks, with at
least eight picks per camera and at least two views per point. It requires
square-pixel pinhole calibration, zero distortion and a fixed principal point.
The camera ceiling is a resource guard for the current dense numerical fit,
not a mathematical maximum. All participating views are fitted jointly; no
overlapping batches are necessary within this limit. Larger sets may take
longer to register, and the subsequent focal fit retains its time limit.
It refuses unsupported constraints instead of ignoring them: Known 3D, On
Ground, VP strokes, pose locks and Fit Only belong to the existing Sync/lens
workflows. It does
not estimate distortion or unknown crop offsets. If the shared picks can be
explained by planar geometry or rotation without reliable depth evidence, the
mode declines to change the cameras. A successful result applies the jointly
fitted cameras, points and lines together; a refusal initially leaves the existing scene intact.
Without an external reference, scale remains arbitrary.

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

Line strokes supplement the shared point picks; they do not replace the eight
point picks required per camera for startup and the depth-evidence check. The
fit uses distance to an infinite projected line: mark any clearly visible
portion of the same straight edge, in either drawing direction. Endpoints do
not need to identify the same physical locations across images. Assumed Pick
Error also applies to the perpendicular error of each stroke endpoint.
Ordinary free lines and line-to-line **Is Mirror Of** pairs are supported.
This first implementation allows up to 24 lines and 96 strokes per fit.
A line needs two-view strokes, or a reconstructed mirror partner to supply its
geometry. Known 3D lines and mixed point/line mirror pairs remain unsupported.
Degenerate or inconsistent line evidence can
cause a refusal even when the point-only fit would pass.

Line **Is in Plane** and **Is Parallel To** also participate in this joint fit:

- Put a line and at least one picked point in the same **X/Y/Z #** group to
  establish the shared coordinate. For **Free #**, include at least three
  non-collinear picked points in that group. Each supporting point still needs
  two-view picks. The supporting plane follows those points during fitting;
  they do not become Known 3D. Line-only plane groups are not supported here yet.
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
The independent focal fit currently holds the anchor camera's orientation
fixed. If that orientation is only an initial guess, a valid equal-height
relation on the object may disagree with Blender's world Z. Changing focal
length alone cannot generally repair that frame mismatch. A Free plane
expresses coplanarity without asserting world alignment; use it only when
that matches the intended evidence, not to discard a known physical direction.

Point landmarks may use **Is in Plane** (X/Y/Z or Free) and **Is Mirror Of**
with a supplied **Mirror Empty** or an on-plane **Mirror Landmark**. Plane Slack and Mirror Slack keep their
existing meanings: plane membership can be softened, and Mirror Slack lets
the effective mirror plane slide along its normal relative to the selected object or live landmark.
Each member still needs picks in at least two cameras; the one-view constrained
reconstruction available in ordinary Sync is not part of this FOV mode.
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
Sync uses the corrected frame. This applies to **Estimate FOV from Landmarks**;
ordinary Sync and VP-based refinement retain their existing anchor rules.

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

The eight-pick minimum is an eligibility rule, not an accuracy guarantee.
Frozen tests include successful exact and noisy 12- and 16-landmark sets with
partial overlap, but noisy focal errors can still approach 10% while lying
inside the reported intervals. More points on the same weak part of the object
do not necessarily determine its depth better. Check the rest of the object.
When a fit fails to converge or disagrees with the stated pick error, it may
suggest a pair of views whose shared picks deserve review. This optional hint
needs at least 12 shared picks in that pair; it does not identify a particular
wrong landmark, prove a mismatch, or change whether a result is accepted.

### Calibrated ground-only workflow (no VP lines)

When the anchor has no usable VP solve, **Solve Sync** and **Diagnose** can initialize its ground frame directly from calibrated views:

1. Give the anchor and at least two supporting matches (three images total) a complete locked K — Manual FOV, imported camera-info YAML, or 1-point mode. Imported `fx`, `fy`, `cx`, `cy`, distortion, and plate dimensions are all used.
2. Mark at least four well-spread, non-collinear point landmarks **On Ground** and pick the same landmarks in each image. Five or six are recommended so a bad pick can be rejected.
3. Use images taken from different positions. A pure camera rotation has no plane-normal cue, and two images alone have a genuine two-solution ground-plane ambiguity.
4. Press **Solve Sync**. Sync infers the anchor Z/vertical and compatible orientations for the other unsolved matches, auto-picks each missing Origin from its first ground landmark, then runs the normal landmark solve.

The ground plane determines Z but has no preferred compass direction, so anchor X/Y yaw is chosen deterministically from the previous camera orientation. All inferred matches share that choice. The usual scale ambiguity still applies; without Known 3D or another metric constraint, the initial camera height is conventional rather than measured.

## Landmarks

Each landmark keeps a stable `item_id` plus a `creation_index` (add order). UI helpers:

- **A–Z** toggle: alphabetical by name vs original add order (display only).
- **Filter** toggle: show only landmarks with a pick defined in the active match. List px and **Sort by Error** then use this still's overlay miss instead of the all-views RMSE. The selected landmark always shows both numbers when this still can be scored.
- **Font** toggle: show landmark names next to picks on the plate.
- Click a pick on the plate to select it in the list (while the **Perspective Match** sidebar tab is open). The selected pick draws in red. **On Ground** picks draw in magenta; Known 3D picks in cyan.
- **Duplicate**: copies type / On Ground / Is in Plane / Use in Sync / Sync Weight, clears Known 3D links, parallel links, mirror links, picks, and solved positions. The new name flips a trailing left/right or top/bottom, or increments a trailing ` 3`-style number, when that name is free; otherwise it appends `copy`.
- **Use in Sync**: exclude a landmark from Solve Sync / Diagnose without deleting picks. With **Landmark Empties** on, that also removes its helper from `PM_Sync_Landmarks`.
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

Ordinary Sync needs the reference picked in at least two cameras allowed to
contribute 3D, or explicitly linked as Known 3D. Point-FOV fitting needs at least
two picked views and retains its existing point-only restrictions. Missing,
excluded, line or paired references are refused rather than replaced with a
static plane. Rename/reorder keeps the selection by landmark identity; after
deleting it, choose a replacement. Diagnose keeps the reference in place during
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

Diagnose labels these points **Plane + one view**. Their depth follows your plane constraint; a small pick error does not independently verify it. A pick in another camera allowed to contribute 3D adds a separate geometric check. Removing the plane or making the only contributing camera Fit Only removes the point unless other evidence determines it.

A hard plane established independently by other reconstructed geometry also helps fit existing free lines, including mirrored pairs. The line is fitted within the plane using its strokes. A compatible world-axis or Known 3D **Is Parallel To** direction remains fixed while fitting a free line inside that plane; incompatible directions cannot be enforced together. Compatible planes on mirrored partners preserve both plane membership and reflection. Free lines and points whose depth comes from that plane alone do not count as independent plane support for this check. This does not add reconstruction of an ordinary line with only one stroke.

Mirrored lines also retain a compatible **Is Parallel To** direction supplied by a world axis or a **Known 3D** edge. When a supported hard plane applies, reconstruction keeps that direction, plane membership and reflection together. A fixed direction alone does not determine the line's position: nearly coincident reflected strokes can still produce a **Weak 3D line support** warning. Nonzero Plane Slack does not turn the plane into a hard depth reference.

### Line landmarks

Add with the mesh icon next to +. Drag the same physical edge in each still — endpoints do **not** need to be the same 3D points, only the same infinite edge. Optional: assign two Empties as **Known 3D** / **Known 3D B** so the edge is metric. **Is Parallel To** can constrain the edge to shared-world **X Axis**, **Y Axis**, or **Z Axis**, or to another Line landmark that shares its 3D direction. **Is Mirror Of** pairs a line with its counterpart across the scene Mirror Empty, the same as for points. **Is in Plane** can keep a line in a wall, table, or other shared plane with point landmarks.

Without Known 3D ends, a free line needs **three or more** stills — two views alone cannot constrain relative pose from lines. Ordinary point landmarks must be picked in **both** stills when Known 3D sit on one line. Expand **Pick Confidence** (collapsed by default, under **Pick in Active Match**) to set the next-pick default or per-still confidence; it multiplies the landmark **Sync Weight**.

**What “px” means:** For **point** landmarks, RMSE is how far the projected 3D Empty lands from your 2D pick. For **line** landmarks, each drawn endpoint’s perpendicular distance to the projected infinite 3D line is measured. Those two distances combine offset (the stroke sitting beside the projected edge) and heading (angle miss scaled by half the stroke length): RMS = hypot(midpoint offset, ½ length × sin(angle)). A short stroke therefore reports a parallel miss more than a heading miss; a long stroke also punishes a twist. Pose accept still uses **point** RMSE, so a line that is not yet sitting on the overlay cannot skip a still that already fits the 3D cloud. After that still is placed, recovered-camera polish still uses the line (and spatially isolated picks) to rotate it — a dense cluster of well-fitting picks cannot Huber-ignore isolated landmarks that pin orientation.

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

**Solve Sync** seeds pairwise pose then runs a joint bundle-adjustment over Empty transforms + landmarks (Huber-weighted, with extra influence on poorly covered regions of each still so a cluster of central picks cannot ignore a few near the edge that pin camera distance). Raise **Sync Weight** on a landmark when those automatic boosts are not enough. Pairwise growth starts from the geometrically strongest still pair (spread, overlap, parallax, pair RMSE) and adds the easiest next camera — never alphabetical names, and not every still vs the Anchor just because it shares five picks. With only free 2D points and no pose locks, it connects the pair's better-spread member to the Anchor first, then compares each later direct-Anchor pose with bridges through registered views so a few matching pixels cannot pull the graph apart. The Anchor remains the shared world. 3D landmarks are triangulated from all registered rays, with near-parallel views downweighted, behind-camera views dropped, and a short reprojection polish. Options between Solve Sync and Refine Lenses:

- **Lock Rotation** — keep each Empty’s rotation on a 90° world-axis jump (identity, ±90°, 180° about X/Y/Z, including an X/Y swap); only solve translation/scale. Use when VP axes already match across stills so a free solve would only add a few degrees of noise.
- **Lock Translation** — keep Empty translation fixed; only solve rotation/scale.
- Both checked — leave cameras unmoved; only adjust 3D landmark / Empty positions.
- **Ground Slack** — how far On Ground landmarks may sit off Z=0 (scene units). 0 pins them to the floor raycast. The default (0.02) is enough for plank cup / tag thickness on a boarded floor.
- **Known 3D Slack** — how far Known 3D point landmarks may sit off their Empty (scene units). 0 (the default) pins them. A small value is a spring toward the Empty while each still's 2D pick pulls the point along that camera's ray, so CAD that is slightly wrong can share the error with the cameras instead of stretching the overlay. Linked Empties stay put; **Landmark Empties** show the eased positions. Known 3D lines stay pinned. **Use Known 3D** (Camera) still treats the Empty as fixed. A point that is both Known 3D and **On Ground** uses the tighter of the two slacks for Z.
- **Plane Slack** — how far **Is in Plane** landmarks may leave their shared plane (scene units). 0 (the default) pins them. A small value lets a slightly warped wall or table flex.
- **Mirror Empty / Plane / Mirror Slack** — one object whose chosen local face is the shared mirror for every **Is Mirror Of** pair. Slack 0 pins the plane to the Empty; a small value lets it slide along the normal. The Empty is not moved. Mirror Slack sits beside Plane Slack.

**Diagnose** measures sync quality without moving cameras, then opens a self-contained local HTML report in the default browser. The report leads with actionable problems, shows an interactive camera-overlap graph (pan, zoom, drag; hover a link for the shared-point count), names the best available registration route and its shared-point deficit, lists every match in a sortable table, and provides a searchable error-ranked landmark table. Technical solver text and constraint counts stay available in collapsible sections. **Open Report** reopens the newest temporary report; **Export** saves that single portable HTML file permanently. Reports use Blender's configured temporary directory, contain no remote resources, and are not uploaded.

Diagnose checks that its Sync inputs and active scene still match when the background job finishes. If picks, constraints, camera roles or other solver inputs changed, it keeps the current landmark errors and asks you to run Diagnose again. Unrelated modeling edits do not invalidate the job. Loading another file cancels the old Diagnose, Solve Sync and Refine Lenses jobs and allows a new job to start; old completion/cancel callbacks cannot take over the new job.

**Solve Sync** registers cameras in the same background way: the Solve row shows the current stage (including how many cameras are already posed) and elapsed time, and **Esc** or **Cancel** stops it. There is no cursor progress overlay. If picks, constraints, camera roles or other solver inputs change while it runs, it keeps the current cameras and asks you to run Solve Sync again. Unrelated modeling edits do not invalidate the job. Scripted `execute()` still runs blocking on the main thread. **Iterate Known 3D** still steps one blocking Solve Sync round per timer tick.

When error is high, Diagnose also runs leave-one-out checks on the worst landmarks. It runs in the background: the Diagnose row shows the current solve stage and elapsed time, and **Esc** or **Cancel** stops it. There is no cursor progress overlay, because most of the work happens before the first coarse stage completes. Leave-one-out keeps the camera graph accepted by the base solve, so a rejected still is not globally re-registered five more times. HTML reports belong to **Diagnose** only; **Solve Sync** keeps its normal Blender status and does not create or open a report. **Clear** resets sync transforms and forgets the current report link. Diagnose and Solve Sync cache each still-pair pose so a second run skips the expensive pairwise search when those two cameras' shared picks, Known 3D, and private K/pose are unchanged; **Clear** drops that cache. Independent still pairs on the first run are solved in parallel.

**Refine Lenses** searches focal length to lower reprojection error across supported point picks. The **Same Lens** checkbox and **%** field sit above the button. **Same Lens** (on by default) applies one scale to every still — use this when they share a physical camera / imported YAML; it does not need VP lines. Off normally uses a per-still VP search (re-orients from VP lines, skips 1-point / weak-VP stills). Coupled polish and Solve Sync follow, retaining the same plane groups, Plane Slack and other geometric constraints as Solve Sync. Alternatively, enable **Estimate FOV from Landmarks** for the independent no-VP workflow above; that mode applies the joint camera/point fit directly and supports point plane/mirror relations, with the limits above. Both run in a background thread — watch the progress slider, press **Esc** or **Cancel** to stop. The % field is the ± search window around current fx (default 18 for the existing searches, 40 for point FOV estimation). Disable unrelated matches or landmarks before refining a subset. Matches in **Adjusted Camera** mode are skipped so the button stays available for the others.

For the existing shared/VP searches, the lens score includes recovered stills whose errors may be excluded from Solve Sync’s joint headline. Once a successful candidate exists, later candidates must keep its registered cameras and reconstructed points/lines and remain successful. An initially refused solve can still improve its lenses, even if Sync continues to refuse. Line-only searches retain their existing line score. Point FOV estimation automatically applies only an accepted joint result; **Use Best Fit** explicitly applies an eligible provisional candidate with its warning.

If the initial Sync is rejected, Refine Lenses registers cameras afresh for new focal candidates until it obtains a successful solve. It then reuses the solved poses to speed subsequent trials. Explicit **Lock Pose** settings apply throughout; a rejected starting focal length does not prevent recovery when a suitable candidate lies within the search window.

Refine Lenses also checks its inputs and target cameras before applying a background result. Changing picks, constraints, origin or VP evidence, search settings, or camera transforms while it runs discards that result and preserves your edits. Switching the active match or editing an unrelated object does not invalidate it. If applying a valid result encounters an internal error, it restores the previous cameras, landmark state and cached plates and reports the error. A numerical Sync refusal still retains the refined lenses and reports the refusal.

**Iterate Known 3D** (beside Refine Lenses) repeats **Auto from VPs** with **Use Known 3D** then **Solve Sync** until joint RMSE stops falling. That is the hand-click loop that still moves FOV and camera after a match Empty has been registered: Known 3D is expressed in the private frame, so a new root transform changes the pins. Eligible stills need Use Known 3D, four Known 3D picks, and enough VP lines. Pose-locked non-anchor matches and Adjusted Camera stills are skipped. Esc / Cancel keeps the last improvement (at most eight rounds).

The eye icon on **Sync Matches** toggles landmark picks on the plate (same pattern as Vanishing Point Lines, Origin, and Camera); **Landmark Empties** controls the 3D helpers after sync. **Hide Origin Empty** is per match and hides that Origin Empty in the viewport (camera and collection stay visible); it stays in sync with the Origin's Outliner visibility. With the **Perspective Match** sidebar tab open, click a pick on the plate — or select that landmark’s Empty / line helper or Known 3D object in the viewport — to select it in the list (red overlay). Shift-selecting additional objects leaves the Sync landmark unchanged so the viewport multi-select is kept. Selecting a landmark in the list selects its Known 3D object or solved helper in the viewport (including when that object is parented). Inactive **On Ground** picks are magenta; Known 3D are cyan. **Pick in Active Match** can still place or move the active landmark; clicking a different pick selects it instead of overwriting. Dragging a new line rubber-band uses the same red as the selected pick, not the VP axis colors. **Snap to AprilTag** (point landmarks) looks around the click for a dark four-sided blotch with a brighter border and moves the pick to that quad's diagonal intersection. Per-match pick coordinates, confidence, and last-sync RMSE are under the collapsed **Pick Confidence** header.

## Debugging a bad or rejected sync

- **Rejected (~40+ px)** — no pose fits your picks. Status / **Diagnose** lists the worst landmarks — re-pick those features in *both* stills.
- **Plenty of picks, still rejected** — On Ground is load-bearing. Only landmarks that actually lie on the ground plane should be On Ground. A still that cannot lock is skipped so the others can still sync. After the remaining cameras lock, that still is retried as PnP against their triangulated 3D (floor tags alone if off-plane picks disagree). Diagnose / Solve Sync name the skipped match, and if one pick disagrees with the other stills they name that landmark (uncheck **Use in Sync** or re-pick it). A photo looking straight down at the ground is registered from those On Ground picks (plane homography); generic 2D↔2D pose is a poor fit there even when the tags are correct. If a portrait locked K was copied onto a landscape still of the same pixel count, Import YAML / copy keep the calibrated focal length (axes swap). Solve Sync sets fy=fx when they differ by more than 20%.
- **Accepted but camera looks wrong** with RMSE still a few–tens of px — wrong local minimum or soft constraints. Prefer **Diagnose**, fix the worst landmarks, **Clear**, then **Solve Sync** again. If a few picks far from the main cluster look sacrificed while the rest sit perfectly, raise those landmarks' **Sync Weight** (try 4–8) and solve again. Matches without Origin but with On Ground picks get an auto Origin on Sync; if tilt persists, add an elevated (off-ground) landmark or a 4th ground pick.
- **Flipped the object for underside photos** — On Ground is one shared plane. Table tags from the flipped session must not be On Ground if the original table tags already pin Z=0. Side tags glued to the object *are* the same 3D points and do connect the graphs. After a physical flip, a camera that photographed the underside is placed *below* the original ground looking up (object frame), not above the table in room coordinates.
- **List shows ~1px but the Empty is far from the pick** — the px number is RMSE across the stills that participated in the last solve, not the currently viewed still. Landmarks that only appear on recovered / hanging stills used to keep a stale number; Solve Sync now triangulates those tags and poses the hanging still from them. If the Empty is still off, switch to that still and compare this match's residual under **Pick Confidence**.
- **One landmark huge, others fine** — that pick is mismatched (same ID / feature on a different physical point). Uncheck **Use in Sync** and re-run Diagnose. If a still was skipped, Diagnose names the pick on that still.
- **Many landmarks all high** — FOV or VP solve is likely off on one match; try **Iterate Known 3D** (Use Known 3D on that still) or **Refine Lenses**, or re-refine that camera manually.
- **Sync broke after adding one landmark** — turn off **Use in Sync** on the new one and Diagnose again.
- **Known 3D warn (Empty vs anchor pick)** — the Empty moved or the anchor camera changed; re-run **Landmarks from Selected**.
- **Landmarks jump on the plate when panning one still** — that match Empty was shrunk to a point (typical for a below-ground camera). Switch to the match or re-run **Solve Sync**; the camera should stay below-ground, but overlay picks stay on the photo.
- **A free line should follow a world axis** — set **Is Parallel To** to X Axis, Y Axis, or Z Axis. For two arbitrary parallel edges, link the bad free line to a better-fitting free line or a Known 3D edge.
- **Weak 3D line support** — the strokes and geometric constraints supply nearly coincident supporting planes, so small pick errors can move or rotate the reconstructed line substantially despite a good pixel fit. Solve Sync names these lines; Diagnose marks their rows and reports the best supporting-plane angle. Mirror pairs include reflected views in this check. An independently established hard shared plane counts when the reconstructed line actually lies in it. Try longer strokes or a stroke from a more distinct viewing angle, using a camera in **Solve** or **Lock Pose**. Fit Only strokes do not supply reconstruction support or clear this warning. Known 3D lines and their reflected partners get geometry directly and are exempt. This warning preserves the solved geometry; it is a sensitivity indicator, not a confidence interval or proof that unflagged lines are correct.
