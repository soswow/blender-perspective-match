"""Named thresholds for landmark-graph sync.

Keep these in one place so peel, pairwise accept, and K-stretch repair
do not drift apart. Update AGENTS.md if a stage or constant changes.
"""

from __future__ import annotations

import numpy as np

# Stable EnumProperty identifiers for Line > Is Parallel To world-axis targets.
# These are graph nodes, not landmark IDs; their values must remain stable so
# saved .blend files keep their selection across add-on upgrades.
WORLD_AXIS_DIRECTIONS = {
    "WORLD_AXIS_X": np.array((1.0, 0.0, 0.0), dtype=np.float64),
    "WORLD_AXIS_Y": np.array((0.0, 1.0, 0.0), dtype=np.float64),
    "WORLD_AXIS_Z": np.array((0.0, 0.0, 1.0), dtype=np.float64),
}

# Pixel RMSE above which a camera is peeled from the joint graph, and
# below which a pairwise or resected pose is accepted.
ACCEPT_RMSE_PX = 40.0

# Only the worst failed-pose picks get a warm leave-one-out resection check.
RESECT_MISMATCH_CANDIDATE_LIMIT = 5

# |fx-fy| / max(fx, fy) above this: treat K as aspect-stretched and set fy=fx.
STRETCHED_PIXEL_RATIO = 0.2

# |Z| vs point scale (and On Ground vs triangulation) for "on the ground plane".
GROUND_PLANE_Z_FRACTION = 0.15

# Default On Ground Z slack in Blender units (plank cup / tag thickness).
# 0 pins On Ground to the Z=0 raycast when triangulation agrees.
GROUND_SLACK_DEFAULT = 0.02

# Soft Z residual at |z| = slack equals this many pixels before Huber.
GROUND_Z_RESIDUAL_PX = 6.0

# Default Known 3D XYZ slack in Blender units. 0 pins points to the Empty.
KNOWN_3D_SLACK_DEFAULT = 0.0

# Soft XYZ residual at |offset| = slack equals this many pixels before Huber.
KNOWN_3D_RESIDUAL_PX = 6.0

# Default Mirror Plane Slack: 0 pins the plane to the Empty (Empty is not moved).
MIRROR_SLACK_DEFAULT = 0.0

# Soft plane-offset residual at |δ| = slack equals this many pixels before Huber.
MIRROR_PLANE_RESIDUAL_PX = 6.0

# Default permitted point/line mirror-pair position mismatch in scene units.
# Zero is represented exactly by the common fit rather than by a stiff spring.
MIRROR_PAIR_SLACK_DEFAULT = 0.0

# Legacy initializer mirror-pair target. The common fit uses explicit
# Mirror Pair Slack and never treats this value as an implicit allowance.
MIRROR_PAIR_HARD_GAP = 0.01

# Soft XYZ residual for B − reflect(A) at |gap| = MIRROR_PAIR_HARD_GAP.
MIRROR_PAIR_RESIDUAL_PX = 6.0
# Numerical-only allowance for float32 Blender persistence of exact mirror
# geometry, scaled by the represented scene extent at certification time.
MIRROR_PAIR_EXACT_RELATIVE_TOLERANCE = 8.0 * float(np.finfo(np.float32).eps)

# Shared-plane buckets (Is in Plane). Axis X/Y/Z share that coordinate;
# Free fits an unknown plane. 0 slack is a hard pin (tiny spring).
PLANE_SLACK_DEFAULT = 0.0
PLANE_RESIDUAL_PX = 6.0
PLANE_HARD_SLACK = 1.0e-4
PLANE_GROUP_LIMIT = 10
PLANE_AXIS_ALIGNED_MIN = 2
PLANE_FREE_MIN = 4
PLANE_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

# When On Ground is a hard Z=0 pin (slack 0) but Known 3D is thawed, BA still
# needs a Z spring. This tiny slack is stiff enough to keep |Z| ≈ 0.
GROUND_Z_HARD_SLACK = 1.0e-4

# LM clips log-scale to ± this. Hitting the floor collapses the match Empty
# (det ≈ 0) so overlay corners project as noise and jump when the view pans.
LOG_SCALE_CLIP = 18.0

# Multiply an outlier landmark's pick weight by this in joint BA.
OUTLIER_WEIGHT_FACTOR = 0.15

# Landmark Sync Weight above this keeps full BA pull (no outlier downweight).
SYNC_WEIGHT_PROTECT = 1.0

# Free-line 3D geometry anchors to pose-locked cameras once this many see it.
LINE_FIXED_ANCHOR_MIN = 2
# Prefer separated interpretation planes; flag free lines with only weaker support.
LINE_PLANE_MIN_SINE = 0.12
# Compatible directions may differ by one float32 epsilon after Blender storage.
LINE_CONSTRAINT_DIRECTION_TOLERANCE = float(np.finfo(np.float32).eps)
# Truncate per-view line RMSE when ranking reconstruction pairs.
LINE_RECONSTRUCT_TRUNCATE_PX = 80.0

# Recovered pose-only BA Huber. Joint BA uses 6 px because outliers hide in
# hundreds of picks. A resected still often has a tight inlier cluster plus
# one isolated line or point that pins yaw; 6 px treats those as outliers.
RECOVERED_HUBER_DELTA_PX = ACCEPT_RMSE_PX

# Existing joint-BA acceptance budget, also applied per previously solved view
# when assessing a recovered camera's proposed 3D update.
BA_ACCEPT_RMSE_FLOOR_PX = 8.0
BA_ACCEPT_RMSE_SLACK_PX = 2.0

# Joint BA freezes 3D and refines poses only above this landmark count.
# Triangulation stays the 3D prior so cameras move instead of a few
# landmarks absorbing edge error.
BA_FREE_LANDMARK_LIMIT = 40

# Per-camera image grid for spatial residual balancing. Each occupied cell
# gets similar total weight so a cluster of central tags cannot outvote a
# few peripheral picks that pin camera distance.
SPATIAL_GRID_SIZE = 3
SPATIAL_WEIGHT_CLIP = 4.0
# Extra leverage for picks far from the principal point: 1 + gain * r_norm^2.
RADIAL_WEIGHT_GAIN = 2.0

# N-view triangulation: Gauss–Newton reprojection steps after the linear midpoint.
TRIANGULATION_GN_STEPS = 4
# Floor on sin²(angle) so a near-parallel extra view cannot zero a ray weight.
TRIANGULATION_ANGLE_WEIGHT_FLOOR = 1.0e-3
# Rays with direction cosine above this share one stereo weight (same viewpoint).
TRIANGULATION_PARALLEL_COSINE = 0.995

# Investigate Problems bounds its shared-fit counterfactual work separately
# from the ordinary Solve Sync route.
DIAGNOSE_COMMON_MAX_CANDIDATES = 5
DIAGNOSE_COMMON_MAX_SECONDS = 60.0
