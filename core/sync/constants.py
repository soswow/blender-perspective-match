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

# Default Mirror Slack: 0 pins the plane to the Empty (Empty is not moved).
MIRROR_SLACK_DEFAULT = 0.0

# Soft plane-offset residual at |δ| = slack equals this many pixels before Huber.
MIRROR_PLANE_RESIDUAL_PX = 6.0

# Pair reflection gap of this many scene units equals MIRROR_PAIR_RESIDUAL_PX.
MIRROR_PAIR_HARD_GAP = 0.01

# Soft XYZ residual for B − reflect(A) at |gap| = MIRROR_PAIR_HARD_GAP.
MIRROR_PAIR_RESIDUAL_PX = 6.0

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
