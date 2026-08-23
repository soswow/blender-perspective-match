"""Named thresholds for landmark-graph sync.

Keep these in one place so peel, pairwise accept, and K-stretch repair
do not drift apart. Update AGENTS.md if a stage or constant changes.
"""

from __future__ import annotations

# Pixel RMSE above which a camera is peeled from the joint graph, and
# below which a pairwise or resected pose is accepted.
ACCEPT_RMSE_PX = 40.0

# |fx-fy| / max(fx, fy) above this: treat K as aspect-stretched and set fy=fx.
STRETCHED_PIXEL_RATIO = 0.2

# |Z| vs point scale (and On Ground vs triangulation) for "on the ground plane".
GROUND_PLANE_Z_FRACTION = 0.15

# LM clips log-scale to ± this. Hitting the floor collapses the match Empty
# (det ≈ 0) so overlay corners project as noise and jump when the view pans.
LOG_SCALE_CLIP = 18.0

# Multiply an outlier landmark's pick weight by this in joint BA.
OUTLIER_WEIGHT_FACTOR = 0.15

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
