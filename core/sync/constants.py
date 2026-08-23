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
