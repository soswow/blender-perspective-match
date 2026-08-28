# Dump Perspective Match sync state

Headless diagnostic for a `.blend`: match calibrations, optical-axis tilt vs world Z, landmark overlap, pairwise registration RMSE, and a Diagnose-style solve. Does not write the blend.

Agents: if you invent another dump or probe while debugging, check it in here (or under `tools/`) and add a line in `AGENTS.md`.

## Run

From the repo root (Blender 5.1, `--factory-startup` so only this checkout’s add-on loads):

```sh
BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender 5.1.app/Contents/MacOS/blender}"

"$BLENDER_BIN" --factory-startup -b --python tools/debug-sync/dump_sync.py -- \
    --blend "/path/to/scene.blend" \
    --out /tmp/pm-sync-dump.txt
```

Prints the same report to stdout. `--out` is optional.

Landmark overlap, stored Empty vs pick, timed solve, per-observation residuals:

```sh
"$BLENDER_BIN" --factory-startup -b --python tools/debug-sync/probe_graph.py -- \
    --blend "/path/to/scene.blend" --out /tmp/pm-graph.txt
```

Add `--leave-one-out` to time the accepted-camera leave-one-out report after
the base solve. The probe also prints each relative-pose call live, including
its stage and the number/time of PnP and mixed-refinement attempts.

Recovered-still overlay (ground vs off-plane RMSE after `solve_landmark_sync`):

```sh
"$BLENDER_BIN" --factory-startup -b --python tools/debug-sync/probe_resected.py -- \
    --blend "/path/to/scene.blend" [--match PM_some_Origin]
```

Per-match K, distortion, Blender lens/FOV, undistorted-plate flags:

```sh
"$BLENDER_BIN" --factory-startup -b --python tools/debug-sync/probe_cameras.py -- \
    --blend "/path/to/scene.blend" --out /tmp/pm-cameras.txt
```

## What to look at

- **Stored nadir_deg** — current Blender camera vs world ±Z. Only meaningful if that match already locked; unsynced cameras often still sit on a leftover default pose.
- **empty_s** — match-Empty scale from `matrix_world`. Values near `exp(-18) ≈ 1.5e-8` mean a collapsed log-scale pose; overlay corners project as one point and landmarks jump when panning. Re-activate the match or Solve Sync to rewrite it as rigid `s=1`.
- **rec_nadir_deg** — optical axis of the *recovered* pairwise pose in the anchor/shared frame. Near 0° = looking along ±Z (image plane ≈ ground). That is the planar / essential-matrix degeneracy, even when On Ground tags are correct.
- **H_rmse** — DLT homography residual between the two stills. Low H_rmse with high mixed_rmse means the correspondence is planar (homography fits; metric PnP does not).
- **center_vs_z_deg** — stored ray through the image center (differs from stored nadir_deg when PP is off-center).
- **Pairwise vs anchor** — `2d_rmse` ignores On Ground metric; `mixed_rmse` includes ground raycasts. A huge mixed error with a modest 2D error is that degeneracy, not “too few landmarks” and not mis-tagged ground.
- **Residual vs radius** (`probe_graph.py`) — per-match inner (r<0.35) vs outer (r≥0.55) RMSE. Outer much larger than inner means a central cluster is winning and edge picks that pin camera distance are being ignored.
