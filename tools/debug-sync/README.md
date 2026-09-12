# Dump Perspective Match sync state

Headless diagnostic for a `.blend`: match calibrations, optical-axis tilt vs world Z, landmark overlap, pairwise registration RMSE, and a Diagnose-style solve. Does not write the blend. `dump_sync.py`, `probe_graph.py`, and `probe_resected.py` pass the same sidebar locks, slack, and **This Camera** role as Solve Sync.

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

For independent point-FOV eligibility, run `probe_lens_inputs.py` with the same
`--blend` and optional `--out` arguments (add `--disable-autoexec` to Blender's
arguments when inspecting an external file). It lists per-camera point counts,
line landmarks, and named plane/mirror members with their point views. By
default it only collects inputs. Optional `--fit-seconds 180` runs a cooperative
time-bounded trial without applying it; with `--out report.json`, numerical
inputs and startup state are retained in `report.json.inputs.json` and
`report.json.startup.json`. These can contain private scene evidence: keep them
outside the repository. No mode saves the source blend.

The main solves in `dump_sync.py`, `probe_graph.py` and `probe_resected.py` use
the same complete prepared request as Solve Sync and Diagnose, including live
pose locks, workspace rotation/translation locks, all slack settings, confidence
and geometric constraints. Preparation can initialize ground/origins **in
memory**, as the product does; these tools never save the source `.blend`.
`probe_graph.py --no-solve` inspects stored evidence without that preparation.
Pairwise sub-experiments in the dump remain deliberately simplified and labelled.

Landmark overlap, stored Empty vs pick, timed solve, per-observation residuals:

```sh
"$BLENDER_BIN" --factory-startup -b --python tools/debug-sync/probe_graph.py -- \
    --blend "/path/to/scene.blend" --out /tmp/pm-graph.txt
```

Add `--leave-one-out` to time the accepted-camera leave-one-out report after
the base solve. The probe also prints each relative-pose call live, including
its stage and the number/time of PnP and mixed-refinement attempts.
Add `--snapshot /tmp/pm-request.json` to retain the exact prepared request before
solving. This option cannot be combined with `--no-solve`.

## Capture once, replay without Blender

```sh
"$BLENDER_BIN" --factory-startup --disable-autoexec -b --python-exit-code 1 \
    --python tools/sync_snapshot.py -- capture \
    --blend "/path/to/scene.blend" --out /tmp/pm-request.json

python3 tools/sync_snapshot.py replay /tmp/pm-request.json --out /tmp/pm-replay
```

Use a new JSON filename and output directory. Capture copies the product's
prepared numerical input, not the `.blend`, images or helper meshes. It preserves
full calibration (including stored distortion coefficients), points/strokes,
weights and outlier protection, constraints, live fixed similarities, camera
roles (including Fit Only), and all slack/lock settings. No new distortion simulation is involved. A checksum covers
the evidence and its ordering; metadata records the code/runtime and preparation
notes separately. Schema version 2 includes camera roles; version 1 snapshots
are explicitly migrated with the original unrestricted-camera semantics.
Missing fields and unknown versions are rejected.

Replay needs Python and NumPy. It writes `result.json` with the complete solver
result and effective calibrations, plus the existing portable diagnostic HTML.
A refused solve exits 1; an exception is recorded separately and also fails.
`--cache` enables caching within replay; cache contents are not part of a snapshot.
The tool neither applies results to a scene nor saves the input project.

These snapshots contain names and numerical project geometry. Keep private
captures local. Public regressions should use generic synthetic geometry as
required by `AGENTS.md`. A numerical snapshot is **not** independent ground truth:
its report describes fit and diagnostics, not reconstruction accuracy. Use
`tools/synthetic_sync/` for withheld-object validation. A viewport/undo/image-state
bug still needs a Blender operation case, since this snapshot starts after
preparation and image-coordinate conversion. Keep the code revision with a
snapshot; a dirty-tree flag cannot reconstruct uncommitted code.

The generated parity test is:

```sh
"$BLENDER_BIN" --factory-startup --disable-autoexec -b --python-exit-code 1 \
    --python tools/synthetic_sync/verify_requests.py -- --out /tmp/pm-parity
```

It compares every forwarded field and the numerical results for Solve Sync,
Diagnose and all three probes against restored snapshots. Four isolated Blender
processes cover imperfect Known 3D/mirror references with nonzero slack and a
live pose lock, global locks, an unset origin, and Fit Only participation. Global-lock refusal is compared
as a refusal, not required to become a successful solve. Leave-one-out forwarding
is checked without repeating its separate numerical test suite. A source-file
checksum verifies that probes did not save the generated input. These are request
and result parity checks; they do not establish accuracy under arbitrary slack.

## Additional probes

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

Known 3D camera pin polish (source-image residuals vs VP lines, rotation locked):

```sh
"$BLENDER_BIN" --factory-startup -b --python tools/debug-sync/probe_pin_refine.py -- \
    --blend "/path/to/scene.blend" --match PM_some_Origin
```

## What to look at

- **Stored nadir_deg** — current Blender camera vs world ±Z. Only meaningful if that match already locked; unsynced cameras often still sit on a leftover default pose.
- **empty_s** — match-Empty scale from `matrix_world`. Values near `exp(-18) ≈ 1.5e-8` mean a collapsed log-scale pose; overlay corners project as one point and landmarks jump when panning. Re-activate the match or Solve Sync to rewrite it as rigid `s=1`.
- **rec_nadir_deg** — optical axis of the *recovered* pairwise pose in the anchor/shared frame. Near 0° = looking along ±Z (image plane ≈ ground). That is the planar / essential-matrix degeneracy, even when On Ground tags are correct.
- **H_rmse** — DLT homography residual between the two stills. Low H_rmse with high mixed_rmse means the correspondence is planar (homography fits; metric PnP does not).
- **center_vs_z_deg** — stored ray through the image center (differs from stored nadir_deg when PP is off-center).
- **Pairwise vs anchor** — `2d_rmse` ignores On Ground metric; `mixed_rmse` includes ground raycasts. A huge mixed error with a modest 2D error is that degeneracy, not “too few landmarks” and not mis-tagged ground.
- **Residual vs radius** (`probe_graph.py`) — per-match inner (r<0.35) vs outer (r≥0.55) RMSE. Outer much larger than inner means a central cluster is winning and edge picks that pin camera distance are being ignored.
