# Locally biased Known 3D references

## Protocol

One generated anchor-gauge scene (seed 17) uses the same truth and picks in every condition. Two of four nearby Known 3D points receive a local `[0.16, -0.06, 0.05]` scene-unit offset (magnitude 0.178); the other two remain truthful. Six unbiased On Ground observations stay constrained hard at Z=0, so the metric/world frame does not come from aligning the answer to truth. `view_1` and `view_2` remain fully solved cameras and are checked on withheld 3D object samples.

The soft setting is `0.20` scene units. Slack is a spring scale, not a bound. The four predeclared conditions are unbiased/hard, unbiased/soft, biased/hard, and biased/soft. The ordinary oracle limits remain unchanged; hard wrong references are intentional model conflict.

Baseline revision `80a6d829ab7c8065a55383e09b40ffc84ed5515a`; Python 3.14.7, NumPy 2.4.1, macOS-26.5-arm64-arm-64bit-Mach-O. Four solves took 1.21 s in total. Case filenames are `tools/synthetic_sync/cases/biased-reference-exact-*.json`.

Replay from the repository root with `python3 tools/synthetic_sync/run.py --case tools/synthetic_sync/cases/biased-reference-exact-biased-soft.json --out /tmp/pm-biased-replay` (change the filename for the other conditions; a biased oracle flag gives `run.py` a nonzero exit). To regenerate all four conditions, use `python3 tools/synthetic_sync/biased_references.py --out /tmp/pm-biased-comparison`.

## Results

| Picks | Condition | Fit all px | Fit biased px | Fit truthful px | Biased truth RMS | Biased prior gap | Truthful ref RMS | view_1 holdout px | view_2 holdout px | Ground max | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact | unbiased-hard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 | pass |
| exact | unbiased-soft | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 | pass |
| exact | biased-hard | 6.817 | 13.158 | 5.038 | 0.178 | 0.000 | 0.000 | 2.212 | 6.978 | 0.000000 | flag |
| exact | biased-soft | 0.916 | 1.571 | 0.616 | 0.025 | 0.154 | 0.008 | 1.199 | 1.359 | 0.000000 | flag |

Every Known 3D point has three-view support. The camera evidence is 9 point picks in `view_0`, 9 in `view_1`, and 8 in `view_2`; the last view sees four rather than five ground points.

### Biased point coordinates

| Condition | Point | Truth XYZ | CAD prior XYZ | Reconstructed XYZ | Truth error | Prior gap |
| --- | --- | --- | --- | --- | ---: | ---: |
| biased-hard | point_44 | [0.2820, -0.0875, 2.0000] | [0.4420, -0.1475, 2.0500] | [0.4420, -0.1475, 2.0500] | 0.1780 | 0.0000 |
| biased-hard | point_45 | [0.5880, -0.0680, 2.0000] | [0.7480, -0.1280, 2.0500] | [0.7480, -0.1280, 2.0500] | 0.1780 | 0.0000 |
| biased-soft | point_44 | [0.2820, -0.0875, 2.0000] | [0.4420, -0.1475, 2.0500] | [0.3066, -0.0912, 2.0042] | 0.0252 | 0.1536 |
| biased-soft | point_45 | [0.5880, -0.0680, 2.0000] | [0.7480, -0.1280, 2.0500] | [0.6121, -0.0720, 2.0043] | 0.0248 | 0.1540 |

### Solver messages and flags

- **unbiased-hard:** Synced 2 match(es) · 10 landmarks · RMSE 0.00 px · constraints: 4 known 3D + 6 ground · joint BA Flags: none.
- **unbiased-soft:** Synced 2 match(es) · 10 landmarks · RMSE 0.00 px · constraints: 4 known 3D + 6 ground · joint BA Flags: none.
- **biased-hard:** Synced 2 match(es) · 10 landmarks · RMSE 6.82 px · constraints: 4 known 3D + 6 ground · joint BA Flags: point_44: reconstructed point error 0.04938 of object diagonal; point_45: reconstructed point error 0.04938 of object diagonal; view_1: holdout_rmse_px 2.212 > 1; view_2: holdout_rmse_px 6.978 > 1; view_2: center_fraction 0.0308 > 0.02.
- **biased-soft:** Synced 2 match(es) · 10 landmarks · RMSE 0.92 px · constraints: 4 known 3D + 6 ground · joint BA Flags: view_1: holdout_rmse_px 1.199 > 1; view_2: holdout_rmse_px 1.359 > 1.

## Interpretation

On exact picks, biased-point truth RMS changed 0.178 → 0.025 scene units (improvement 0.153) and the worst non-anchor withheld-camera RMS changed 6.978 → 1.359 px (improvement 5.619) from biased hard to biased soft. Soft biased points pass the unchanged direct-point oracle; the overall soft biased case flags its unchanged oracle (view_1: holdout_rmse_px 1.199 > 1; view_2: holdout_rmse_px 1.359 > 1). Unbiased controls both pass; their biased-group truth RMS differs by 0.000000 scene units. A low fitted-pick error alone cannot establish geometric accuracy.

The JSON results retain per-camera support, per-group fitted residuals, every Known 3D truth/prior gap, withheld camera pose/projection metrics, request fingerprints, solver messages, and all oracle flags.

This is one seed, one local bias, one slack setting, exact picks and ideal calibrated pinhole cameras; it does not establish sensitivity over other geometries or noise. Known 3D slack softens every Known 3D point, including the two truthful references. The solver messages here report fit/constraints but do not warn of the intentional reference conflict. This numerical runner omits Blender preparation: Diagnose already warns when a Known 3D Empty differs from its stored anchor pick by more than 5 px (`scene.known_anchor_pick_warnings`). The experiment does not establish a gap in that existing check or a calibrated mismatch diagnostic.
