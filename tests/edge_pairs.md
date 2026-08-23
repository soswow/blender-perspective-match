# Pair covering

Find pipeline bugs **before** a real shoot: list independent axes from the data flow, then require a test where **two** non-default values co-exist.

Oracle: generate picks from a true camera; feed the solver the **stored** K/pose the UI would have (copied, remapped, leftover). Tests that create and solve the same perfect object do not count as covering a pipeline pair.

Before changing remap / bind / copy-K / `solve_landmark_sync`: scan **Untested** below. Add a test in `test_edge_pairs.py` (or mark N/A with why). Do not only add a story after a miss.

## Axes (from inputs, not from incidents)

| Axis | Values |
| --- | --- |
| A. Plate vs calibrated size | same · uniform scale · exact W↔H swap · crop (neither) |
| B. How K arrives | native on this still · Import YAML remap · copy from another match · leftover already in the session |
| C. Tilt | oblique · nadir · from below |
| D. Structure | ground only · ground + raised · raised only in one still |
| E. Pose seed | identity / true · leftover look-at |
| F. Graph | all views overlap · one still un-registerable · one still ground-only |

## Pairs

| Pair | Test | Status |
| --- | --- | --- |
| A same × B native | `test_ros_camera_info.RosCameraInfoTests.test_no_scale_when_sizes_match` | covered |
| A uniform scale × B YAML | `test_scale_intrinsics` | covered |
| A swap × B YAML (portrait→landscape) | `test_portrait_yaml_on_landscape_swaps_axes` | covered |
| A swap × B YAML (landscape→portrait) | `test_edge_pairs.KRemapPairTests.test_swap_landscape_yaml_onto_portrait` | covered |
| A crop × B YAML | `test_edge_pairs.KRemapPairTests.test_crop_is_scaled_not_rotated` | covered |
| A swap × C nadir × D raised × E leftover | `test_edge_pairs.SyncPairTests.test_nadir_landscape_after_portrait_yaml` | covered |
| A swap × D raised-only on nadir | `test_edge_pairs.SyncPairTests.test_nadir_raised_only_does_not_crash` | covered |
| C from-below × overlap | `test_graph_bridge_recovers_camera_below_ground` | covered |
| C from-below × D raised-only × E leftover | `test_edge_pairs.SyncPairTests.test_from_below_raised_only_keeps_metric_scale` | covered |
| F un-registerable × rest lock | `test_partial_sync_keeps_registered_matches` | covered |
| C nadir × leftover × same K | `test_nadir_camera_registers_from_ground_plane` | covered |
| A swap × 180° (same W×H, upside-down JPEG) | — | N/A — size remap cannot see 180° |
| A swap × Brown–Conrady p1/p2 axes | — | untested |
| B leftover wrong **width-scale** K × C nadir × D raised | — | untested (old blends until YAML re-import) |
| Two different calibrated zooms in one graph | — | untested |
| F ground-only still × C nadir | — | untested |

Helpers: `tests/pair_fixtures.py`.
