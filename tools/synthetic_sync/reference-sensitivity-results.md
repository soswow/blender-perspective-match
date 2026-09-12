# Known 3D prior-release sensitivity pilot

## Protocol

Eight predeclared cases: four frozen exact conditions and the same four with the generator's fixed 0.3 px noise draw. Each original solve is paired with one solve after removing all Known 3D point priors. The points retain their image picks; cameras, calibration, hard ground, roles, and every other request field are unchanged. The diagnostic sees only the request and two result records. No truth or condition metadata enters its signal, and no threshold produces a warning/classification.

Revision `16a6cb682ba0213fbf722741f746889be589080e`; 16 numerical solves; 25.67 s solver time. No alignment to truth was fitted. The requested anchor plus hard On Ground references define the comparison frame.

Replay from the repository root: `python3 tools/synthetic_sync/reference_sensitivity.py --out /tmp/pm-reference-sensitivity-replay` (choose an absent output directory). Each case, request, result pair, and exact per-case diagnostic is retained there.

## Results

| Picks | Condition | Stored anchor >5 px | Fit original → release px | Max reference fit original → release px | Max prior gap original → release | Max camera projection shift px | Lost cameras/points/picks | Comparable | Original → release oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| exact | unbiased-hard | 0 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 | 0/0/0 | yes | pass → pass |
| exact | unbiased-soft | 0 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 → 0.000 | 0.000 | 0/0/0 | yes | pass → pass |
| exact | biased-hard | 2 | 6.817 → 0.000 | 13.239 → 0.000 | 0.000 → 0.178 | 9.014 | 0/0/0 | yes | flag → pass |
| exact | biased-soft | 2 | 0.916 → 0.000 | 1.586 → 0.000 | 0.154 → 0.178 | 2.759 | 0/0/0 | yes | flag → pass |
| noise-0p3 | unbiased-hard | 0 | 0.616 → 0.568 | 0.684 → 0.595 | 0.000 → 0.009 | 1.084 | 0/0/0 | yes | pass → pass |
| noise-0p3 | unbiased-soft | 0 | 0.564 → 0.568 | 0.556 → 0.595 | 0.008 → 0.009 | 0.191 | 0/0/0 | yes | pass → pass |
| noise-0p3 | biased-hard | 2 | 6.735 → 0.568 | 13.232 → 0.595 | 0.000 → 0.180 | 9.904 | 0/0/0 | yes | flag → pass |
| noise-0p3 | biased-soft | 2 | 1.035 → 0.568 | 1.558 → 0.595 | 0.155 → 0.180 | 2.701 | 0/0/0 | yes | flag → pass |

### Per-reference prior gaps after release (scene units)

| Picks | Condition | Reference gaps (sorted ID order) |
| --- | --- | --- |
| exact | unbiased-hard | point_44: 0.0000, point_45: 0.0000, point_46: 0.0000, point_47: 0.0000 |
| exact | unbiased-soft | point_44: 0.0000, point_45: 0.0000, point_46: 0.0000, point_47: 0.0000 |
| exact | biased-hard | point_44: 0.1780, point_45: 0.1780, point_46: 0.0000, point_47: 0.0000 |
| exact | biased-soft | point_44: 0.1780, point_45: 0.1780, point_46: 0.0000, point_47: 0.0000 |
| noise-0p3 | unbiased-hard | point_44: 0.0054, point_45: 0.0031, point_46: 0.0090, point_47: 0.0049 |
| noise-0p3 | unbiased-soft | point_44: 0.0054, point_45: 0.0031, point_46: 0.0090, point_47: 0.0049 |
| noise-0p3 | biased-hard | point_44: 0.1763, point_45: 0.1796, point_46: 0.0090, point_47: 0.0049 |
| noise-0p3 | biased-soft | point_44: 0.1763, point_45: 0.1796, point_46: 0.0090, point_47: 0.0049 |

### Per-camera fit and projection movement

| Picks | Condition | Camera | Fitted picks original → release | Fit RMS original → release px | Projection shift RMS / max px |
| --- | --- | --- | ---: | ---: | ---: |
| exact | unbiased-hard | view_0 | 9 → 9 | 0.000 → 0.000 | 0.000 / 0.000 |
| exact | unbiased-hard | view_1 | 9 → 9 | 0.000 → 0.000 | 0.000 / 0.000 |
| exact | unbiased-hard | view_2 | 8 → 8 | 0.000 → 0.000 | 0.000 / 0.000 |
| exact | unbiased-soft | view_0 | 9 → 9 | 0.000 → 0.000 | 0.000 / 0.000 |
| exact | unbiased-soft | view_1 | 9 → 9 | 0.000 → 0.000 | 0.000 / 0.000 |
| exact | unbiased-soft | view_2 | 8 → 8 | 0.000 → 0.000 | 0.000 / 0.000 |
| exact | biased-hard | view_0 | 9 → 9 | 8.099 → 0.000 | 0.000 / 0.000 |
| exact | biased-hard | view_1 | 9 → 9 | 5.657 → 0.000 | 2.269 / 3.278 |
| exact | biased-hard | view_2 | 8 → 8 | 6.420 → 0.000 | 6.414 / 9.014 |
| exact | biased-soft | view_0 | 9 → 9 | 1.117 → 0.000 | 0.000 / 0.000 |
| exact | biased-soft | view_1 | 9 → 9 | 1.034 → 0.000 | 1.550 / 2.759 |
| exact | biased-soft | view_2 | 8 → 8 | 0.350 → 0.000 | 1.261 / 2.080 |
| noise-0p3 | unbiased-hard | view_0 | 9 → 9 | 0.233 → 0.388 | 0.000 / 0.000 |
| noise-0p3 | unbiased-hard | view_1 | 9 → 9 | 0.511 → 0.431 | 0.141 / 0.187 |
| noise-0p3 | unbiased-hard | view_2 | 8 → 8 | 0.937 → 0.820 | 0.702 / 1.084 |
| noise-0p3 | unbiased-soft | view_0 | 9 → 9 | 0.344 → 0.388 | 0.000 / 0.000 |
| noise-0p3 | unbiased-soft | view_1 | 9 → 9 | 0.435 → 0.431 | 0.046 / 0.069 |
| noise-0p3 | unbiased-soft | view_2 | 8 → 8 | 0.829 → 0.820 | 0.129 / 0.191 |
| noise-0p3 | biased-hard | view_0 | 9 → 9 | 8.068 → 0.388 | 0.000 / 0.000 |
| noise-0p3 | biased-hard | view_1 | 9 → 9 | 5.620 → 0.431 | 2.190 / 3.200 |
| noise-0p3 | biased-hard | view_2 | 8 → 8 | 6.218 → 0.820 | 7.083 / 9.904 |
| noise-0p3 | biased-soft | view_0 | 9 → 9 | 1.047 → 0.388 | 0.000 / 0.000 |
| noise-0p3 | biased-soft | view_1 | 9 → 9 | 1.097 → 0.431 | 1.529 / 2.701 |
| noise-0p3 | biased-soft | view_2 | 8 → 8 | 0.945 → 0.820 | 1.391 / 2.271 |

## Interpretation

Reference gaps and camera movement show sensitivity to removing the priors. They do not by themselves identify a faulty reference: ambiguity, weak support, or shared calibration error can cause the same movement. The existing product warning compares each Known 3D prior with its stored anchor pick and flags deltas above 5 px; its numerical counterpart is shown above. A release trial can add multiview fit and movement evidence, but these cases already have anchor picks and therefore test incremental value against that cheaper guard. Compare accurate exact and noisy controls before interpreting a biased case. The per-case JSON retains every reference displacement, per-reference/per-camera fitted residual, projection movement on baseline reconstructed landmarks, support count/loss, and independent withheld-truth assessment.

This is one constructed scene, one local bias, one fixed noise draw, and ideal pinhole calibration. The withheld oracle is used only after the signal is computed; its flags assess whether the observed sensitivity was useful, not whether a particular prior is intrinsically wrong. The stored-anchor calculation follows the numerical comparison in the existing warning; this pilot does not exercise Blender's UI or scene collection.

Across the four accurate controls, the existing 5 px guard warning counts are [0, 0, 0, 0]; across the four biased controls they are [2, 2, 2, 2]. Release trials are comparable in 8/8 cases and pass the independent oracle in 8/8. The measured prior gaps and camera movement quantify model dependence. These cases do not demonstrate additional detection beyond the existing guard. This pilot does not justify a new product warning or automatic reference selection.
