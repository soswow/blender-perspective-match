# Independent point-focal production controls

These requests were frozen before the first production numerical call. The
three-view exact and noisy requests come from the prior no-VP corpus. `four-view`
adds one independently projected camera to the exact mixed-focal graph;
`planar` supplies three translated cameras observing only one plane. The
generator's truth is used only after a product decision has been recorded.

`source-before.tar.xz` holds the raw pre-edit files. Each `run-*/source.tar.xz`
holds the exact source bytes used by that run, and the Sync ledger reserves each
complete request before numerical execution. The bundle ledger similarly
records its input and outcome before the oracle assessment. Run 01 failed before
any Sync call because its process deadline conflicted with the budget's alarm.
Runs 03 and 04 retain failed bundle serialization attempts on the weak case;
run 05 records the corrected refusals. These failures remain as evidence.

The initial exploration used ten reserved ordinary Sync calls and ten bundle
attempts across the saved runs, within the declared ceilings of twelve each.
Each Sync call had a 120-second limit and a 600-second cumulative ceiling. Each
bundle fit was limited in code to 100 Jacobian iterations, at most twelve trial
residuals per iteration, and 30 seconds; this is within the declared 300
Jacobian / 30,000 residual cap per fit and 3,600 / 360,000 total. The runner was
invoked with `gtimeout 420`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

Fresh Sync plus the point fit accepted the exact mixed/shared and four-view
graphs, and refused weak translation, pure rotation, and the translated planar
graph as homography-compatible. Noisy mixed/shared fits were accepted with
0.338/0.249 px raw training RMSE. Their stricter synthetic 2% focal / 1 px
withheld-geometry oracle did not pass; the product reports broad local 95%
focal intervals under the explicitly stated pick-noise assumption rather than
claiming exact FOV. The homography graph is a conservative screen under a local
pixel-noise model; its nominal tail bound is not a universal false-positive
guarantee. Later focused tests cover the adjusted homography, Jacobian, rank,
and gauge guards without another exploratory corpus search.
