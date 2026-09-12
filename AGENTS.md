# Agent notes

Conventions for humans and coding agents working in this repo.

Prefer coherent commits containing implementation, tests and relevant docs;
combine review corrections and checkpoint notes before integration. An
experiment budget is a review boundary, not completion of the larger task.
See `docs/development.md#commit-size-and-completion`.

## Changelog

User-visible work must land with a bullet under `## [Unreleased]` in `CHANGELOG.md` **in the same commit** as the code (Keep a Changelog: Added / Changed / Fixed / Removed).

Do this when the change affects matching, sync, UI, install, OpenCV extras, or documented behavior. Skip it for refactors, tests, comments, and internal-only edits.

Write one short user-facing line, not a commit subject. Describe the product behavior, not the `.blend` or stills used to reproduce the bug. Do not name user files, match names, landmark names, or geometry that only exists in that debug scene. Do not invent a version heading or bump `blender_manifest.toml` — that happens at release.

```markdown
## [Unreleased]

### Added
- Optional OpenCV: hide Detect VP Lines when the wheel is missing.
```

If `[Unreleased]` has no matching subsection yet, add it. Leave dated `## [x.y.z]` sections untouched.

Do not rewrite a dated changelog bullet. If later work revises that behavior, add a new **Fixed** / **Changed** / **Removed** line. Rewrite an **Unreleased** bullet in place only when it leaked a debug scene (file, match, landmark, or that file’s layout); do not add a second line that still names the scene.

To ship: `./scripts/release.sh 0.3.7` on a clean `main`. That cuts Unreleased, bumps `blender_manifest.toml`, tags `v0.3.7`, and pushes. GitHub Actions builds the four platform zips and creates the GitHub Release.

## Regression tests

When fixing a bug, add a focused automated regression test whenever the failure can be reproduced deterministically and testing it is practical. The test should fail before the fix and pass afterward. If a regression test is not practical, state why in the handoff.

## Docs

If the sidebar workflow, sync rules, or install story changed, update `docs/user-guide.md`, `docs/sync.md`, and/or `README.md` in that same change. Do not leave the changelog as the only record.

## Where code lives

Keep this map accurate when you add a module, move a stage, or change a named constant.

| Area | Path |
| --- | --- |
| VP / single-camera geometry | `core/geometry.py` |
| Landmark-graph sync | `core/sync/` (package; import as `match_perspective.core.sync`) |
| Focal search | `core/lens_refine.py` (supported-point reprojection score, successful-incumbent support retention and pose reuse; refused candidates restart registration while explicit pose locks remain active) |
| Independent FOV from landmarks | `core/focal_bundle.py` (NumPy joint focal/pose/point/line fit; bounded runtime/size, point depth screen, conditional sensitivity); `core/focal_lines.py` (four-coordinate infinite lines, stroke residuals); `core/focal_constraints.py` (point plane and shared mirror plane constraints); `core/focal_line_constraints.py` (point-supported line plane position/direction, line/line or line/world-axis parallelism, `LINE_RELATION_DIRECTION_RESIDUAL_PX` and `LINE_RELATION_DIRECTION_HARD_SINE`). Optional route in `core/lens_refine.py` initializes from points before the joint fit. Line planes need one same-bucket point for X/Y/Z or three non-collinear same-bucket points for Free; all supporting points need two views. |
| Provisional focal startup | `core/focal_startup.py` — bounded linear/planar PnP seeds for missing cameras from a partial point cloud; these are only starting poses for the joint fit, never accepted Sync results. |
| Bound-aware focal steps | `core/focal_optimizer.py` — feasible active-set damped least-squares steps; recalculates free columns when another parameter hits a limit, with cancellation between re-solves. Acceptance remains in `core/focal_bundle.py`. |
| Point-FOV plane and mirror constraints | `core/focal_constraints.py` (fixed-anchor coordinate conversion, axis/Free-plane and object/live-landmark point-mirror springs, normal-only Mirror Slack and appropriate baseline scale freedom (live reference keeps global scale free); geometric rows are not independent image picks) |
| Lens job input and execution | `scene/__init__.py` (`LensRefinePrep.solver_kwargs`, `run_lens_refine`; `collect_lens_refine_inputs` reads without preparation; apply checks numerical inputs, scene identity and camera targets; `PinSyncSnapshot` also restores state after an application error) |
| Sync input collection and Diagnose result ownership | `scene/__init__.py` (`collect_sync_request` reads without preparation; Diagnose checks its captured request hash and scene identity before applying diagnostics) |
| Known-3D pin refine | `core/pin_refine.py` (Iterate Known 3D loop is applied in `scene/__init__.py`) |
| Blender cameras, stills, Solve Sync apply | `scene/__init__.py` |
| Image analysis (AprilTags, VP detect, edge/tag snap) | `detect/` |
| Background Sync job lifecycle | `ui/operators.py` (per-operator result ownership; `reset_sync_background_jobs` retires jobs on file load and unregister) |
| Operators / panel / overlay | `ui/` (`overlay.py` POST_PIXEL; `overlay_gpu.py` unit-batch TRS; `overlay_style.py` plate stroke colors; `match_history.py` last-10 match visits) |
| Self-contained HTML sync diagnostics | `ui/sync_report.py` (Cytoscape.js in `ui/vendor/` for the camera graph) |
| RNA | `properties/__init__.py` |

Do not special-case a user `.blend` (filename, match names, landmark names, or that file’s layout) in solver code, comments, constants, tests, UI copy, changelog, or docs. Reproduce the geometry with generic synthetic fixtures; keep the motivating file in the chat, not in the repo.

### Sync package (`core/sync/`)

`from match_perspective.core import sync` still works. Submodules:

| Module | Role |
| --- | --- |
| `constants.py` | `WORLD_AXIS_DIRECTIONS`, `ACCEPT_RMSE_PX`, `RESECT_MISMATCH_CANDIDATE_LIMIT`, `STRETCHED_PIXEL_RATIO`, `GROUND_PLANE_Z_FRACTION`, `GROUND_SLACK_DEFAULT`, `GROUND_Z_RESIDUAL_PX`, `GROUND_Z_HARD_SLACK`, `KNOWN_3D_SLACK_DEFAULT`, `KNOWN_3D_RESIDUAL_PX`, `MIRROR_SLACK_DEFAULT`, `MIRROR_PLANE_RESIDUAL_PX`, `MIRROR_PAIR_HARD_GAP`, `MIRROR_PAIR_RESIDUAL_PX`, `PLANE_SLACK_DEFAULT`, `PLANE_RESIDUAL_PX`, `PLANE_HARD_SLACK`, `LOG_SCALE_CLIP`, `LINE_FIXED_ANCHOR_MIN`, `LINE_PLANE_MIN_SINE`, `LINE_CONSTRAINT_DIRECTION_TOLERANCE`, `LINE_RECONSTRUCT_TRUNCATE_PX`, `RECOVERED_HUBER_DELTA_PX`, `BA_ACCEPT_RMSE_FLOOR_PX`, `BA_ACCEPT_RMSE_SLACK_PX`, `BA_FREE_LANDMARK_LIMIT`, `SPATIAL_GRID_SIZE`, `SPATIAL_WEIGHT_CLIP`, `RADIAL_WEIGHT_GAIN`, `TRIANGULATION_GN_STEPS`, `TRIANGULATION_ANGLE_WEIGHT_FLOOR`, `TRIANGULATION_PARALLEL_COSINE`, `SYNC_WEIGHT_PROTECT` |
| `types.py` | `SimilarityTransform`, observations, `SyncSolveResult` |
| `request.py` | Complete `SyncSolveRequest`, versioned JSON capture/restore and request fingerprint |
| `projection.py` | Project, rays, triangulate, image-line geometry; analytic infinite-line projection for pinhole cameras, sampled approximation for distortion |
| `pose.py` | Essential / PnP / IPPE / pairwise register; free-ray refinement holds the seed camera-baseline scale gauge; shared-ground seeds from registered location-enabled cameras |
| `ground.py` | Calibrated On Ground plane init (`estimate_anchor_ground_plane`) |
| `lines.py` | Free / Known 3D lines, Is-Parallel-To, compatible independent axis/CAD direction priors, supporting-plane angle diagnostics |
| `mirrors.py` | Point/line Is-Mirror-Of pairs across one scene plane; reflected line fitting within compatible supported planes and independently fixed parallel directions |
| `planes.py` | Is-in-Plane buckets; preserve ground seeds, initialize one-view points, and fit existing lines in independently supported hard planes while retaining compatible axis/CAD directions |
| `ba.py` | Joint BA, residuals, leave-one-out Diagnose |
| `solve.py` | `solve_landmark_sync` stages; rebuild triangulates point-observation IDs separately from line midpoints |

**Solve stages** (in order): seed per-match pose locks from their live root transforms → pairwise register (strongest-pair seed, then easiest-next camera, composed into the Anchor; free 2D point graphs choose the better-spread Anchor connection and compare later direct poses with registered-view bridges; **Fit Only** matches are skipped here) → peel cameras above `ACCEPT_RMSE_PX` (never peel a pose-locked match) → joint BA (pose-only above `BA_FREE_LANDMARK_LIMIT`, then a thaw of free 3D if that helps; locked-match observations remain active but their similarities have no parameters; **Fit Only** 2D pulls pose, not 3D; the Anchor always contributes 3D) → peel again → resect skipped and Fit Only stills against frozen 3D (On Ground / near-Z=0 if off-plane picks disagree; a one-view Is Mirror Of line is mixed like a Known 3D line against the partner's reflected 3D; pose accept is point RMSE so a line overlay cannot skip a still that already fits the cloud) → triangulate landmarks now visible in recovered views that may move 3D and PnP stills that had no cloud support → pose-only BA of recovered cameras (3D and line midpoints frozen; Huber at `RECOVERED_HUBER_DELTA_PX` so a dense inlier cluster cannot ignore isolated landmarks that pin orientation; pose accept remains point RMSE) → propose a constrained 3D update from recovered stills that may move 3D (retain prior geometry if established point fits exceed the per-camera BA acceptance budget; soft Known 3D stays free) → rebuild free 3D lines from cameras allowed to move 3D (a recovered stroke can pin line depth unless This Camera is Fit Only) → report. On Ground landmarks with `ground_slack > 0` are a soft Z spring, not a hard Z=0 pin. Known 3D points with `known_3d_slack > 0` are a soft XYZ spring toward the Empty (pairwise still uses the Empty; linked Empties are not moved). Known 3D that is also On Ground uses `min(ground_slack, known_3d_slack)` for Z so a looser Known 3D leash cannot lift a floor pin. Is Mirror Of pairs are joint-BA plus resect mixed / recovered-camera BA (not pairwise 2D↔2D); one scene Mirror Empty supplies the plane, or `mirror_landmark_id` supplies its live point while the normal remains fixed; `mirror_slack > 0` then thaws the plane along its normal with non-mirror 3D frozen (Empty stays put). Two-sided pairs are projected onto that reflection after triangulation so the springs start small. Is in Plane buckets are joint BA (not pairwise): X/Y/Z share that coordinate within a #1–#10 group; Free fits an unknown plane once four or more members are reconstructed; `plane_slack > 0` is a soft distance spring. At zero Plane Slack, a missing point with one posed location-enabled pick can be seeded from another located axis-bucket member, or three non-collinear Free members; reject grazing/backward intersections and exclude Fit Only. Initial rebuild and post-resection expansion use this route; result `plane_seeded_landmark_ids` identifies the remaining one-view points for diagnostics. Ground / Known 3D / mirror / plane springs skip Huber so a noisy pick cannot drop a user constraint. Landmark Sync Weight multiplies every pick of that landmark (and Pick Confidence); values above `SYNC_WEIGHT_PROTECT` skip outlier auto-downweight. Recovered cameras must not fail the joint RMSE. Copying locked K onto a different aspect uses one scale for fx and fy unless the sizes are an exact portrait/landscape swap (same pixels, axes swapped). Solve Sync sets fy=fx when they already differ by more than `STRETCHED_PIXEL_RATIO`. This Camera (Solve / Lock Pose / Fit Only) is disabled on the Anchor.

**Live mirror reference:** `mirror_landmark_id` is part of Sync request v4 (v1–3 load with None), lens preparation and job fingerprints. `mirror_plane` supplies the normal only when the reference is set. The reference must be a point with two location-enabled picked views or an explicit Known 3D point; it cannot be a mirror-pair member. Joint BA includes reference derivatives for both point and line mirrors. `_SolveState` resolves the current reference and accepted normal offset for rebuild/recovery; losing it refuses the solve. Positive Mirror Slack is relative to the live point; offset state rolls back with rejected geometry. Diagnose excludes the model-defining reference from leave-one-out candidates. Blender stores its landmark ID, never copies the estimated position into a fixed plane. Landmark mode without an orientation object uses the selected world plane.

**When you change sync:** update this map if stages or files moved; put a new threshold in `constants.py` instead of a raw `40.0`; keep function docstrings to a short contract (what / what not), not algorithm history. Tests: `tests/test_sync_pose.py`, `test_sync_ground.py`, `test_sync_ba.py`, `test_sync_lines.py`, `test_sync_projection.py`, `test_sync_mirrors.py`, `test_sync_planes.py`, `test_sync_solve.py` (helpers in `tests/sync_fixtures.py`). Pairwise covering (true camera vs stored K/pose): `tests/edge_pairs.md`, `tests/pair_fixtures.py`, `tests/test_edge_pairs.py`. Joint BA reweights picks so occupied image-grid cells share influence (a central cluster cannot ignore a few edge picks that pin camera distance).

## Debugging tools

- `tools/synthetic_sync/focal_line_constraints.py` — independent varied-stroke axis/Free-plane, line/line and line/world-axis parallel fixtures; integration tests cover withheld camera/line geometry, simultaneous mirrors and contradictory relations. `verify_focal_lines_blender.py --relations` checks native preparation/application and stale plane/parallel edits with one oracle-start bundle (add `--fresh` for registration).
- `tools/debug-sync/compare_focal_optimizers.py` — capture the initialized objective from saved lens inputs/startup and compare bounded SciPy TRF or NumPy active-set steps, retaining endpoints without another Sync or applying results. Source-local capture is guarded and records hashes; this research tool requires SciPy, the extension does not.
- `tools/synthetic_sync/verify_lens_progress.py` — native Blender operator callback/RNA checks for textual independent-FOV activity and correctly scaled ordinary lens-search progress; no numerical solves.
- `tools/synthetic_sync/focal_crop.py` — exact off-center crop of an independent camera oracle; shifted-principal-point recovery and deliberately centered-calibration control, with withheld geometry. This isolates known crop calibration, not unknown-offset estimation.

- `tools/debug-sync/probe_focal_startup.py` — read-only diagnostics from saved lens `.inputs.json` / `.startup.json`: bounded OpenCV PnP/focal and raw-pair checks, or `--joint` to reuse startup for one production bundle without another Sync; optional `--span-percent` changes only the diagnostic trial.

- `tools/debug-sync/probe_lens_inputs.py` — lens eligibility report with named constraints; optional `--fit-seconds` performs a bounded numerical trial and preserves inputs/startup alongside `--out`. Never applies results or saves the source blend.
- `tools/synthetic_sync/focal_lines.py` and `verify_focal_lines_blender.py` — independent varied/reversed line-stroke and mirrored-line fixtures; native preparation, one bundle from explicit oracle point/pose startup (or one real Sync with `--fresh`), apply and stale-stroke checks. No user file or saved blend.

Headless helpers under `tools/` (and `scripts/validate_addon.py`) for investigating a `.blend` without clicking the sidebar. If you build a new dump, probe, or reproduction script while solving a problem, **check it in** and add a bullet here so the next agent can find it.

- `tools/debug-memory/` — process `vmmap` (GPU / `IOAccelerator`) vs headless `bpy.data.images` / numpy / Memory Statistics. See `tools/debug-memory/README.md`.
- `tools/debug-sync/` — sync graph, stored vs *recovered* optical-axis tilt vs world Z, match-Empty scale (`empty_s`), homography vs mixed RMSE, `solve_landmark_sync` on a saved scene (consumes sidebar locks, slack, and This Camera role); `probe_resected.py` splits ground vs off-plane RMSE on a recovered still; `probe_graph.py` dumps overlap, stored-Empty vs pick, per-pose PnP/refine timing, per-observation residuals, inner vs outer RMSE by image radius, and optional leave-one-out timing; `probe_cameras.py` dumps per-match K, D, Blender lens/FOV, and whether the undistorted plate is active; `probe_pin_refine.py` dumps Known 3D pins vs per-axis VP residuals and runs the camera pin polish (orientation rebuilt from VP lines at the current K). See `tools/debug-sync/README.md`.
- `tools/explore-vp-intrinsics/` — VP-line residual vs FOV / principal-point / λ (does not touch landmarks).
- `scripts/validate_addon.py` — Blender smoke test (register, match CRUD, VP solve, origin, import).
- `tools/sync_snapshot.py` — capture the product's prepared Sync input in factory-startup Blender; replay JSON with ordinary Python/NumPy. Includes locks, slack, calibration, confidence and constraints. Probes use the same request; `tools/synthetic_sync/verify_requests.py` verifies their parity on generated state, including automatic origins. See `tools/debug-sync/README.md`.
- `tools/synthetic_sync/` — seeded Sync cases with an independent projection/visibility oracle, withheld object checks, input-order/cache replay, and generated Blender scene/save-reopen validation; `evidence.py` compares added landmarks with paired noise and equal picking cost; `constraints.py` checks information supplied by Known 3D lines and one-sided mirror features, including live removal via Blender's `--drop-constraint` option. See `tools/synthetic_sync/README.md`; preserve exact JSON evidence when investigating a failure.
- `tools/synthetic_sync/roles.py` — Fit Only constraint cases and paired stroke controls; the Blender runner's `--role-case` option checks a live role change and stale-helper cleanup against a second exact case.
- `tools/synthetic_sync/preparation.py` — generated preset/missing/locked-origin controls with unchanged independent truth, prepared-request capture, evaluated Blender camera checks and raw-input save/reopen replay.
- `tools/synthetic_sync/reduce.py` — bounded reduction of named forbidden-geometry failures, or named line direction/offset/plane-distance failures with `--line-accuracy`, against separate old/fixed checkouts; preserves references including plane members, roles and all other independent accuracy checks, with exact candidate inputs, assessments and cold replay logs.
- `tools/synthetic_sync/graphs.py` — five-camera chain/loop, disconnected-link and Fit Only/locked bridge cases with physically visible edge-local picks, ground scale and withheld object/landmark checks.

- `tools/synthetic_sync/recovery.py` — observe ground/Known 3D/mirror/plane gaps across recovered-camera 3D refinement, compare an explicit frozen-stage control, and assess healthy cameras with independent withheld geometry.
- `tools/synthetic_sync/recovery_acceptance.py` — frozen mixed point/line recovery case, stage-freeze and no-line controls; checks the line-ID crash boundary and reports independent geometry separately from intentional contradictory-camera flags. `--expect-crash` recognizes the historical triangulation KeyError; see `tools/synthetic_sync/recovery-acceptance-results.md`.
- `tools/synthetic_sync/accepted_recovery.py` — stage-only positive recovered-update control with paired freeze, before/candidate/applied/final snapshots and independent camera/line/plane checks; checks exact declared parallel direction separately from broad truth-accuracy limits. See `tools/synthetic_sync/accepted-recovery-results.md`.
- `tools/synthetic_sync/line_position_gauge.py` — explicit historical sampled-projection baseline, Jacobian-only and analytic-projection trials on the accepted and mirror/plane cases; captures optimizer calls, separate diagnostic probe time, same-line invariance and independent geometry. See `tools/synthetic_sync/line-position-gauge-results.md`; instrumented timings are not clean benchmarks.
- `tools/synthetic_sync/biased_references.py` — four paired hard/soft Known 3D controls with local reference bias, unchanged truth/picks and an independently fixed world frame; reports fitted evidence, reference drift and withheld alignment without treating deliberate model conflict as a solver defect.
- `tools/synthetic_sync/reference_sensitivity.py` — eight paired prior-release controls with truth-free reference/camera movement and support checks; compares incremental value against the existing stored-anchor warning. See `tools/synthetic_sync/reference-sensitivity-results.md`.

- `tools/synthetic_sync/planes.py` — independent axis/Free-plane and point/line controls, hard-ground intersection, paired plane removal and Fit Only stroke checks; `--contribution` checks single-view depth from a supported plane with removal/role controls; `--mirrored-line` tests an independently supported plane against the frozen weak mirror strokes, retaining identical Known 3D references in its removal control. Blender `--drop-constraint` also supports these plane cases. Line-support diagnostics exclude free lines and plane-seeded points from independent plane evidence, and require actual line membership before counting a plane's normal.

- `tools/synthetic_sync/verify_jobs.py` checks blocking/background lens inputs through generated Blender RNA and the actual operator worker callback, including planes, slack, locks, roles and shared/per-match lens settings. `--ownership` runs real numerical Diagnose jobs after controlled plane/role/pick edits, missing anchors and scene changes. `--lens-ownership` replays one real improved lens result against input/VP/origin/search/camera edits, with unchanged/unrelated-edit/match-switch controls and independent checks of applied cameras. Rejections must be `StaleSyncResult`, preserve current output state and avoid preparation. `--lens-ownership --apply-failures` injects errors during camera writes, Sync apply and plate rebuilding, with exact state/cache restoration and successful/numerical-refusal controls. Window-manager plumbing is substituted and job callbacks are deferred deterministically; it does not test live UI scheduling. Lens-input mode substitutes numerical search too.

- `tools/synthetic_sync/verify_job_reload.py` — generated-file reload with deferred real worker callbacks; checks cancellation, new-job availability, old finish/cancel/timer isolation and successful new application against independent withheld geometry. Numerical results are solved once and replayed from identical inputs; native UI scheduling is not simulated.

- `tools/synthetic_sync/lens_support.py` — traces real focal searches with exact inputs, supported-pick coverage and independent withheld geometry; successful and refused-start controls. See `tools/synthetic_sync/lens-results.md`.
- `tools/synthetic_sync/lens_initialization.py` — compares refused-result warm starts with fresh registration at identical true-focal inputs; saves complete requests and independent camera checks. Supports exact `--case` replay; see `tools/synthetic_sync/lens-initialization-results.md`.
- `tools/synthetic_sync/constraint_interactions.py` — paired plane/mirror/parallel cases and float32 controls with independent direction/reflection checks; `--case ... --result ...` post-checks a saved numerical or Blender result. See `tools/synthetic_sync/constraint-interaction-results.md`.

Parallel agent work uses isolated worktrees and distinct file ownership; see `docs/development.md#parallel-agent-work`.

- `tools/synthetic_sync/budget.py` — POSIX experiment ledger reserves each instrumented numerical call before execution, caps calls and cumulative active time, retains exact inputs/results/failures, and rejects reuse with changed metadata or limits. Python signal deadlines can be delayed in native code; use an outer process timeout for hard limits. It does not count model tokens or uninstrumented solves.
- `tools/synthetic_sync/verify_no_vp_preparation.py` — generated no-ground/no-Known-3D/no-VP startup through Manual FOV, Sync/lens preparation and fresh-process reopening; captures unchanged requests and shared/independent focal eligibility, with numerical Sync explicitly blocked. This checks preparation, not numerical calibration or solved-camera application.
- `tools/synthetic_sync/no_vp_bootstrap.py` — frozen 2D-only free-scale calibration controls, arbitrary private poses including the anchor, and independent focal/withheld geometry checks. The first true-intrinsics control failed and stopped the pilot after one ledgered solve; see `tools/synthetic_sync/no-vp-bootstrap-results.md` before spending more calls.
- `tools/synthetic_sync/no_vp_startup_trace.py` — bounded stage trace for frozen no-VP camera registration, pair scores, triangulation, peeling and recovery; supports an input-order control and records exact results in a separate follow-up ledger.
- `tools/synthetic_sync/unknown_focal.py` — one persistent budget for the frozen guessed-K, weak-baseline and pure-rotation no-VP cases and a five-call run through the actual Same Lens API; records complete trial inputs, source/runtime identity, assessments and selected search output. See `tools/synthetic_sync/unknown-focal-results.md`.
- `tools/synthetic_sync/independent_focal.py` — bounded three-focal, camera-pose and free-point bundle-adjustment prototype seeded by saved Sync results, with one global scale gauge and input-only selection. `independent_focal_sensitivity.py` measures local focal uncertainty under an explicit pixel-noise assumption. Their frozen mixed/shared/weak/rotation evidence and optimizer ledgers are in `tools/synthetic_sync/independent-focal-results.md`.
- `tools/synthetic_sync/independent_focal_noise.py`, `independent_focal_dense.py`, `independent_focal_true_k_control.py`, `independent_focal_sync_probe.py`, and `independent_focal_observability.py` — frozen noisy picks and alternate starts, dense-vs-sparse convergence controls, calibrated-startup isolation, pair-collapse tracing, and raw-pick homography/local-sensitivity diagnostics. See the same focal results report; convergence and fitted parallax do not certify depth. The calibrated noisy regression prevents baseline collapse without promising exact geometry.
- `tools/synthetic_sync/independent_focal_production.py` — frozen production point-FOV controls, per-run source archives, real inner-Sync budgets, and independent focal/camera/withheld checks; includes separate planar and four-view controls.
- `tools/synthetic_sync/independent_focal_reliability.py` and `independent_focal_epipolar_probe.py` — bounded incomplete/noisy/bad-pick/alternative-FOV controls, explicitly separated saved-start isolation and fresh-registration evidence. The pair hint is tentative context for an existing refusal, never a new acceptance gate. See `tools/synthetic_sync/cases/independent-focal-reliability/README.md`.
- `tools/synthetic_sync/verify_point_focal_apply.py` — generated Blender point-FOV integration with controlled numerical results: direct joint application through `scene._apply_sync_solve_result`, no-op refusal, stale settings, rollback and plate ownership.
- `tools/synthetic_sync/verify_point_focal_numerical_blender.py` — real point-FOV fit/apply on generated mixed/shared guessed-K cases and weak/rotation refusals; at most one inner Sync per fresh case. `--prepare-only` and `--reopen` do not solve. Applies the fitted result directly, checks evaluated Blender cameras independently, and only saves generated files.
- `tools/synthetic_sync/focal_constraints.py` — independent landmark-scaffold plane/mirror fixtures, matched removal controls and raw-world geometry checks. Free-scale assessment aligns one positive scale from training points about the fixed anchor; a supplied mirror offset can impose conditional scale without providing independent metric truth.
- `tools/synthetic_sync/focal_constraint_trial.py` — ledgered public point-FOV constraint trials and bundle-only replay from archived initial camera/point state; preserves source archives and independent assessments. See `tools/synthetic_sync/cases/focal-constraints/RESULTS.md`.
- `tools/synthetic_sync/focal_constraint_reliability.py`, `focal_constraint_reliability_trial.py` and `focal_constraint_saved_start.py` — paired weak-view plane/mirror cases with partial overlap and deliberate reference/frame errors; ledgered public fits and saved-start cross-controls distinguish startup failure from infeasible relations. Assessments retain fixed-frame errors alongside one training-point-derived global similarity, never aligned using withheld points.
- `tools/synthetic_sync/focal_live_reference.py` — independent exact-pick mirror scaffold with a separately picked on-plane reference and deliberately unrelated supplied plane position; used by focal regressions and native verification.
- `tools/synthetic_sync/verify_landmark_mirror_blender.py` — generated live-reference Sync/point-FOV preparation, cached-position independence, stable selector identity, orientation-only object use, stale-job refusal, native camera application and read-only reopening. One Sync and optionally one bundle per fresh run; `--prepare-only`/`--reopen` use zero solves.
- `tools/synthetic_sync/verify_focal_constraints_blender.py` — generated point-FOV constraint preparation, numerical fit, stale-constraint rejection, direct application and read-only reopening, including arbitrarily oriented Mirror Empties. Each fresh numerical run permits one initial Sync and one bundle; `--prepare-only` and `--reopen` use zero solves.

## Do not

- Do not save a user-provided `.blend` file in place. Treat it as read-only unless the user explicitly asks for a save or saving is required for the fix. If a save is necessary, write a sibling copy with a modified base name and report its exact path; otherwise do not save a `.blend` at all.
- Commit or push unless the user explicitly asks to (e.g. “commit this”, “create a commit”)
- Commit `*.zip` or `wheels/*.whl`
- `pip install` into Blender’s Python
- Create GitHub releases or tags unless asked (use `./scripts/release.sh` when asked to release)
