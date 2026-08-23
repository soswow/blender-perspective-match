# Sync accuracy plan

Working list for the next sync improvements. Walk it **in order** unless a later item is blocking a test you want to run now.

This is not a changelog and not a promise that every item ships. Each section is: what happens today, what we would change, UI, risk, how we will know it helped.

Your current scene (imported YAML `K`, ~72 landmarks, boarded veranda as “ground”, high error on correctly picked tags far from the optical axis) is the right test for several of these at once. Peripheral error with good picks is a classic **focal-length / camera-distance** symptom as much as a pairing symptom.

---

## Suggested order

| # | Item | Why this early | Default UI |
| --- | --- | --- | --- |
| 1 | Same-lens Refine Lenses | Cheapest test of “is my YAML `K` slightly wrong?” | Checkbox **on** |
| 2 | Soft On Ground (Z slack) | Veranda boards may not be one plane | Numeric slack, conservative default |
| 3 | Thaw 3D after cameras settle | You have 72 landmarks; polish currently freezes 3D above 40 | Always on (no extra checkbox) |
| 4 | Strongest-edge seed + easiest-next camera | Pairwise growth still prefers the anchor and alphabetical names | Always on |
| 5 | Better N-view triangulation prior | Feeds 3 and 4; small isolated change | Always on |
| 6 | Pose graph (all strong pairs, cycles, several forks) | “Serious reconstruction” direction | Always on once 4 exists |
| 7 | Remember / show which pair seeded each camera | Debugging 4–6 | Status / Diagnose |
| 8 | Optional hard-drop of persistent outliers | You do not want silent drops | Checkbox **off** |
| 9 | Diverse landmark subset + holdout validation | Experimental; do not change the main solve until 1–6 | Diagnose first |

Items 1–3 are **implemented** (Same Lens Refine Lenses, Ground Slack, thaw 3D). Walk remaining items in order unless a later item is blocking a test you want to run now.

---

## 0. What Refine Lenses does today (read before item 1)

It does **not** treat every still as one camera.

1. Each sync-enabled match has its **own** `fx` (with `fy = fx`).
2. **Coordinate descent:** move one still’s focal at a time, rebuild orientation from **vanishing-point lines**, re-run Solve Sync, keep the better cost.
3. **Coupled polish:** jointly wiggle focals for a few pairs that share landmarks, plus a global relative-scale probe.
4. Cost is sync RMSE + a VP-line prior, with hard VP guardrails so a trial cannot wreck the line fit.

Frozen stills (1-point mode, or not enough VP lines to re-orient) **keep their starting `fx`**. They still contribute picks to sync.

**Manual FOV / imported YAML is searchable** — Refine Lenses was added to adjust those focals — **but each still is still allowed a different `fx`**. If you copied the same YAML onto every match, the search can drift them apart.

If a still has **no usable VP lines**, it is frozen and Refine Lenses will not move its lens at all. A YAML-only, ground-landmark shoot with weak or missing VP lines therefore gets little or no help from the current button.

---

## 1. Same-lens Refine Lenses

**Goal:** one shared focal (one physical lens) for every still in the refine set, then see whether sync RMSE drops.

**Proposal**

- Checkbox on the Refine Lenses row: **All images as same lens** (default **on**).
- When on, search **one** `fx` (apply to every unfrozen still; remap if plate sizes differ the same way copy-K already does).
- When off, keep today’s per-still search (mixed cameras / zooms).
- New search path that **does not need VP lines**: leave each still’s orientation as already solved, vary shared `fx`/`fy`, re-run Solve Sync. YAML-only scenes can use this. If VP lines exist, keep the VP prior as a regularizer so the shared focal cannot fly away.
- Report the shared `fx` (and % vs the imported YAML) in the status string so you can judge “my calibration was 3% short” vs “nope, lens is fine.”

**Risk**

- Default-on is wrong if two stills were different zooms. Off recovers today’s behavior.
- A shared wrong `cx, cy` or distortion is **not** in this first pass. If RMSE barely moves, next is principal point / distortion, not more pairwise tricks.

**How we’ll know**

- Shared `fx` stable across a ± search window, and peripheral landmark RMSE falls more than central.
- If RMSE does not move, YAML `K` is probably not the main leak — go to items 2–3.

---

## 2. Soft On Ground (how far tags may leave Z = 0)

**Today:** an On Ground pick in the anchor is raycast onto **exactly Z = 0**. That 3D point is pinned for scale/PnP if multi-view triangulation agrees within `GROUND_PLANE_Z_FRACTION` (15% of the point’s distance from the origin). There is no user control. Boards, gaps, and tags sitting on timber can be “On Ground” in the UI and still fight the solver.

**Proposal**

- A Sync numeric: **Ground slack** (Blender units, same as the scene). Meaning: On Ground landmarks may sit that far off Z = 0.
- In joint polish, On Ground is a **soft** pull toward Z = 0 (spring with that slack), not an infinite pin, unless slack is 0.
- The “does triangulation agree with the floor raycast?” gate should use the same slack, not a hidden 15% of distance.
- Status/Diagnose: list On Ground landmarks whose settled Z exceeds slack (“veranda not flat” vs “bad pick”).

**Suggested default**

- Small but not zero (enough for plank cup / tag thickness). Exact default when we implement; you can open it for a boarded floor and close it for a machined plate.

**Risk**

- Too much slack and scale drifts (camera height vs object size). Too little and we are back to today.
- Do not auto-uncheck On Ground; only report offenders.

**How we’ll know**

- Unchecking On Ground on a few high tags currently moves cameras a lot → slack should absorb that without throwing those tags out of sync.
- Known 3D / a ruler still needed for true metric scale; slack only stops a wavy floor from bending the cameras.

---

## 3. Thaw 3D after cameras settle

**Today:** above **40** free landmarks, joint polish **freezes 3D** and only moves cameras. That stops a few points from eating edge error. With **72** landmarks you always hit this. A bent triangulation from pairwise then cannot fully unbend.

**Proposal** (no checkbox)

1. Pass A — current behavior: cameras move against frozen 3D (or against the thawed set if ≤ 40).
2. Pass B — thaw free landmarks (not Known 3D), cameras still free, shorter iteration cap.
3. Optional Pass C — cameras only, 3D frozen at the Pass B positions (locks the shape).

Keep Known 3D fixed. On Ground uses item 2’s slack in B, not a hard Z = 0.

**Risk**

- Pass B can let a cluster of central tags drag 3D and undo edge leverage. Keep the existing **spatial / radial reweight** in B (edge cells and far-from-center picks keep extra pull).
- If B raises RMSE, revert to A (already implemented pattern: keep the better of two candidates).

**How we’ll know**

- Peripheral RMSE drops after B while central stays similar.
- Probe: inner vs outer RMSE by image radius (already in `tools/debug-sync/`).

---

## 4. Strongest-edge seed + easiest-next camera

Two related pairwise bugs:

- Any still with ≥5 well-spread picks vs the **anchor** is registered **only vs the anchor**, even if another still shares a wider baseline.
- Waiting stills are visited in **alphabetical match-empty name** order. A slightly wrong camera can join early and poison the 3D used to scale the next one.

Changing **which match is the Anchor** in the UI also changes the Blender world (up, origin, compass). That is a **user** choice. Automation should **not** silently swap the Anchor Empty.

**Proposal**

- Keep your Anchor as shared world.
- Build the **registration graph** from the geometrically strongest pair in the whole set (spread, overlap count, baseline / parallax — not name, not “is it the anchor”).
- Grow by **easiest next**: most overlap with already-in cameras, lowest pair RMSE, largest baseline. Never alphabetical.
- Compose that chain into the Anchor so the scene still lives in the Anchor frame.
- Optional later (not required for accuracy): **Suggest Anchor** in Diagnose — “match M has the best connectivity / spread.” Applying it would re-root the world; make that an explicit button.

**Risk**

- A pair with many tags on a plane can look “strong” and still be the two-fold homography fork. Item 6 is the real guard; here we only change **order**, and still keep several candidates per still (today’s 5 px pair-branch slack).

**How we’ll know**

- Same landmarks, different match names, same seed (order independence).
- A still that used to peel (~40 px) stays in when it shares a healthy pair with someone other than the Anchor.

---

## 5. Better N-view triangulation prior

**Today, honestly:** we already shoot **every registered ray** (3–6 views) into one linear least-squares “midpoint of skew rays.” It is not two-view-only. The name “midpoint” undersells that.

What is still weak:

- A grazing / near-parallel ray counts like a strong stereo pair.
- We do not minimize **reprojection** (pixels on the plates), only a 3D ray-distance proxy.
- A cheirality-bad view (point behind a camera) can still pull the point.

**Proposal**

- Weight rays by triangulation angle / baseline (downweight almost-the-same viewpoint).
- After the linear solve, take a few Gauss–Newton steps on **reprojection** in all views that see the point.
- Drop a view that puts the point behind the camera; require ≥2 surviving views.

This runs whenever we build 3D: pairwise growth, peel, resect, and the seed for item 3.

**Risk**

- Small, test-covered change. Worst case a poorly angled tag is ignored until more cameras join (same as needing two views today).

---

## 6. Pose graph (serious-pipeline core)

Item 4 still **commits** to one growth path. Reconstruction pipelines instead:

1. Solve **every** still-pair that has enough well-spread picks (you already compute many of these while bridging; vs-anchor cameras skip the rest).
2. Keep **several discrete forks** per pair (two hemispheres, two plane solutions) — not one RMSE winner.
3. **Rotation averaging** + translation/scale with **cycle checks** (A→B→C→A should close).
4. Drop or downrank edges that break cycles; retry a camera that was about to peel using the next fork.
5. Hand that seed to the same joint polish (items 3 + 5).

**Proposal**

- After item 4, replace greedy composition with this graph. Pairwise stays a seed; polish still uses **all** multi-view picks.
- Do **not** brute-force spanning trees (“Deep sync”). That repeats the slow pair solve and mostly ignores forks that RMSE cannot see.

**Risk**

- Largest item. Needs fixtures for: cycle inconsistency, planar two-fold, underside vs above-ground, name-order independence.
- Ship behind a short internal flag only if the first graph version is slower and not yet better; user-facing should become the normal Solve Sync once tests pass.

---

## 7. Show which pair seeded each camera

**Proposal**

- Diagnose / Solve Sync status: `Match C via B (pair 8px, graph 12px)` plus the discrete fork (relative pose vs homography vs PnP).
- Persist on the match for the last successful sync (alongside “was this match in the last sync?” — already on the TODOs list).

This does not change accuracy. It makes items 4–6 inspectable on the veranda set.

---

## 8. Optional hard-drop of persistent outliers

**Today:** Huber / Cauchy **downweights** bad picks; they never leave the problem. A wrong ID in 6 views still pulls a 3D point, just less.

**Proposal**

- Checkbox: **Drop persistent outliers** (default **off**).
- When on: after polish, picks whose residual stays huge relative to the still’s median are excluded, then solve **once more**.
- Report names/IDs that were dropped. **Use in Sync** stays as you set it; this is a solve-time mask, not a silent uncheck.

**Why default off**

You are right: with a slightly wrong `K` or a wavy floor, peripheral tags look like outliers **because the model is wrong**, not because the pick is. Items 1–3 should run first. If we then drop edge tags, we would hide the FOV/distance error you care about.

---

## 9. Diverse landmark subset + holdout

You asked for an algorithm that keeps a **small, well-spread** set (far from image center, different directions, far from each other), solves on those, and uses the rest as **validation**.

**Today:** polish already **reweights** a 3×3 image grid and boosts radius from the principal point so a central clump cannot ignore edge picks. It does not **drop** landmarks, so 72 points still freeze 3D (item 3) and still cost BA time.

**Proposal — two layers**

1. **Diagnose holdout (first):** solve on all landmarks as now. Also compute RMSE on a reported subset (e.g. every tag with radius above X, or a greedy diverse set). No change to cameras. Tells you whether error is “center fits, edges don’t” (lens/distance) vs uniform (bad graph).
2. **Optional solve subset (later):** greedy coverage per still — max–min distance in normalized image coords, require at least one pick in outer cells when they exist, cap N per still. Solve on the subset; print holdout RMSE on the ignored tags. If holdout is much worse than in-sample, do not trust the subset solve.

**Risk**

- A naive subset that keeps “most tags” but **drops** the few outer ones is the failure mode you already hit. Coverage must **prefer** periphery, not majority vote.
- Do not turn this on as the default Solve Sync. Item 3 + reweight is the accuracy path; subset is speed + a canary.

---

## What we will not do (unless these items fail)

- Brute-force “try every pairing and keep the lowest RMSE.” Expensive, and RMSE often cannot see a flipped camera.
- Silently change the Anchor match to “whatever pair looks best.”
- Auto-uncheck On Ground or Use in Sync.
- Put focal, principal point, and distortion all into one joint BA in the first pass. Shared-`fx` Refine Lenses (item 1) is the controlled experiment; full inner-K BA is a follow-up if 1 helps but plate edges still disagree.

---

## How this maps to your last shoot

| Observation | First items to try |
| --- | --- |
| YAML `K`, not sure it is perfect | **1** (same-lens, including no-VP path) |
| Boarded veranda, tags in plank gaps | **2** |
| 72 landmarks, outer tags worst, picks look correct | **1**, **3**, **5** (not “delete outer tags”) |
| Want less dependence on which still is Anchor / names | **4**, then **6** |
| Fear of dropping good picks | Keep **8** off until 1–3 |
| Want a faster / cleaner landmark set | **9** as Diagnose only until holdout matches in-sample |

---

## Implementation notes (for whoever codes this)

- New thresholds / slack defaults live in `core/sync/constants.py` (or a named RNA prop that the solver reads). No raw `40.0` / `0.15` in a second place.
- Sync stage map in `AGENTS.md` + `.cursor/rules/sync.mdc` when stages move.
- User-visible checkboxes / Ground slack / status text: `CHANGELOG.md` Unreleased + `docs/sync.md` / `docs/user-guide.md` in the same change.
- Tests: freeze/thaw RMSE on a 72-ish synthetic cloud; same-lens vs per-still `fx`; ground slack lets Z move ±slack and not more; registration order independent of match-id strings; triangulation cheirality drop; outlier-drop off by default.
- Pair covering (`tests/edge_pairs.md`) if item 4/6 changes how stored K/pose is consumed.
