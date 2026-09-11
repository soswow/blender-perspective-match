# Recovered-camera update investigation

11 September 2026. Synthetic evidence only; no private project was modified.

The final recovered-camera 3D update could damage previously accurate cameras.
The fix evaluates that update separately, preserves the joint solve's constraints,
and retains the previous solution if an established camera's point fit degrades
past the existing acceptance budget. It still allows useful recovered evidence
to improve the reconstruction.

## Evidence and control

The frozen `cases/recovered-ground-conflict.json` has three cameras, ground and
elevated points, and 0.3 px pick noise. Only the third camera's elevated picks
are shifted, by 180 px horizontally. These inputs intentionally conflict with
the rigid-object model; exact recovery of every observation is not promised.
Production naturally recovers that camera against the floor. `recovery.py`
observes the subsequent 3D update, and `--compare-freeze` bypasses just that
stage in a paired control. Independent object checks remain unchanged.

| Implementation | Largest ground height after update | Healthy non-anchor camera's withheld RMS |
| --- | ---: | ---: |
| `76e79b1`, before the intervening plane work | 0.487 scene units | 49.36 px |
| `f117f4c`, with shared-plane support and constraint forwarding | below 0.000001 | 36.89 px |
| Candidate guard / frozen-stage control | 0 | below the original 1.8 px limit |

The intervening plane change already restored ground/mirror/Known 3D/plane
arguments and kept constraint springs outside Huber downweighting. Do not
attribute those changes to this fix. The remaining reproduced defect is the
unconditional adoption of a damaging reconstruction. Passing the priors alone
was insufficient. The contradictory camera still has **2.432 px** withheld RMS
after the fix, above its unchanged 1.8 px limit. It remains an accuracy flag;
the regression requires protection of the healthy camera and hard ground.

## Behavior and integration

Initial and recovered joint adjustment now obtain constraints from one method.
The recovered update also retains the already refined positions of soft Known
3D points and allows them to remain free. Hard, consistent ground stays fixed.
The candidate is isolated from live solve state; cancellation remains the
caller's callback, including a background `threading.Event` method.

Acceptance compares the same existing point observations in each established
camera, including the anchor. It requires their 3D points to remain present and
uses the existing joint-BA budget: the larger of 8 px or the camera's previous
RMS plus 2 px, capped by the ordinary camera acceptance limit. A tighter initial
proposal wrongly blocked the existing test in which recovered evidence should
influence 3D. Reusing the established budget preserves that behavior. This is
a bounded regression guard, not an uncertainty estimate or monotonic accuracy
guarantee. It does not directly protect line-only evidence or compare the full
constrained objective.

A separate positive **stage-only control** explicitly marks a posed camera
recovered. Three soft Known 3D references are biased by 0.03 scene units, with
Known 3D slack 0.08, ground slack 0.02 and an exact Y-plane bucket. The update
is accepted, all three references move closer to independent truth, and camera
checks pass. Plane deviation is below 0.000000004; deliberately omitting plane
arguments only at this stage raises it to 0.00127. This validates integration
with **Is in Plane**, not a claim that the new plane feature had that omission.

The reducer now preserves plane members. Request-parity fixtures include
nonzero plane slack and a Free bucket, alongside locks, mirrors and Known 3D.

## Validation and limits

- Merged full suite: 294 tests, 9 optional skips, passed. The subsequently added
  background-callback regression and both recovery tests also passed together.
- The healthy-camera regression fails on the previous implementation. The
  cancellation regression failed with a naive deep copy before callback sharing.
  The existing recovered-influence regression passes unchanged.
- All 20 product/Diagnose/probe request and result comparisons passed across
  four generated states, now including nonzero plane constraints. Saved input
  checksums stayed unchanged.
- The positive Known 3D/plane case passed generated Blender creation/application
  and fresh-process reopening; the standard Blender smoke passed. This ordinary
  scene replay does not force the recovered stage.
- The conflicting fixture completes Blender collection/application/reopening but
  fails the full accuracy gate because of the remaining contradictory camera.
  It is not counted as a passing Blender accuracy fixture.

The experiment does not settle which conflicting picks are wrong, quantify
real-world confidence, or cover arbitrary camera counts and line-only recovery.
It establishes that a late update need not sacrifice good cameras to inconsistent
new evidence. Next investigate shared-plane accuracy and evidence ownership with
independent geometry checks; keep generic uncertainty and image transforms as
separate questions.
