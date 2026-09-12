"""Ledgered public-path run of frozen weak and preserved tilted-mirror cases.

Use an outer 720-second process timeout. The shared ledger caps all attempts,
including failures and retries, at 11 inner Sync and 18 bundle calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import focal_constraint_trial as trial
from . import focal_constraint_reliability as weak
from . import focal_constraints as original


CASES = weak.CASE_NAMES + ("mirror-tilted-hard",)


def _validate(case):
    if case["name"] == "mirror-tilted-hard":
        original.validate(case)
    else:
        weak.validate(case)


def _assess(case, fitted):
    if case["name"] == "mirror-tilted-hard":
        return original.assess(case, fitted)
    return weak.assess(case, fitted)


def configure():
    trial.CASE_NAMES = CASES
    trial.CASE_ROOT = weak.ROOT
    trial.validate = _validate
    trial.assess = _assess
    trial.SOURCE_PATHS = tuple(sorted(set(trial.SOURCE_PATHS + (
        Path("tools/synthetic_sync/focal_constraint_reliability.py"),
        Path("tools/synthetic_sync/focal_constraint_reliability_trial.py"),
        Path("tools/synthetic_sync/geometry.py"),
    ))))
    trial.SYNC_LIMITS = dict(max_calls=11, per_call_seconds=120, wall_seconds=600)
    trial.BUNDLE_LIMITS = dict(max_calls=18, per_call_seconds=30, wall_seconds=540)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--replay-sync-from", type=Path)
    options = parser.parse_args()
    configure()
    if options.replay_sync_from:
        trial.replay_bundles(options.out, tuple(options.cases), options.replay_sync_from)
    else:
        trial.run(options.out, tuple(options.cases),
                  stop_on_positive_failure=not options.continue_on_failure)
