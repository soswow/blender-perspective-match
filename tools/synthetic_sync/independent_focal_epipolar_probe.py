"""Budgeted raw-pick epipolar leave-one-out probe; no scene truth enters scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from tools.synthetic_sync.budget import ExperimentBudget

FROZEN = ROOT / "tools/synthetic_sync/cases/independent-focal-production"


def _normalize(points):
    center = points.mean(axis=0)
    spread = np.linalg.norm(points - center, axis=1).mean()
    scale = 2**0.5 / spread
    transform = np.array(((scale, 0, -scale * center[0]),
                          (0, scale, -scale * center[1]), (0, 0, 1)))
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return homogeneous @ transform.T, transform


def _fit_fundamental(first, second):
    a, ta = _normalize(first)
    b, tb = _normalize(second)
    design = np.column_stack((b[:, 0] * a[:, 0], b[:, 0] * a[:, 1], b[:, 0],
                              b[:, 1] * a[:, 0], b[:, 1] * a[:, 1], b[:, 1],
                              a[:, 0], a[:, 1], np.ones(len(a))))
    _u, singular, vh = np.linalg.svd(design, full_matrices=True)
    if singular[-2] < singular[0] * 1e-10:
        return None
    f = vh[-1].reshape(3, 3)
    u, s, vh = np.linalg.svd(f)
    s[-1] = 0
    return tb.T @ ((u * s) @ vh) @ ta


def _sampson(f, first, second):
    a = np.column_stack((first, np.ones(len(first))))
    b = np.column_stack((second, np.ones(len(second))))
    fa = a @ f.T
    ftb = b @ f
    numerator = np.sum(b * fa, axis=1)
    denominator = np.sum(fa[:, :2]**2, axis=1) + np.sum(ftb[:, :2]**2, axis=1)
    return numerator**2 / denominator


def score(request):
    by_camera = {}
    for item in request["observations"]:
        by_camera.setdefault(item["match_id"], {})[item["landmark_id"]] = (item["u"], item["v"])
    scores = []
    count = 0
    ids = sorted(by_camera)
    for i, first_id in enumerate(ids):
        for second_id in ids[i+1:]:
            common = sorted(by_camera[first_id].keys() & by_camera[second_id].keys())
            if len(common) < 10:
                continue
            first = np.asarray([by_camera[first_id][key] for key in common])
            second = np.asarray([by_camera[second_id][key] for key in common])
            for leave in range(len(common)):
                kept = np.arange(len(common)) != leave
                f = _fit_fundamental(first[kept], second[kept])
                count += 1
                if f is None:
                    continue
                errors = _sampson(f, first, second)
                scores.append(dict(pair=[first_id, second_id], omitted=common[leave],
                                   withheld_error_px=float(np.sqrt(errors[leave])),
                                   inlier_sse=float(np.sum(errors[kept])),
                                   inlier_max_px=float(np.sqrt(max(errors[kept])))))
    if count > 1000:
        raise RuntimeError("Epipolar fit count exceeded")
    return dict(model_fits=count, scores=scores)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(out / "source.tar.xz", "w:xz") as archive:
        archive.add(Path(__file__), arcname="tools/synthetic_sync/independent_focal_epipolar_probe.py")
    metadata = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    max_model_fits_per_probe=1000, max_seconds_per_probe=10)
    names = ("noisy-mixed-23-spread-none", "noisy-mixed-23-spread-bad_pick",
             "noisy-shared-23-spread-none", "noisy-shared-23-spread-bad_pick",
             "no-vp-mixed-guessedK-23-spread-none",
             "no-vp-weak-baseline-guessedK", "no-vp-pure-rotation-guessedK", "planar")
    old = ROOT / "tools/synthetic_sync/cases/independent-focal-reliability"
    with ExperimentBudget(out / "ledger.jsonl", metadata=metadata, max_calls=8,
                          per_call_seconds=10, wall_seconds=80) as budget:
        for name in names:
            if "-23-" in name:
                source = old / ("run-03" if "bad_pick" in name else "run-02") / "bundle-ledger.jsonl"
                rows = [json.loads(line) for line in source.read_text().splitlines()]
                started = next(row for row in rows if row.get("kind") == "started" and row["label"] == name)
                request = started["request"]["request"]
            else:
                request = json.loads((FROZEN / f"{name}.json").read_text())["request"]
            with budget.attempt(name, request) as attempt:
                result = score(request)
                attempt.complete(result)
            ranked = sorted(result["scores"], key=lambda row: row["inlier_sse"])
            print(name, result["model_fits"], ranked[:2], flush=True)


if __name__ == "__main__":
    main()
