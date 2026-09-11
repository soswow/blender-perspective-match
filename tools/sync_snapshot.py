"""Capture the product's prepared Sync request, or replay it without Blender."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def capture_record(prep, *, source_name=""):
    """Keep preparation notes separate from the checksummed numerical evidence."""
    from tools.synthetic_sync.solver import environment
    record = prep.to_record()
    record["capture"] = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source_name=source_name,
        environment=environment(),
        preparation={name: getattr(prep, name) for name in (
            "ground_frame_note", "auto_origin_notes", "warnings", "skipped_matches", "excluded_landmarks"
        )},
    )
    return record


def write_snapshot(prep, path, *, source_name=""):
    """Write a new JSON snapshot; never overwrite an existing file."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Snapshot output must have a .json extension")
    record = capture_record(prep, source_name=source_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def capture(args):
    import bpy
    # Use the existing factory-startup extension loader, without loading any scene helpers.
    sys.path.insert(0, str(Path(__file__).parent / "debug-sync"))
    from probe_graph import _load_extension
    _load_extension()
    source = args.blend.expanduser().resolve(strict=True)
    bpy.ops.wm.open_mainfile(filepath=str(source))
    from match_perspective import scene
    prep = scene.prepare_diagnose_sync(bpy.context)
    record = write_snapshot(prep, args.out, source_name=source.name)
    print(f"Captured {len(prep.matches)} cameras, {len(prep.observations)} point picks, "
          f"{len(prep.line_observations or [])} strokes, {len(prep.fixed_similarities or {})} pose locks")
    print(f"request_sha256={record['sha256']}")
    print(f"Snapshot: {args.out.resolve()}")
    return 0


def replay(args):
    from tools.synthetic_sync.solver import environment, load_core
    _core, sync = load_core()
    from match_perspective.core.sync.request import SyncSolveRequest, json_values
    from match_perspective.ui import sync_report
    record = json.loads(args.snapshot.read_text(encoding="utf-8"))
    request = SyncSolveRequest.from_record(record)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        result = sync.solve_landmark_sync(**request.solver_kwargs(), use_pose_cache=args.cache)
    except Exception:
        (args.out / "result.json").write_text(json.dumps(dict(
            request_sha256=record["sha256"], environment=environment(), exception=traceback.format_exc(),
            elapsed_s=time.perf_counter()-started,
        ), indent=2)+"\n", encoding="utf-8")
        raise
    output = dict(request_sha256=record["sha256"], environment=environment(), use_pose_cache=args.cache,
                  elapsed_s=time.perf_counter()-started, result=json_values(result),
                  used_matches=json_values(request.matches))
    (args.out / "result.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = sync_report.build_sync_report(
        operation="Snapshot replay", source_name=record.get("capture", {}).get("source_name", args.snapshot.name),
        matches=request.matches, observations=request.observations,
        line_observations=request.line_observations or [], result=result, anchor_id=request.anchor_id,
        known_world=request.known_world or {}, known_lines=request.known_lines or {},
        parallel_pairs=request.parallel_pairs or [], mirror_pairs=request.mirror_pairs or [],
        fixed_match_ids=set(request.fixed_similarities or {}),
    )
    (args.out / "report.html").write_text(sync_report.render_sync_report_html(report), encoding="utf-8")
    print(result.message)
    print(f"request_sha256={record['sha256']}")
    print(f"Report: {(args.out / 'report.html').resolve()}")
    return 0 if result.success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="Run inside factory-startup Blender; source file is never saved")
    capture_parser.add_argument("--blend", type=Path, required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    replay_parser = commands.add_parser("replay", help="Run with ordinary Python and NumPy; no Blender required")
    replay_parser.add_argument("snapshot", type=Path)
    replay_parser.add_argument("--out", type=Path, required=True, help="New output directory")
    replay_parser.add_argument("--cache", action="store_true", help="Enable pose caching within this solve")
    args = parser.parse_args(argv)
    return capture(args) if args.command == "capture" else replay(args)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(arguments))
