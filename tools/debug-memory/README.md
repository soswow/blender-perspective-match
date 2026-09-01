# Memory dumps

Headless datablock/Python snapshot, plus a macOS `vmmap` wrapper for a **live GUI** Blender. Does not write the `.blend`.

The process RSS Activity Monitor shows is not Blender's Memory Statistics. On Apple Silicon, leaked GPU buffers show up as `IOAccelerator (graphics)` in `vmmap`, not as `bpy.data.images`.

## Live GUI process (GPU)

```sh
tools/debug-memory/probe_process.sh
# or
tools/debug-memory/probe_process.sh PID
```

Look at **Physical footprint**, `IOAccelerator (graphics)` size **and region count**, vs `MALLOC_SMALL` / DefaultMallocZone. Millions of ~16 KiB GPU regions with a flat image list is a draw-batch leak, not packed plates.

## Datablocks / Python (headless)

```sh
BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender 5.1.app/Contents/MacOS/blender}"

"$BLENDER_BIN" --factory-startup -b --python tools/debug-memory/probe_memory.py -- \
    --blend "/path/to/scene.blend" --addon --out /tmp/pm-memory.txt
```

`--addon` registers this checkout. Omit it to dump the file with no Perspective Match. Background (`-b`) will not exercise viewport GPU batches.
