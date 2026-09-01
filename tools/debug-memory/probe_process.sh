#!/bin/sh
# Snapshot macOS heaps for a live Blender PID (GPU included).
# Usage: tools/debug-memory/probe_process.sh [pid]
set -e
PID="${1:-}"
if [ -z "$PID" ]; then
  PID="$(pgrep -n -f '/Applications/Blender .*/MacOS/Blender$' || true)"
fi
if [ -z "$PID" ]; then
  echo "No Blender GUI pid found" >&2
  exit 1
fi
echo "pid=$PID"
ps -o pid,rss,vsz,%mem,etime,command -p "$PID"
echo
echo "=== vmmap summary (GPU / malloc) ==="
vmmap -summary "$PID" | awk '
  BEGIN { keep=1 }
  /^MALLOC ZONE/ { keep=1 }
  keep && ($0 ~ /IOAccelerator|IOSurface|MALLOC_|Physical footprint|TOTAL|REGION TYPE|DefaultMalloc/) { print }
'
