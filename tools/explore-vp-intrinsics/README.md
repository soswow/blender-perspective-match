# Explore VP intrinsics space

Throwaway diagnostic for Perspective Match. Given a `.blend` and a match name
(same string as the sidebar dropdown), it scores how VP-line error changes when
you move **horizontal FOV** and **principal-point offset** (optional **division λ**).

It does **not** touch landmarks or sync — only VP lines → locked-focal
`refine_camera` → `vp_line_residual_rms` (same metric the lens-refine prior uses).

## What you get

An **interactive 3D Plotly scatter** in your browser (drag to orbit, scroll to
zoom, hover a point for FOV / PP / λ / error):

| Axis | Meaning |
|------|---------|
| X | PP offset X (px from image center) |
| Y | PP offset Y (px from image center) |
| Z | HFOV (°) |
| Color | VP line RMS (px), or angular residual with `--metric angle` |

Markers: cyan diamond = current match seed, lime × = lowest-error sample.
Sampling is a BFS from the seed: neighbors expand only while error ≤
`--max-error`, so bad regions stop growing.

## Requirements

- Blender 5.1 (for reading the `.blend`)
- A small local venv with `numpy` + `plotly` (created once):

```sh
cd tools/explore-vp-intrinsics
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```sh
cd /path/to/match_perspective

tools/explore-vp-intrinsics/.venv/bin/python \
  tools/explore-vp-intrinsics/explore.py /path/to/scene.blend "MatchRootName" \
  --max-error 18 \
  --pp-radius 200 \
  --fov-span 15 \
  --pp-step 20 \
  --fov-step 1.5
```

That opens the interactive plot. Optional exports:

```sh
  --out /tmp/vp-space.html \   # save interactive HTML
  --csv /tmp/vp-space.csv \
  --keep-json /tmp/match.json
```

## Defaults

Required args: `blend` and `match`. Everything else:

| Parameter | Default |
|-----------|---------|
| `--blender` | `$BLENDER_BIN`, else `/Applications/Blender 5.1.app/Contents/MacOS/blender` |
| `--json` | none (extract from the `.blend`) |
| `--keep-json` | none |
| `--max-error` | `25.0` (px line RMS) |
| `--pp-radius` | `250` px from seed |
| `--fov-span` | `20`° from seed |
| `--pp-step` | `25` px |
| `--fov-step` | `2`° |
| `--lambda` | match’s current λ |
| `--vary-lambda` | off |
| `--lambda-span` | `0.15` (only with `--vary-lambda`) |
| `--lambda-step` | `0.05` |
| `--max-samples` | `4000` |
| `--metric` | `line_rms` |
| `--out` | none |
| `--csv` | none |
| `--no-show` | off (opens browser) |
| `--show-rejected` | off |
| `--slide-axis` | `lambda` (`lambda` / `hfov` / `ppx` / `ppy`) |

## Knobs

- `--max-error` — prune threshold (line RMS px by default). Stops BFS from
  expanding past that error; the plot only colors samples ≤ this value
  (frontier points above it are hidden unless `--show-rejected`).
- `--show-rejected` — also draw gray dots for evaluated samples above `--max-error`.
- `--pp-radius` / `--fov-span` — hard bounds around the seed.
- `--pp-step` / `--fov-step` — grid resolution (smaller = denser / slower).
- `--vary-lambda` — also walk λ. With more than one λ (and other axes), the plot
  gets a **slider** plus a **dropdown** to choose which axis you slide:
  - Slide **λ** → plot X/Y/Z = PP X, PP Y, FOV
  - Slide **FOV** → plot X/Y/Z = PP X, PP Y, λ
  - Slide **PP X** → plot X/Y/Z = λ, PP Y, FOV
  - Slide **PP Y** → plot X/Y/Z = PP X, λ, FOV
- `--slide-axis` — which axis the slider starts on (default `lambda`).
- `--metric angle` — prune/color by `vp_angular_residual_degrees` instead.
- `--json path.json` — skip Blender and reuse a previous `--keep-json` dump.
- `--no-show` — skip the browser window (use with `--out` / `--csv`).
- `--blender` / `BLENDER_BIN` — override Blender path.

## How to read the plot

- A compact cool-colored valley near the seed means FOV/PP are locally consistent
  with your VP lines.
- If the seed sits on a hot shoulder and a cooler pocket is elsewhere, try those
  FOV/PP values in Manual FOV + Manual PP Offset.
- If almost nothing expands under a modest `--max-error`, the lines themselves
  may be inconsistent (mixed axes, short segments, or a λ that belongs elsewhere).
  If the printed seed error is already above `--max-error`, raise the threshold
  (the tool only force-expands one ring around a bad seed).

## Re-run from dumped JSON

```sh
tools/explore-vp-intrinsics/.venv/bin/python \
  tools/explore-vp-intrinsics/explore.py /dev/null "ignored" \
  --json /tmp/match.json \
  --max-error 12
```
