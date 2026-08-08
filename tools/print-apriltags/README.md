# Print AprilTags

Small helper to generate **AprilTag** sheets you can print on **A4** or **A3**.
Default family is **`apriltag-25h9`** — good detection range, and more than enough
IDs when you only need ~20 markers. Tags default to **90×90 mm** with an ID
label under each one.

Not part of the Blender add-on.

## Requirements

```sh
cd tools/print-apriltags
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Generate a printable PDF and open it:

```sh
cd tools/print-apriltags
.venv/bin/python print_apriltags.py --ids 0-19 --paper a4 --open
```

A3, landscape, custom size:

```sh
.venv/bin/python print_apriltags.py \
  --ids 0-11 \
  --paper a3 \
  --landscape \
  --tag-size-mm 90 \
  --open
```

Send straight to the default printer:

```sh
.venv/bin/python print_apriltags.py --ids 0-5 --paper a4 --print
```

Also dump individual PNGs next to the PDF:

```sh
.venv/bin/python print_apriltags.py --ids 0-19 --also-png
```

## Defaults

| Option | Default |
|--------|---------|
| `--ids` | `0-19` |
| `--paper` | `a4` |
| `--tag-size-mm` | `90` |
| `--dictionary` | `apriltag-25h9` |
| `--margin-mm` | `8` (white around each tag on every side) |
| `--padding-mm` | `0` (extra between margin boxes; alias `--gap-mm`) |
| `--labels` / `--no-labels` | labels on |
| `--label-height-mm` | `8` (ignored with `--no-labels`) |
| `--cut-guides` / `--no-cut-guides` | faint dotted cut lines on |
| `--dpi` | `300` |

### Margin vs padding

```text
|← margin →| TAG |← margin →|← padding →|← margin →| TAG |← margin →|
```

- **`--margin-mm 8 --padding-mm 0`** — each tag keeps 8 mm white around it; neighbouring
  margins abut (16 mm white between black squares; cut guides share an edge).
- **`--padding-mm 5`** — adds 5 mm extra between those margin boxes.

```sh
.venv/bin/python print_apriltags.py --ids 0-19 --paper a3 \
  --margin-mm 8 --padding-mm 0 --open
```

Cut without guides:

```sh
.venv/bin/python print_apriltags.py --ids 0-5 --no-cut-guides --open
```

Dictionary aliases: `apriltag-25h9`, `apriltag-36h11`, `apriltag-16h5`,
`aruco-4x4-50`, `aruco-5x5-50` (or any OpenCV `DICT_*` name).

## Print tips

- Print at **100% / actual size** (disable “fit to page”).
- Prefer matte paper; keep tags flat and high-contrast.
- Cut on the faint dotted guides — they sit **on** the margin-box edge (no extra
  space). With `--padding-mm 0`, neighbouring boxes share one cut line.
- Detect later with OpenCV:

```python
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(gray)
```

## Layout

Each cell is `tag (+ label) + margin on all sides`. The grid is centred on the page.

Rough packing for 90 mm tags, 8 mm margin, 0 padding, labels on:

- **A4 portrait** → **1×2** per page (outer cell ≈ 106×114 mm)
- **A3 portrait** → **2×3** per page
