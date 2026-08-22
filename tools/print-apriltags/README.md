# Print AprilTags

Small helper to generate **AprilTag** sheets you can print on **A4** or **A3**.
Default family is **`DICT_APRILTAG_25h9`** — good detection range, and more than enough
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

Also generate vinyl-cutter SVG outlines:

```sh
.venv/bin/python print_apriltags.py --ids 0-19 --padding-mm 5 --cut-lines-svg
```

Each SVG has the same physical page size and orientation as its corresponding
PDF page. It contains only solid black, stroke-free cut shapes: one rounded
rectangle per tag at the PDF cut-guide position, with a 2 mm corner radius. A single-page
sheet writes `<pdf-name>-cut.svg`; multi-page sheets write numbered
`<pdf-name>-cut-page-001.svg` files. SVG output is independent of
`--cut-guides` / `--no-cut-guides`.

Cut-shape coordinates use the PDF's rasterized grid spacing, so generated SVGs
also align with PDFs printed by versions of this tool from before SVG export was
added.

Use positive `--padding-mm` for vinyl cutting so adjacent rounded outlines do
not share an edge.

## Defaults

| Option | Default |
|--------|---------|
| `--ids` | `0-19` |
| `--paper` | `a4` |
| `--tag-size-mm` | `90` |
| `--dictionary` | `DICT_APRILTAG_25h9` |
| `--margin-mm` | `8` (white around each tag on every side) |
| `--padding-mm` | `0` (extra between margin boxes and around the group; alias `--gap-mm`) |
| `--labels` / `--no-labels` / `--embed-label` | labels below tags |
| `--label-height-mm` | `8` (ignored with `--no-labels` and `--embed-label`) |
| `--cut-guides` / `--no-cut-guides` | faint dotted cut lines on |
| `--cut-lines-svg` | off (write matching vinyl-cutter SVG outline pages) |
| `--dpi` | `300` |

### Margin vs padding

```text
|← padding →|← margin →| TAG |← margin →|← padding →|← margin →| TAG |← margin →|← padding →|
```

- **`--margin-mm 8 --padding-mm 0`** — each tag keeps 8 mm white around it; neighbouring
  margins abut (16 mm white between black squares; cut guides share an edge).
- **`--padding-mm 5`** — adds 5 mm extra between those margin boxes and at least
  5 mm outside the top row, bottom row, left column, and right column. The whole
  padded group remains centred on the page.

```sh
.venv/bin/python print_apriltags.py --ids 0-19 --paper a3 \
  --margin-mm 8 --padding-mm 0 --open
```

Cut without guides:

```sh
.venv/bin/python print_apriltags.py --ids 0-5 --no-cut-guides --open
```

Embed each numeric label in the tag's bottom-right black border instead of
placing it below the tag:

```sh
.venv/bin/python print_apriltags.py --ids 0-19 --embed-label --open
```

Embedded text is sized from the marker's module grid, right-aligned so additional
digits grow to the left, and printed dark gray so it remains black under normal
binary detection while still being readable up close. `--label-height-mm` has no
effect in this mode.

## Dictionaries

Pass an exact, case-sensitive OpenCV predefined dictionary name. Marker IDs start
at zero; for example, `DICT_4X4_50` accepts IDs `0-49`, not ID `50`.

| Dictionary names | Valid IDs |
|---|---:|
| `DICT_4X4_50`, `DICT_5X5_50`, `DICT_6X6_50`, `DICT_7X7_50` | `0-49` |
| `DICT_4X4_100`, `DICT_5X5_100`, `DICT_6X6_100`, `DICT_7X7_100` | `0-99` |
| `DICT_4X4_250`, `DICT_5X5_250`, `DICT_6X6_250`, `DICT_7X7_250` | `0-249` |
| `DICT_4X4_1000`, `DICT_5X5_1000`, `DICT_6X6_1000`, `DICT_7X7_1000` | `0-999` |
| `DICT_ARUCO_ORIGINAL` | `0-1023` |
| `DICT_APRILTAG_16h5` | `0-29` |
| `DICT_APRILTAG_25h9` | `0-34` |
| `DICT_APRILTAG_36h10` | `0-2319` |
| `DICT_APRILTAG_36h11` | `0-586` |
| `DICT_ARUCO_MIP_36h12` | `0-249` |

OpenCV maintains the canonical list in its
[`PredefinedDictionaryType` documentation](https://docs.opencv.org/5.0/main_modules/aruco__dictionary_8hpp.html).

Example:

```sh
.venv/bin/python print_apriltags.py --dictionary DICT_4X4_50 --ids 0-49
```

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
