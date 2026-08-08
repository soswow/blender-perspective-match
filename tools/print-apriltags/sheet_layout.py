"""Generate AprilTag images and lay them out on A4/A3 print sheets.

Business logic only — the CLI lives in ``print_apriltags.py``.

Layout model
------------
Each tag (+ optional ID label) sits inside its own **margin** on all sides.
**Padding** is extra white *between* those margin boxes.

With ``margin=8`` and ``padding=0``::

    |←8→| TAG |←8→|←8→| TAG |←8→|
          └─── margins abut; nothing else between ───┘

Cut guides follow the outer edge of each margin box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
from PIL import Image, ImageDraw, ImageFont

# ISO paper sizes in millimetres (portrait).
PAPER_SIZES_MM: dict[str, tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
}

# OpenCV dictionary name → cv2.aruco constant name.
DICTIONARY_ALIASES: dict[str, str] = {
    "apriltag-36h11": "DICT_APRILTAG_36h11",
    "apriltag-25h9": "DICT_APRILTAG_25h9",
    "apriltag-16h5": "DICT_APRILTAG_16h5",
    "aruco-4x4-50": "DICT_4X4_50",
    "aruco-5x5-50": "DICT_5X5_50",
}

# Prefer fewer bits when only ~20 IDs are needed (better at distance).
DEFAULT_DICTIONARY = "apriltag-25h9"
DEFAULT_TAG_SIZE_MM = 90.0
DEFAULT_DPI = 300
DEFAULT_MARGIN_MM = 8.0
DEFAULT_PADDING_MM = 0.0
DEFAULT_LABEL_HEIGHT_MM = 8.0
CUT_GUIDE_COLOR = (170, 170, 170)


@dataclass(frozen=True)
class SheetConfig:
    """Physical layout settings for one print run."""

    paper: str = "a4"
    tag_size_mm: float = DEFAULT_TAG_SIZE_MM
    # White border around each tag (+ label) on every side.
    margin_mm: float = DEFAULT_MARGIN_MM
    # Extra space between neighbouring margin boxes (0 = margins abut).
    padding_mm: float = DEFAULT_PADDING_MM
    show_labels: bool = True
    label_height_mm: float = DEFAULT_LABEL_HEIGHT_MM
    show_cut_guides: bool = True
    dpi: int = DEFAULT_DPI
    dictionary: str = DEFAULT_DICTIONARY
    landscape: bool = False

    @property
    def effective_label_height_mm(self) -> float:
        """Label band height used for packing; zero when labels are off."""
        if not self.show_labels:
            return 0.0
        return max(0.0, self.label_height_mm)

    @property
    def content_width_mm(self) -> float:
        """Tag width (labels sit under the tag, same width)."""
        return self.tag_size_mm

    @property
    def content_height_mm(self) -> float:
        """Tag height plus optional label band."""
        return self.tag_size_mm + self.effective_label_height_mm

    @property
    def cell_width_mm(self) -> float:
        """Outer box width: content + margin on left and right."""
        return self.content_width_mm + 2.0 * self.margin_mm

    @property
    def cell_height_mm(self) -> float:
        """Outer box height: content + margin on top and bottom."""
        return self.content_height_mm + 2.0 * self.margin_mm

    @property
    def pitch_x_mm(self) -> float:
        """Horizontal step from one cell origin to the next."""
        return self.cell_width_mm + self.padding_mm

    @property
    def pitch_y_mm(self) -> float:
        """Vertical step from one cell origin to the next."""
        return self.cell_height_mm + self.padding_mm


@dataclass(frozen=True)
class GridLayout:
    """Computed grid that fits on one page."""

    columns: int
    rows: int
    tags_per_page: int
    cell_width_mm: float
    cell_height_mm: float
    page_width_mm: float
    page_height_mm: float
    # Top-left of the first cell; centres the block when the page has leftover space.
    origin_x_mm: float
    origin_y_mm: float


def resolve_dictionary(name: str):
    """Return an OpenCV Aruco dictionary for a friendly alias or DICT_* name."""
    key = name.strip().lower().replace("_", "-")
    const_name = DICTIONARY_ALIASES.get(key, name.strip().upper())
    if not const_name.startswith("DICT_"):
        const_name = f"DICT_{const_name}"
    if not hasattr(cv2.aruco, const_name):
        known = ", ".join(sorted(DICTIONARY_ALIASES))
        raise ValueError(
            f"Unknown dictionary {name!r}. Use one of: {known} "
            f"(or an OpenCV DICT_* name)."
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, const_name))


def mm_to_px(millimetres: float, dpi: int) -> int:
    """Convert millimetres to pixels at the given print DPI."""
    return max(1, int(round(millimetres / 25.4 * dpi)))


def paper_size_mm(paper: str, landscape: bool) -> tuple[float, float]:
    """Return (width_mm, height_mm) for the chosen paper orientation."""
    key = paper.strip().lower()
    if key not in PAPER_SIZES_MM:
        raise ValueError(f"Unsupported paper {paper!r}. Use: a4, a3.")
    width_mm, height_mm = PAPER_SIZES_MM[key]
    if landscape:
        return height_mm, width_mm
    return width_mm, height_mm


def compute_grid(config: SheetConfig) -> GridLayout:
    """Fit as many margin-boxed tag cells as possible on one page."""
    page_width_mm, page_height_mm = paper_size_mm(config.paper, config.landscape)
    cell_w = config.cell_width_mm
    cell_h = config.cell_height_mm
    padding = config.padding_mm

    if cell_w > page_width_mm or cell_h > page_height_mm:
        label_note = " (+ label)" if config.show_labels else ""
        raise ValueError(
            f"Tag {config.tag_size_mm:g} mm{label_note} with "
            f"{config.margin_mm:g} mm margin does not fit on "
            f"{config.paper.upper()}."
        )

    columns = int((page_width_mm + padding) // (cell_w + padding))
    rows = int((page_height_mm + padding) // (cell_h + padding))
    columns = max(1, columns)
    rows = max(1, rows)

    # Centre the packed block on the page.
    block_w = columns * cell_w + max(0, columns - 1) * padding
    block_h = rows * cell_h + max(0, rows - 1) * padding
    origin_x_mm = max(0.0, (page_width_mm - block_w) / 2.0)
    origin_y_mm = max(0.0, (page_height_mm - block_h) / 2.0)

    return GridLayout(
        columns=columns,
        rows=rows,
        tags_per_page=columns * rows,
        cell_width_mm=cell_w,
        cell_height_mm=cell_h,
        page_width_mm=page_width_mm,
        page_height_mm=page_height_mm,
        origin_x_mm=origin_x_mm,
        origin_y_mm=origin_y_mm,
    )


def generate_marker_image(dictionary, marker_id: int, size_px: int) -> Image.Image:
    """Render one AprilTag/ArUco marker as a PIL RGB image."""
    marker_bgr = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    if marker_bgr.ndim == 2:
        rgb = cv2.cvtColor(marker_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(marker_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _load_label_font(height_px: int) -> ImageFont.ImageFont:
    """Pick a readable bitmap/truetype font for the ID label."""
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    size = max(10, int(height_px * 0.7))
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _chunked(values: Sequence[int], size: int) -> Iterable[list[int]]:
    """Yield successive chunks of ``size`` from ``values``."""
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _draw_dotted_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: tuple[int, int, int],
    dash_px: int = 4,
    gap_px: int = 4,
) -> None:
    """Draw a light dashed segment between two points."""
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    step = dash_px + gap_px
    unit_x = dx / length
    unit_y = dy / length
    pos = 0.0
    while pos < length:
        dash_end = min(pos + dash_px, length)
        sx = int(round(x0 + unit_x * pos))
        sy = int(round(y0 + unit_y * pos))
        ex = int(round(x0 + unit_x * dash_end))
        ey = int(round(y0 + unit_y * dash_end))
        draw.line((sx, sy, ex, ey), fill=fill, width=1)
        pos += step


def _normalize_edge(
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Order endpoints so shared edges hash to the same key."""
    if end < start:
        return end, start
    return start, end


def _cell_edges(
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return the four boundary edges of a cut box."""
    return [
        _normalize_edge((left, top), (right, top)),
        _normalize_edge((right, top), (right, bottom)),
        _normalize_edge((right, bottom), (left, bottom)),
        _normalize_edge((left, bottom), (left, top)),
    ]


def _draw_dotted_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int] = CUT_GUIDE_COLOR,
) -> None:
    """Draw a faint dotted cut rectangle (left, top, right, bottom)."""
    left, top, right, bottom = box
    for start, end in _cell_edges(left, top, right, bottom):
        _draw_dotted_line(draw, start, end, fill)


def _draw_cut_guides(
    draw: ImageDraw.ImageDraw,
    boxes: Sequence[tuple[int, int, int, int]],
    share_edges: bool,
) -> None:
    """Draw cut guides for each margin box.

    When ``share_edges`` is true (padding == 0), abutting boxes contribute one
    shared dotted line instead of a double line.
    """
    if not boxes:
        return
    if not share_edges:
        for box in boxes:
            _draw_dotted_rect(draw, box)
        return

    unique_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for left, top, right, bottom in boxes:
        unique_edges.update(_cell_edges(left, top, right, bottom))
    for start, end in unique_edges:
        _draw_dotted_line(draw, start, end, CUT_GUIDE_COLOR)


def render_sheet_page(
    marker_ids: Sequence[int],
    dictionary,
    config: SheetConfig,
    grid: GridLayout,
) -> Image.Image:
    """Compose one page image with tags, optional labels, and cut guides.

    Cut guides are drawn on top of layout — they do not add spacing. They follow
    each margin-box outer edge; with padding 0, shared edges are drawn once.
    """
    page_w = mm_to_px(grid.page_width_mm, config.dpi)
    page_h = mm_to_px(grid.page_height_mm, config.dpi)
    margin_px = mm_to_px(config.margin_mm, config.dpi)
    padding_px = mm_to_px(config.padding_mm, config.dpi)
    tag_px = mm_to_px(config.tag_size_mm, config.dpi)
    label_px = mm_to_px(config.effective_label_height_mm, config.dpi)
    cell_w_px = mm_to_px(grid.cell_width_mm, config.dpi)
    cell_h_px = mm_to_px(grid.cell_height_mm, config.dpi)
    origin_x_px = mm_to_px(grid.origin_x_mm, config.dpi)
    origin_y_px = mm_to_px(grid.origin_y_mm, config.dpi)
    pitch_x_px = cell_w_px + padding_px
    pitch_y_px = cell_h_px + padding_px

    page = Image.new("RGB", (page_w, page_h), "white")
    page.info["dpi"] = (config.dpi, config.dpi)
    draw = ImageDraw.Draw(page)
    font = _load_label_font(label_px) if config.show_labels and label_px > 0 else None
    cut_boxes: list[tuple[int, int, int, int]] = []

    for index, marker_id in enumerate(marker_ids):
        column = index % grid.columns
        row = index // grid.columns
        cell_x = origin_x_px + column * pitch_x_px
        cell_y = origin_y_px + row * pitch_y_px
        tag_x = cell_x + margin_px
        tag_y = cell_y + margin_px

        marker = generate_marker_image(dictionary, marker_id, tag_px)
        page.paste(marker, (tag_x, tag_y))

        if font is not None:
            label = f"ID {marker_id}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_x = tag_x + max(0, (tag_px - text_w) // 2)
            text_y = tag_y + tag_px + max(
                1, (label_px - (text_bbox[3] - text_bbox[1])) // 2
            )
            draw.text((text_x, text_y), label, fill="black", font=font)

        if config.show_cut_guides:
            # Exact outer edge of the margin box (no -1), so padding-0 neighbours share a coord.
            left = cell_x
            top = cell_y
            right = cell_x + cell_w_px
            bottom = cell_y + cell_h_px
            cut_boxes.append((left, top, right, bottom))

    if cut_boxes:
        _draw_cut_guides(draw, cut_boxes, share_edges=(config.padding_mm <= 0))

    return page


def build_print_pages(
    marker_ids: Sequence[int],
    config: SheetConfig,
) -> tuple[list[Image.Image], GridLayout]:
    """Render all pages needed to print ``marker_ids``."""
    if not marker_ids:
        raise ValueError("Need at least one marker ID.")
    if any(marker_id < 0 for marker_id in marker_ids):
        raise ValueError("Marker IDs must be non-negative.")

    dictionary = resolve_dictionary(config.dictionary)
    grid = compute_grid(config)
    pages = [
        render_sheet_page(chunk, dictionary, config, grid)
        for chunk in _chunked(list(marker_ids), grid.tags_per_page)
    ]
    return pages, grid


def save_pdf(pages: Sequence[Image.Image], output_path: Path, dpi: int = DEFAULT_DPI) -> None:
    """Write a multi-page PDF from RGB page images."""
    if not pages:
        raise ValueError("No pages to save.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = pages
    first.save(
        output_path,
        "PDF",
        resolution=dpi,
        save_all=True,
        append_images=list(rest),
    )


def save_individual_markers(
    marker_ids: Sequence[int],
    output_dir: Path,
    config: SheetConfig,
) -> list[Path]:
    """Also dump one PNG per marker (useful for single-tag reprints)."""
    dictionary = resolve_dictionary(config.dictionary)
    tag_px = mm_to_px(config.tag_size_mm, config.dpi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for marker_id in marker_ids:
        path = output_dir / f"{config.dictionary}_{marker_id:03d}.png"
        generate_marker_image(dictionary, marker_id, tag_px).save(path)
        paths.append(path)
    return paths


def parse_id_list(spec: str) -> list[int]:
    """Parse ``0-19`` / ``0,1,5-8`` style ID specs into a unique sorted list."""
    ids: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid ID range {token!r}.")
            ids.update(range(start, end + 1))
        else:
            ids.add(int(token))
    return sorted(ids)
