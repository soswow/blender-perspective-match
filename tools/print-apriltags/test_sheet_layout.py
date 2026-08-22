"""Focused tests for printable marker-sheet layout."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from sheet_layout import (
    SheetConfig,
    compute_grid,
    cut_boxes_mm,
    cut_boxes_px,
    px_to_mm,
    save_cut_svgs,
)


class GridPaddingTests(unittest.TestCase):
    def test_padding_wraps_all_four_group_edges(self) -> None:
        config = SheetConfig(
            paper="a4",
            tag_size_mm=20,
            margin_mm=2,
            padding_mm=10,
            show_labels=False,
        )

        grid = compute_grid(config)
        block_width = (
            grid.columns * grid.cell_width_mm
            + (grid.columns - 1) * config.padding_mm
        )
        block_height = (
            grid.rows * grid.cell_height_mm
            + (grid.rows - 1) * config.padding_mm
        )

        self.assertGreaterEqual(grid.origin_x_mm, config.padding_mm)
        self.assertGreaterEqual(grid.origin_y_mm, config.padding_mm)
        self.assertGreaterEqual(
            grid.page_width_mm - grid.origin_x_mm - block_width,
            config.padding_mm,
        )
        self.assertGreaterEqual(
            grid.page_height_mm - grid.origin_y_mm - block_height,
            config.padding_mm,
        )

    def test_padding_that_leaves_no_room_for_one_cell_is_rejected(self) -> None:
        config = SheetConfig(
            paper="a4",
            tag_size_mm=20,
            margin_mm=2,
            padding_mm=94,
            show_labels=False,
        )

        with self.assertRaisesRegex(ValueError, "94 mm padding"):
            compute_grid(config)


class CutSvgTests(unittest.TestCase):
    def test_svg_matches_legacy_pdf_accumulated_raster_pitch(self) -> None:
        config = SheetConfig(
            paper="a4",
            tag_size_mm=7,
            margin_mm=2,
            padding_mm=2,
            show_labels=False,
            embed_label=True,
            dpi=400,
            dictionary="DICT_APRILTAG_36h10",
        )
        grid = compute_grid(config)
        self.assertEqual((grid.columns, grid.rows), (16, 22))

        svg_boxes = cut_boxes_mm(grid.tags_per_page, config, grid)
        raster_boxes = cut_boxes_px(grid.tags_per_page, config, grid)
        svg_group_width = svg_boxes[-1][0] + svg_boxes[-1][2] - svg_boxes[0][0]
        svg_group_height = svg_boxes[-1][1] + svg_boxes[-1][3] - svg_boxes[0][1]
        raster_group_width = (raster_boxes[-1][2] - raster_boxes[0][0]) * 25.4 / 400
        raster_group_height = (raster_boxes[-1][3] - raster_boxes[0][1]) * 25.4 / 400
        # Preserve the pre-SVG PDF layout, whose independently rounded cell and
        # padding widths accumulate across this dense 16 x 22 grid.
        self.assertGreater(abs(raster_group_width - svg_group_width), 0.5)
        self.assertGreater(abs(raster_group_height - svg_group_height), 0.9)

        with TemporaryDirectory() as temp_dir:
            svg_path = save_cut_svgs(
                list(range(grid.tags_per_page)),
                Path(temp_dir) / "tags.pdf",
                config,
                grid,
            )[0]
            rects = (
                ET.parse(svg_path)
                .getroot()
                .findall(".//{http://www.w3.org/2000/svg}rect")
            )
        svg_output_width = (
            float(rects[-1].attrib["x"])
            + float(rects[-1].attrib["width"])
            - float(rects[0].attrib["x"])
        )
        svg_output_height = (
            float(rects[-1].attrib["y"])
            + float(rects[-1].attrib["height"])
            - float(rects[0].attrib["y"])
        )
        self.assertAlmostEqual(svg_output_width, raster_group_width, places=5)
        self.assertAlmostEqual(svg_output_height, raster_group_height, places=5)

    def test_svg_matches_page_size_and_cut_boxes(self) -> None:
        config = SheetConfig(
            paper="a3",
            landscape=True,
            tag_size_mm=20,
            margin_mm=2,
            padding_mm=5,
            show_labels=False,
        )
        grid = compute_grid(config)

        with TemporaryDirectory() as temp_dir:
            paths = save_cut_svgs(
                [0, 1, 2],
                Path(temp_dir) / "tags.pdf",
                config,
                grid,
            )
            self.assertEqual([path.name for path in paths], ["tags-cut.svg"])
            root = ET.parse(paths[0]).getroot()

        self.assertEqual(root.attrib["width"], "420mm")
        self.assertEqual(root.attrib["height"], "297mm")
        self.assertEqual(root.attrib["viewBox"], "0 0 420 297")
        cut_group = root.find(".//{http://www.w3.org/2000/svg}g")
        self.assertIsNotNone(cut_group)
        self.assertEqual(cut_group.attrib["fill"], "#000000")
        self.assertEqual(cut_group.attrib["stroke"], "none")
        self.assertNotIn("stroke-width", cut_group.attrib)
        rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
        self.assertEqual(len(rects), 3)
        first_box_px = cut_boxes_px(3, config, grid)[0]
        left, top, right, bottom = first_box_px
        self.assertAlmostEqual(
            float(rects[0].attrib["x"]), px_to_mm(left, config.dpi), places=6
        )
        self.assertAlmostEqual(
            float(rects[0].attrib["y"]), px_to_mm(top, config.dpi), places=6
        )
        self.assertAlmostEqual(
            float(rects[0].attrib["width"]),
            px_to_mm(right - left, config.dpi),
            places=6,
        )
        self.assertAlmostEqual(
            float(rects[0].attrib["height"]),
            px_to_mm(bottom - top, config.dpi),
            places=6,
        )
        self.assertEqual(rects[0].attrib["rx"], "2")
        self.assertEqual(rects[0].attrib["ry"], "2")

    def test_multi_page_svg_names_and_page_contents(self) -> None:
        config = SheetConfig(
            paper="a4",
            tag_size_mm=90,
            margin_mm=8,
            show_labels=False,
        )
        grid = compute_grid(config)
        marker_ids = list(range(grid.tags_per_page + 1))

        with TemporaryDirectory() as temp_dir:
            paths = save_cut_svgs(
                marker_ids,
                Path(temp_dir) / "tags.pdf",
                config,
                grid,
            )
            names = [path.name for path in paths]
            rect_counts = [
                len(
                    ET.parse(path)
                    .getroot()
                    .findall(".//{http://www.w3.org/2000/svg}rect")
                )
                for path in paths
            ]

        self.assertEqual(names, ["tags-cut-page-001.svg", "tags-cut-page-002.svg"])
        self.assertEqual(rect_counts, [grid.tags_per_page, 1])


if __name__ == "__main__":
    unittest.main()
