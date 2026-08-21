"""Focused tests for printable marker-sheet layout."""

from __future__ import annotations

import unittest

from sheet_layout import SheetConfig, compute_grid


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


if __name__ == "__main__":
    unittest.main()
