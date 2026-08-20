#!/usr/bin/env python3
"""CLI: generate printable AprilTag sheets on A4/A3.

Example:

  python3 print_apriltags.py --ids 0-19 --paper a4 --open
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from sheet_layout import (
    DEFAULT_DICTIONARY,
    DEFAULT_DPI,
    DEFAULT_LABEL_HEIGHT_MM,
    DEFAULT_MARGIN_MM,
    DEFAULT_PADDING_MM,
    DEFAULT_TAG_SIZE_MM,
    OFFICIAL_DICTIONARY_NAMES,
    SheetConfig,
    build_print_pages,
    parse_id_list,
    save_individual_markers,
    save_pdf,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate AprilTag / ArUco marker sheets for A4 or A3 printing. "
            "Default family is DICT_APRILTAG_25h9 "
            "(good range, plenty of IDs for ≤20 tags). "
            "Each tag has its own margin; padding is extra space between margin boxes."
        )
    )
    parser.add_argument(
        "--ids",
        default="0-19",
        help="Marker IDs to print, e.g. '0-19' or '0,2,5-8' (default: 0-19).",
    )
    parser.add_argument(
        "--paper",
        choices=("a4", "a3"),
        default="a4",
        help="Paper size (default: a4).",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="Use landscape orientation.",
    )
    parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=DEFAULT_TAG_SIZE_MM,
        help=f"Printed tag side length in mm (default: {DEFAULT_TAG_SIZE_MM:g}).",
    )
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY,
        help=(
            "Exact, case-sensitive OpenCV predefined dictionary name "
            f"(default: {DEFAULT_DICTIONARY}). Choices: "
            f"{', '.join(OFFICIAL_DICTIONARY_NAMES)}."
        ),
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help=(
            f"White border around each tag (+ label) on every side in mm "
            f"(default: {DEFAULT_MARGIN_MM:g}). Also sets the page-edge inset."
        ),
    )
    parser.add_argument(
        "--padding-mm",
        "--gap-mm",
        type=float,
        default=DEFAULT_PADDING_MM,
        dest="padding_mm",
        help=(
            f"Extra space between neighbouring margin boxes in mm "
            f"(default: {DEFAULT_PADDING_MM:g}; 0 = margins abut). Alias: --gap-mm."
        ),
    )
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument(
        "--labels",
        dest="label_mode",
        action="store_const",
        const="external",
        help="Print a numeric label under each tag (default).",
    )
    labels.add_argument(
        "--no-labels",
        dest="label_mode",
        action="store_const",
        const="none",
        help="Omit labels (packs tags tighter vertically).",
    )
    labels.add_argument(
        "--embed-label",
        dest="label_mode",
        action="store_const",
        const="embedded",
        help=(
            "Put a subtle numeric label in the tag's bottom-right black border; "
            "ignores --label-height-mm."
        ),
    )
    parser.set_defaults(label_mode="external")
    parser.add_argument(
        "--label-height-mm",
        type=float,
        default=DEFAULT_LABEL_HEIGHT_MM,
        help=(
            f"Space under each tag for the numeric label in mm "
            f"(default: {DEFAULT_LABEL_HEIGHT_MM:g}; ignored with "
            "--no-labels and --embed-label)."
        ),
    )
    cut_guides = parser.add_mutually_exclusive_group()
    cut_guides.add_argument(
        "--cut-guides",
        dest="show_cut_guides",
        action="store_true",
        help="Draw faint dotted cut lines on each margin box (default).",
    )
    cut_guides.add_argument(
        "--no-cut-guides",
        dest="show_cut_guides",
        action="store_false",
        help="Omit dotted cut guides.",
    )
    parser.set_defaults(show_cut_guides=True)
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Rasterisation DPI (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF path (default: ./apriltag-sheet-<paper>.pdf).",
    )
    parser.add_argument(
        "--also-png",
        action="store_true",
        help="Also write one PNG per marker next to the PDF.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the PDF after writing (Preview / default viewer).",
    )
    parser.add_argument(
        "--print",
        dest="send_to_printer",
        action="store_true",
        help="Send the PDF to the default printer (macOS/Linux lpr).",
    )
    return parser


def open_pdf(path: Path) -> None:
    """Open a PDF in the OS default viewer."""
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", str(path)], check=False)
    elif system == "Windows":
        subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def send_to_printer(path: Path) -> None:
    """Queue the PDF on the default printer."""
    system = platform.system()
    if system == "Darwin":
        # `open -a` Preview still needs manual Print; lpr is direct.
        subprocess.run(["lpr", str(path)], check=True)
    elif system == "Windows":
        subprocess.run(
            ["powershell", "-Command", f"Start-Process -FilePath '{path}' -Verb Print"],
            check=True,
        )
    else:
        subprocess.run(["lpr", str(path)], check=True)


def main(argv: list[str] | None = None) -> int:
    """Parse args, build sheets, write PDF."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        marker_ids = parse_id_list(args.ids)
    except ValueError as error:
        parser.error(str(error))

    if not marker_ids:
        parser.error("No marker IDs parsed from --ids.")

    if args.margin_mm < 0:
        parser.error("--margin-mm must be >= 0.")
    if args.padding_mm < 0:
        parser.error("--padding-mm must be >= 0.")

    config = SheetConfig(
        paper=args.paper,
        tag_size_mm=args.tag_size_mm,
        margin_mm=args.margin_mm,
        padding_mm=args.padding_mm,
        show_labels=args.label_mode == "external",
        embed_label=args.label_mode == "embedded",
        label_height_mm=args.label_height_mm,
        show_cut_guides=args.show_cut_guides,
        dpi=args.dpi,
        dictionary=args.dictionary,
        landscape=args.landscape,
    )

    try:
        pages, grid = build_print_pages(marker_ids, config)
    except ValueError as error:
        parser.error(str(error))

    output_path = args.out
    if output_path is None:
        orientation = "landscape" if args.landscape else "portrait"
        output_path = Path(
            f"apriltag-sheet-{args.paper}-{orientation}-{config.dictionary}.pdf"
        )

    save_pdf(pages, output_path, dpi=config.dpi)

    if args.also_png:
        png_dir = output_path.with_suffix("").parent / f"{output_path.stem}-png"
        save_individual_markers(marker_ids, png_dir, config)
        print(f"Wrote {len(marker_ids)} PNGs to {png_dir}")

    if config.embed_label:
        labels_note = "embedded labels"
    elif config.show_labels:
        labels_note = "labels on"
    else:
        labels_note = "no labels"
    cuts_note = "cut guides on" if config.show_cut_guides else "no cut guides"
    print(
        f"Wrote {output_path} — {len(marker_ids)} tags "
        f"({config.dictionary}), {len(pages)} page(s), "
        f"{grid.columns}×{grid.rows} per page, "
        f"{config.tag_size_mm:g} mm tags, "
        f"{config.margin_mm:g} mm margin, "
        f"{config.padding_mm:g} mm padding, {labels_note}, {cuts_note}, "
        f"on {args.paper.upper()}"
        f"{' landscape' if args.landscape else ''}."
    )
    print("Print at 100% scale (no fit-to-page) so tag size stays accurate.")

    if args.open:
        open_pdf(output_path)
    if args.send_to_printer:
        send_to_printer(output_path)
        print("Sent to default printer.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
