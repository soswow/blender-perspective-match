#!/usr/bin/env python3
"""Keep a Changelog helpers for local release and GitHub Actions."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

_UNRELEASED = "## [Unreleased]"
_HEADING = re.compile(r"^## \[", re.MULTILINE)


def _changelog_path(repo_root: Path) -> Path:
    return repo_root / "CHANGELOG.md"


def _split_unreleased(text: str) -> tuple[str, str, str]:
    marker = text.find(_UNRELEASED)
    if marker < 0:
        raise SystemExit("CHANGELOG.md has no ## [Unreleased] heading")
    after_heading = marker + len(_UNRELEASED)
    rest = text[after_heading:]
    next_heading = _HEADING.search(rest)
    if next_heading is None:
        body = rest
        remainder = ""
    else:
        body = rest[: next_heading.start()]
        remainder = rest[next_heading.start() :]
    prefix = text[:after_heading]
    return prefix, body, remainder


def _unreleased_items(body: str) -> str:
    return body.strip()


def cmd_check(repo_root: Path) -> None:
    text = _changelog_path(repo_root).read_text(encoding="utf-8")
    _, body, _ = _split_unreleased(text)
    if not _unreleased_items(body):
        raise SystemExit(
            "CHANGELOG.md ## [Unreleased] is empty — add user-facing bullets first"
        )


def cmd_cut(repo_root: Path, version: str, date: str) -> None:
    path = _changelog_path(repo_root)
    text = path.read_text(encoding="utf-8")
    prefix, body, remainder = _split_unreleased(text)
    items = _unreleased_items(body)
    if not items:
        raise SystemExit(
            "CHANGELOG.md ## [Unreleased] is empty — add user-facing bullets first"
        )
    heading = f"## [{version}] - {date}"
    if heading[3:] in text or f"## [{version}]" in text:
        raise SystemExit(f"CHANGELOG.md already has a {version} section")
    rebuilt = f"{prefix}\n\n{heading}\n\n{items}\n\n{remainder.lstrip()}"
    path.write_text(rebuilt, encoding="utf-8")


def cmd_notes(repo_root: Path, version: str) -> None:
    text = _changelog_path(repo_root).read_text(encoding="utf-8")
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise SystemExit(f"CHANGELOG.md has no ## [{version}] section")
    body = match.group(0).strip()
    if not body:
        raise SystemExit(f"CHANGELOG.md section [{version}] is empty")
    sys.stdout.write(body + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-unreleased", help="Fail if Unreleased has no bullets")

    cut = sub.add_parser("cut", help="Move Unreleased bullets into a dated version")
    cut.add_argument("version")
    cut.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Release date (YYYY-MM-DD)",
    )

    notes = sub.add_parser("notes", help="Print the section for a version")
    notes.add_argument("version")

    args = parser.parse_args()
    repo_root = args.repo.resolve()
    if args.command == "check-unreleased":
        cmd_check(repo_root)
    elif args.command == "cut":
        cmd_cut(repo_root, args.version, args.date)
    elif args.command == "notes":
        cmd_notes(repo_root, args.version)
    else:
        raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
