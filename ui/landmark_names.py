"""Suggested names when duplicating a landmark."""

from __future__ import annotations

import re
from collections.abc import Iterable

_FLIP_PAIRS = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}
_FLIP_SUFFIX = re.compile(
    r"(?<![A-Za-z])(left|right|top|bottom)$",
    re.IGNORECASE,
)
_NUMBER_SUFFIX = re.compile(r" (\d+)$")


def _match_token_case(token: str, sample: str) -> str:
    """Return ``token`` with the capitalization of ``sample``."""
    if sample.isupper():
        return token.upper()
    if sample[:1].isupper():
        return token.capitalize()
    return token


def _flipped_side_name(name: str) -> str | None:
    """If ``name`` ends with a side token, return it with that token flipped."""
    match = _FLIP_SUFFIX.search(name)
    if not match:
        return None
    token = match.group(1)
    swapped = _FLIP_PAIRS[token.casefold()]
    return name[: match.start(1)] + _match_token_case(swapped, token)


def _incremented_number_name(name: str) -> str | None:
    """If ``name`` ends with a space and digits, return it with that number + 1."""
    match = _NUMBER_SUFFIX.search(name)
    if not match:
        return None
    return name[: match.start()] + f" {int(match.group(1)) + 1}"


def suggested_duplicate_landmark_name(name: str, taken: Iterable[str]) -> str:
    """Prefer a flipped side or next number; otherwise ``'{name} copy'``."""
    taken_names = set(taken)
    for candidate in (_incremented_number_name(name), _flipped_side_name(name)):
        if candidate is not None and candidate not in taken_names:
            return candidate
    return f"{name} copy"
