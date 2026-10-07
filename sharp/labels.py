"""Section-label vocabulary.

The Harmonix annotators used 126 distinct strings ("chorus", "chorus2",
"verse1a", "inst", "gtr", "transition2a", ...). We collapse them into ten
functional classes. Labels with no functional meaning ("section", "end", ...)
map to ``None`` and are ignored by the label loss and the label metrics; their
boundaries are still used.
"""

from __future__ import annotations

import re
from typing import Optional

CLASSES: tuple[str, ...] = (
    "silence",
    "intro",
    "verse",
    "prechorus",
    "chorus",
    "postchorus",
    "bridge",
    "instrumental",   # solos, instrumental breaks, riffs
    "break",          # breakdowns, transitions, quiet/slow interludes
    "outro",
)
N_CLASSES = len(CLASSES)
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
IGNORE_INDEX = -100

DISPLAY_NAMES = {
    "prechorus": "pre-chorus",
    "postchorus": "post-chorus",
    "instrumental": "solo/inst",
}

# Colour per class for structure plots (colour-blind-friendly-ish, intro..outro)
CLASS_COLORS = {
    "silence": "#d9d9d9",
    "intro": "#8da0cb",
    "verse": "#66c2a5",
    "prechorus": "#a6d854",
    "chorus": "#fc8d62",
    "postchorus": "#e5c494",
    "bridge": "#e78ac3",
    "instrumental": "#ffd92f",
    "break": "#b3b3b3",
    "outro": "#7570b3",
}
UNKNOWN_COLOR = "#f0f0f0"

# normalised raw label (digits stripped) -> class
_BASE_MAP: dict[str, Optional[str]] = {
    "silence": "silence",
    # intro
    "intro": "intro", "fadein": "intro", "opening": "intro", "instintro": "intro",
    "rhythmlessintro": "intro", "intropt": "intro", "introverse": "intro", "introchorus": "intro",
    # verse
    "verse": "verse", "miniverse": "verse", "versepart": "verse", "slowverse": "verse",
    "verse_slow": "verse", "preverse": "verse", "postverse": "verse", "raps": "verse", "rap": "verse",
    # pre-chorus
    "prechorus": "prechorus", "build": "prechorus", "buildup": "prechorus",
    # chorus (instrumental choruses keep the chorus material, so they stay "chorus")
    "chorus": "chorus", "altchorus": "chorus", "quietchorus": "chorus", "refrain": "chorus",
    "chorushalf": "chorus", "choruspart": "chorus", "instchorus": "chorus",
    "chorusinst": "chorus", "chorus_instrumental": "chorus", "intchorus": "chorus",
    # post-chorus
    "postchorus": "postchorus",
    # bridge
    "bridge": "bridge", "instbridge": "bridge",
    # instrumental / solo
    "inst": "instrumental", "instrumental": "instrumental", "solo": "instrumental",
    "guitarsolo": "instrumental", "gtr": "instrumental", "guitar": "instrumental",
    "gtrbreak": "instrumental", "mainriff": "instrumental", "oddriff": "instrumental",
    "synth": "instrumental", "saxobeat": "instrumental", "instrumentalverse": "instrumental",
    "verseinst": "instrumental", "drumroll": "instrumental",
    # break / transition
    "break": "break", "bre": "break", "breakdown": "break", "stutter": "break",
    "transition": "break", "quiet": "break", "slow": "break",
    # outro
    "outro": "outro", "outroa": "outro", "bigoutro": "outro", "vocaloutro": "outro",
    # no functional meaning -> ignored for labels
    "section": None, "worstthingever": None, "fast": None, "end": None,
}

_PREFIX_RULES = (
    ("prechorus", "prechorus"),
    ("postchorus", "postchorus"),
    ("chorus", "chorus"),
    ("verse", "verse"),
    ("intro", "intro"),
    ("outro", "outro"),
    ("bridge", "bridge"),
    ("solo", "instrumental"),
    ("inst", "instrumental"),
    ("break", "break"),
    ("transition", "break"),
    ("silence", "silence"),
)


def normalize_label(raw: str) -> str:
    """'Verse 2' -> 'verse', 'pre-chorus3' -> 'prechorus', 'verse1a' -> 'verse'."""
    s = str(raw).strip().lower().replace("-", "").replace(" ", "")
    return re.sub(r"\d+[a-z]?$", "", s)


def to_class(raw: str) -> Optional[str]:
    """Map a raw Harmonix label to one of CLASSES, or None if it has no functional meaning."""
    base = normalize_label(raw)
    if base in CLASS_TO_INDEX:
        return base
    if base in _BASE_MAP:
        return _BASE_MAP[base]
    for prefix, cls in _PREFIX_RULES:
        if base.startswith(prefix):
            return cls
    return None


def to_index(raw: str) -> int:
    cls = to_class(raw)
    return CLASS_TO_INDEX[cls] if cls is not None else IGNORE_INDEX


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def color_for(label: str) -> str:
    return CLASS_COLORS.get(label, UNKNOWN_COLOR)
