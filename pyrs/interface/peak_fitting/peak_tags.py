"""
Peak tags for the peak-fitting window.

Kept free of Qt so the tag rules can be unit-tested without a display.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pyrs.utilities.NXstress.peak_tag import peak_tag_key

# The label `mplfitplottingwidget` gives each newly drawn peak range.
_DEFAULT_PEAK_LABEL = re.compile(r"Peak\d+")


def peak_tags_from_labels(labels: List[Optional[str]], n_peaks: int) -> List[str]:
    """Peak tags for a fit: each peak's table label, or ``peak<N>`` if the label
    is blank or still the auto-generated ``Peak<N>`` default.

    Keeping ``peak<N>`` for untouched labels preserves the tags of files fitted
    before labels were used -- the strain/stress viewer matches peaks across
    files by tag. An edited label (e.g. ``Fe 211``) is what NXstress needs: it
    parses each tag into a phase name and Miller indices.

    Args:
        labels: Label-column values, one per peak range (may be shorter).
        n_peaks: Number of peak ranges being fitted.

    Returns:
        One tag per peak range.

    Raises:
        ValueError: If two peak ranges would get the same tag, or tags that
            NXstress files under the same peak (``Fe 211`` and ``Fe211``); tags
            are keys both for NXstress and for matching peaks across files.

    Example:
        >>> peak_tags_from_labels(["Fe 211", "Peak1"], 2)
        ['Fe 211', 'peak1']
    """
    tags = []
    for index in range(n_peaks):
        label = str(labels[index]).strip() if index < len(labels) and labels[index] is not None else ""
        tags.append(label if label and not _DEFAULT_PEAK_LABEL.fullmatch(label) else f"peak{index}")

    duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
    if duplicates:
        raise ValueError(f"Each peak needs a distinct label; duplicated: {', '.join(duplicates)}")

    by_key: Dict[Tuple[str, Tuple[int, int, int]], List[str]] = defaultdict(list)
    for tag in tags:
        by_key[peak_tag_key(tag)].append(tag)
    collisions = [
        f"{' and '.join(repr(tag) for tag in same)} (phase '{phase}', hkl {h} {k} {l_})"
        for (phase, (h, k, l_)), same in by_key.items()
        if len(same) > 1
    ]
    if collisions:
        raise ValueError(
            "Each peak needs a distinct label, and these labels name the same peak: " + "; ".join(collisions)
        )
    return tags
