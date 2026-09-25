"""
Peak tags for the peak-fitting window.

Kept free of Qt so the tag rules can be unit-tested without a display.
"""

import re
from typing import List, Optional

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
        ValueError: If two peak ranges would get the same tag; tags are keys
            both for NXstress and for matching peaks across files.

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
    return tags
