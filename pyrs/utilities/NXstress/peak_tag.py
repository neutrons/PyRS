"""
pyrs/utilities/NXstress/peak_tag.py

The key NXstress files a peak under, derived from its peak tag.

NXstress records each peak collection by `<phase name>` and Miller indices
`(h, k, l)`, parsed from its tag, so two different tags can name the same peak:
`Fe 211` and `Fe211` both become `("Fe", (2, 1, 1))`. Code outside this package
that must detect such collisions -- before fitting, rather than when the file is
written -- asks for the key here instead of re-implementing the parsing rule.
"""

from ._peaks import _Peaks

#: Miller indices recorded for a tag that names no lattice plane, such as the
#: legacy `peak<N>`.
UNKNOWN_HKL = _Peaks.UNKNOWN_HKL


def peak_tag_key(tag: str) -> tuple[str, tuple[int, int, int]]:
    """The `(phase_name, (h, k, l))` key NXstress files a peak tag under.

    A tag that names no lattice plane -- one that does not parse as
    `<phase> <hkl>`, such as the legacy `peak<N>`, or one that reads as `(000)` --
    is keyed by the whole tag with `UNKNOWN_HKL`, as the writer records it, so it
    collides only with an identical tag.

    Args:
        tag: The peak tag.

    Returns:
        The tag's phase name and Miller indices.

    Example:
        >>> peak_tag_key("Fe 211") == peak_tag_key("Fe211")
        True
    """
    return _Peaks._phase_and_hkl(tag)
