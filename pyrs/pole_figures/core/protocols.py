"""
What plots and the GUI need from a pole figure, predicted or measured.

``prediction.PlanePrediction`` and ``polefig.PolePeak`` both have these members.
Code that only draws or animates the points takes a :class:`PoleFigureLike`, so it works with either kind without
checking which it got.

Example:
    >>> def n_ignored(q: PoleFigureLike) -> int:
    ...     return int(q.ignored.sum())
"""

from typing import Protocol

from .correction import GaugeCorrections
from .geometry import Mount
from .types import BoolArray, FloatArray


class PoleFigureLike(Protocol):
    """
    Points of one pole figure with their path-length correction.

    Attributes:
        hkl: hkl label.
        mount: Beams and rotation used for the Monte Carlo (and to draw cells).
        chi: Cradle chi per point (deg).
        phi: Phi per point (deg).
        gauge: Gauge Monte Carlo per point.
        correction: F(0, 0) / F per point, inf where F is 0.
        ignored: Points left out (correction above the cutoff, or no usable value).
    """

    @property
    def hkl(self) -> str: ...  # noqa: D102 (documented in the class docstring)

    @property
    def mount(self) -> Mount: ...  # noqa: D102

    @property
    def chi(self) -> FloatArray: ...  # noqa: D102

    @property
    def phi(self) -> FloatArray: ...  # noqa: D102

    @property
    def gauge(self) -> GaugeCorrections: ...  # noqa: D102

    @property
    def correction(self) -> FloatArray: ...  # noqa: D102

    @property
    def ignored(self) -> BoolArray: ...  # noqa: D102

    def poles(self) -> FloatArray:
        """Pole vectors in the sample frame, shape (n, 3)."""
        ...
