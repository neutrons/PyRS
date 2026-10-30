"""
Predicted Debye–Scherrer rings drawn on the HB2B detector image.

A ring's position comes from the physical law, unmodified: the Bragg angle ``2 asin(lambda / 2d)``
of each reflection of the sub-run's powder standard, mapped onto the detector through the pixel
2theta of :class:`pyrs.core.detector_model.DetectorModel`, i.e. the same 2theta reduction uses for
that geometry. Misfits of a calibration therefore show directly against the data.

Typical use, for one sub-run:

1. :func:`detector_two_theta_map` for the geometry to show (nominal, or a calibration);
2. :func:`predicted_rings` for the sub-run's powder and wavelength, within the map's 2theta range;
3. :func:`plot_detector_counts` and :func:`plot_ring_overlay` on one matplotlib axes.

The plotting functions place pixel ``(index_0, index_1)`` at ``(index_0 + 0.5, index_1 + 0.5)``,
which is where the calibration GUI's detector view (``DetectorView.plot_detector_view``) draws it,
so an overlay lines up with that view as well as with :func:`plot_detector_counts`.

Example::

    two_theta = detector_two_theta_map(setup, det_2theta=85.0, l2=None, calibration=shift)
    rings = predicted_rings(standard_for_sy(62.0), 1.5308, (np.nanmin(two_theta), np.nanmax(two_theta)))
    plot_detector_counts(ax, counts)
    plot_ring_overlay(ax, two_theta, rings, mask=default_mask)
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage

from pyrs.calibration.powder_standards import PowderStandard
from pyrs.core.detector_model import DetectorModel
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift

__all__ = [
    "PointMarks",
    "plot_ring_points",
    "remove_artists",
    "marker_at",
    "remove_ring_overlay",
    "PredictedRing",
    "hkl_label",
    "predicted_rings",
    "detector_two_theta_map",
    "plot_detector_counts",
    "plot_ring_overlay",
]


@dataclass(frozen=True)
class PredictedRing:
    """One reflection's expected ring.

    Attributes:
        label: Powder and reflection, e.g. ``"Ni 220"``.
        two_theta: Bragg angle ``2 asin(lambda / 2d)`` (degrees).
    """

    label: str
    two_theta: float


def hkl_label(hkl_squared: int, structure: str) -> str:
    """Miller indices of a cubic reflection from ``h^2 + k^2 + l^2``, e.g. ``"220"``.

    The indices are the largest-first triple with that sum of squares allowed by the structure
    (FCC: all odd or all even; BCC: ``h + k + l`` even); for the HB2B reflections it is unique.

    Raises:
        ValueError: If ``structure`` is not FCC or BCC, or no allowed reflection has that sum.
    """
    if structure not in ("FCC", "BCC"):
        raise ValueError(f"structure must be 'FCC' or 'BCC', got {structure!r}")
    limit = int(np.sqrt(hkl_squared)) + 1
    for h, k, l in product(range(limit, -1, -1), repeat=3):  # noqa: E741 - Miller index
        if not h >= k >= l or h * h + k * k + l * l != hkl_squared:
            continue
        parities = {h % 2, k % 2, l % 2}
        allowed = len(parities) == 1 if structure == "FCC" else (h + k + l) % 2 == 0
        if allowed:
            return f"{h}{k}{l}"
    raise ValueError(f"no {structure} reflection has h^2 + k^2 + l^2 = {hkl_squared}")


def predicted_rings(
    standard: PowderStandard, wavelength: float, two_theta_range: Tuple[float, float]
) -> List[PredictedRing]:
    """The standard's reflections whose Bragg angle lies in ``two_theta_range``, lowest angle first.

    Args:
        standard: Powder standard in the beam.
        wavelength: Wavelength (Angstrom) of the geometry being shown.
        two_theta_range: ``(min, max)`` 2theta (degrees) of the detector, e.g. of
            :func:`detector_two_theta_map`.

    Returns:
        One :class:`PredictedRing` per reflection inside the range.

    Raises:
        ValueError: If ``wavelength`` is not positive.
    """
    low, high = two_theta_range
    d_spacing = standard.d_spacings()
    reachable = wavelength <= 2 * d_spacing
    bragg = standard.two_theta(wavelength)  # the reachable reflections, same order as d_spacing
    rings = []
    for hkl2, two_theta in zip(standard.hkl_squared[reachable], bragg):
        if low <= two_theta <= high:
            rings.append(
                PredictedRing(f"{standard.name} {hkl_label(int(hkl2), standard.structure)}", float(two_theta))
            )
    return rings


def detector_two_theta_map(
    setup: DENEXDetectorGeometry,
    det_2theta: float,
    l2: Optional[float],
    calibration: Optional[DENEXDetectorShift],
) -> np.ndarray:
    """2theta (degrees) of every pixel, indexed ``[index_0, index_1]`` (pixel ID ``index_0 * n_rows + index_1``).

    Args:
        setup: Detector geometry.
        det_2theta: Detector arm 2theta (degrees) as logged by DAS (``get_detector_2theta``).
        l2: Sample-detector distance (m); None for the instrument's arm length.
        calibration: Detector calibration; None for the nominal geometry.
    """
    num_rows, num_columns = setup.detector_size
    index_0, index_1 = np.meshgrid(np.arange(num_columns), np.arange(num_rows), indexing="ij")
    return DetectorModel(setup).two_theta(index_0, index_1, -det_2theta, l2, calibration)


def plot_detector_counts(ax: Axes, counts: np.ndarray, **imshow_kwargs) -> AxesImage:
    """Draw detector counts as the calibration GUI does (pixel ``(i0, i1)`` at ``(i0 + 0.5, i1 + 0.5)``).

    Args:
        ax: Matplotlib axes.
        counts: Counts in pixel-ID order (1D) or as ``[index_0, index_1]`` (2D), square detector.
        imshow_kwargs: Passed to ``imshow`` (e.g. ``norm``, ``cmap``).

    Raises:
        ValueError: If ``counts`` is not a square detector.
    """
    counts = np.asarray(counts)
    size = int(round(np.sqrt(counts.size)))
    if size * size != counts.size:
        raise ValueError(f"{counts.size} counts are not a square detector")
    counts2d = counts.reshape(size, size)
    image = ax.imshow(np.rot90(counts2d), extent=(0.0, size, 0.0, size), interpolation="none", **imshow_kwargs)
    ax.set_aspect("equal")
    return image


def plot_ring_overlay(
    ax: Axes,
    two_theta_map: np.ndarray,
    rings: Sequence[PredictedRing],
    mask: Optional[np.ndarray] = None,
    colors: str = "red",
    label: bool = True,
) -> Optional[QuadContourSet]:
    """Draw each predicted ring as the contour ``pixel 2theta = Bragg angle`` of ``two_theta_map``.

    Args:
        ax: Matplotlib axes, e.g. one showing :func:`plot_detector_counts` or the GUI's detector view.
        two_theta_map: Pixel 2theta, ``[index_0, index_1]``, from :func:`detector_two_theta_map`.
        rings: Rings to draw, from :func:`predicted_rings`.
        mask: Detector mask in pixel-ID order or ``[index_0, index_1]`` (1 = use); rings are drawn
            only across unmasked pixels.
        colors: Contour colour.
        label: Write each ring's label on it.

    Returns:
        The contour set, or None if there is no ring to draw.

    Raises:
        ValueError: If ``mask`` does not have one value per pixel.
    """
    if not rings:
        return None
    two_theta = np.array(two_theta_map, dtype=float)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.size != two_theta.size:
            raise ValueError(f"mask has {mask.size} values for {two_theta.size} pixels")
        two_theta[mask.reshape(two_theta.shape) == 0] = np.nan
    num_columns, num_rows = two_theta.shape
    levels = sorted({ring.two_theta for ring in rings})
    contours = ax.contour(
        np.arange(num_columns) + 0.5,  # x: index_0
        np.arange(num_rows) + 0.5,  # y: index_1
        two_theta.T,
        levels=levels,
        colors=colors,
        linewidths=1.0,
    )
    if label:
        names: Dict[float, str] = {}
        for ring in rings:
            names[ring.two_theta] = (
                f"{names[ring.two_theta]} / {ring.label}" if ring.two_theta in names else ring.label
            )
        # one label per ring, at the middle of its longest segment (matplotlib's own choice can land
        # on the detector edge, where the label is clipped)
        positions = []
        for segments in contours.allsegs:
            segments = [segment for segment in segments if len(segment)]
            if segments:
                longest = max(segments, key=len)
                positions.append(tuple(longest[len(longest) // 2]))
        ax.clabel(contours, fmt=names, fontsize=8, manual=positions)
    return contours


def remove_ring_overlay(contours: Optional[QuadContourSet]) -> None:
    """Remove an overlay drawn by :func:`plot_ring_overlay`, labels included (None: nothing to do)."""
    if contours is not None:
        contours.remove()  # also removes the clabel texts (matplotlib >= 3.8)


@dataclass(frozen=True)
class PointMarks:
    """Detector positions to mark, with a description of each (e.g. for a tooltip).

    Attributes:
        index_0: Fractional pixel index across columns.
        index_1: Fractional pixel index along rows.
        text: One description per position.
    """

    index_0: np.ndarray
    index_1: np.ndarray
    text: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not (len(self.index_0) == len(self.index_1) == len(self.text)):
            raise ValueError("index_0, index_1 and text must have one entry per position")


def plot_ring_points(
    ax: Axes, accepted: Optional[PointMarks] = None, rejected: Optional[PointMarks] = None
) -> List[Artist]:
    """Mark picked ring points (open circles) and rejected strips (small crosses) on the detector view.

    A position ``(index_0, index_1)`` is drawn at ``(index_0 + 0.5, index_1 + 0.5)``, as the detector
    image and the ring overlay.

    Returns:
        The artists drawn, for :func:`remove_artists`.
    """
    artists: List[Artist] = []
    if accepted is not None and len(accepted.text):
        x, y = np.asarray(accepted.index_0) + 0.5, np.asarray(accepted.index_1) + 0.5
        artists.append(
            ax.scatter(
                x,
                y,
                s=36,
                facecolors="none",
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )
        )
        artists.append(  # a dark ring around each circle keeps it visible on any colour map
            ax.scatter(
                x,
                y,
                s=64,
                facecolors="none",
                edgecolors="black",
                linewidths=0.6,
                zorder=2.9,
            )
        )
    if rejected is not None and len(rejected.text):
        artists.append(
            ax.scatter(
                np.asarray(rejected.index_0) + 0.5,
                np.asarray(rejected.index_1) + 0.5,
                s=30,
                marker="x",
                color="lightgrey",
                linewidths=1.2,
                zorder=3,
            )
        )
    return artists


def remove_artists(artists: Sequence[Artist]) -> None:
    """Remove artists drawn by :func:`plot_ring_points`."""
    for artist in artists:
        artist.remove()


def marker_at(
    ax: Axes, marks: Sequence[PointMarks], x_display: float, y_display: float, radius: float = 6.0
) -> Optional[str]:
    """Description of the mark nearest to a display position (e.g. the mouse), within ``radius`` pixels.

    Args:
        ax: Axes the marks are drawn on (its current data-to-display transform is used).
        marks: Marked positions, e.g. accepted and rejected.
        x_display: Display x of the position (matplotlib event ``x``).
        y_display: Display y (event ``y``).
        radius: Largest distance, in screen pixels.

    Returns:
        The nearest mark's text, or None if none is within ``radius``.
    """
    best, best_distance = None, radius
    for mark in marks:
        if not len(mark.text):
            continue
        points = np.column_stack([np.asarray(mark.index_0) + 0.5, np.asarray(mark.index_1) + 0.5])
        display = ax.transData.transform(points)
        distance = np.hypot(display[:, 0] - x_display, display[:, 1] - y_display)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= best_distance:
            best, best_distance = mark.text[nearest], float(distance[nearest])
    return best
