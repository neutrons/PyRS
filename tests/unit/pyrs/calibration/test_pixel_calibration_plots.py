"""Unit tests for pyrs.calibration.pixel_calibration_plots: predicted rings on the detector image (#1039)"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from pyrs.calibration.pixel_calibration_plots import (
    PointMarks,
    PredictedRing,
    marker_at,
    plot_ring_points,
    remove_artists,
    detector_two_theta_map,
    hkl_label,
    plot_detector_counts,
    plot_ring_overlay,
    predicted_rings,
    remove_ring_overlay,
)
from pyrs.calibration.powder_standards import HB2B_STANDARDS, REFLECTIONS
from pyrs.core.detector_model import DetectorModel
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from tests.util.instrument_helpers import default_hb2b_geometry

NI, FE, _MO = HB2B_STANDARDS
SHIFT = DENEXDetectorShift(0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656)
WAVELENGTH = 1.5308
ARM = 85.0


@pytest.fixture(scope="module")
def hb2b():
    return default_hb2b_geometry()


@pytest.fixture(scope="module")
def two_theta_map(hb2b):
    return detector_two_theta_map(hb2b, ARM, None, SHIFT)


def _vertices(contours):
    return np.concatenate([path.vertices for path in contours.get_paths() if len(path.vertices)])


@pytest.mark.parametrize(
    "structure, expected",
    [("FCC", ["111", "200", "220", "311", "222", "400", "331"]), ("BCC", ["110", "200", "211", "220", "310", "222"])],
)
def test_hkl_label_of_every_hb2b_reflection(structure, expected):
    assert [hkl_label(n, structure) for n in REFLECTIONS[structure]] == expected


def test_hkl_label_forbidden_sum_raises():
    with pytest.raises(ValueError, match="no FCC reflection"):
        hkl_label(2, "FCC")  # 110 is not allowed in FCC


def test_predicted_rings_are_bragg_angles_in_range():
    # Act - Ni at the Si422 wavelength, on a detector covering 70-100 deg
    rings = predicted_rings(NI, WAVELENGTH, (70.0, 100.0))

    # Assert - the physical law, unmodified, and only the reflections inside the range
    d = NI.lattice_a / np.sqrt([8, 11, 12])
    assert [ring.label for ring in rings] == ["Ni 220", "Ni 311", "Ni 222"]
    np.testing.assert_array_equal([r.two_theta for r in rings], 2 * np.rad2deg(np.arcsin(WAVELENGTH / (2 * d))))


def test_predicted_rings_skip_unreachable_reflections():
    # Act - at 1.731 A, Ni 331 (2d = 1.617 A) cannot diffract
    rings = predicted_rings(NI, 1.731, (0.0, 180.0))

    # Assert
    assert [ring.label for ring in rings] == ["Ni 111", "Ni 200", "Ni 220", "Ni 311", "Ni 222", "Ni 400"]


def test_two_theta_map_is_reductions_pixel_two_theta(hb2b, two_theta_map):
    """The map is indexed [index_0, index_1], i.e. pixel ID index_0 * n_rows + index_1, as reduction"""
    # Arrange
    instrument = ResidualStressInstrument(hb2b)

    # Act
    instrument.build_instrument(-ARM, None, SHIFT)

    # Assert
    np.testing.assert_array_equal(two_theta_map.ravel(), instrument.get_pixels_2theta(1))


def test_overlay_contour_lies_on_the_ring(hb2b, two_theta_map):
    """A contour vertex at (x, y) is pixel (x - 0.5, y - 0.5); its 2theta there is the Bragg angle"""
    # Arrange
    ring = predicted_rings(NI, WAVELENGTH, (np.min(two_theta_map), np.max(two_theta_map)))[0]
    ax = Figure().add_subplot()

    # Act
    contours = plot_ring_overlay(ax, two_theta_map, [ring], label=False)

    # Assert - checked with the forward model at the fractional pixels, not with the map itself
    vertices = _vertices(contours)
    two_theta = DetectorModel(hb2b).two_theta(vertices[:, 0] - 0.5, vertices[:, 1] - 0.5, -ARM, None, SHIFT)
    assert vertices.shape[0] > 100
    np.testing.assert_allclose(two_theta, ring.two_theta, atol=2e-4)


def test_overlay_drawn_only_on_unmasked_pixels(two_theta_map):
    # Arrange - mask the left half of the detector (index_0 < 512)
    mask = np.ones_like(two_theta_map)
    mask[:512, :] = 0
    rings = predicted_rings(NI, WAVELENGTH, (np.min(two_theta_map), np.max(two_theta_map)))

    # Act
    contours = plot_ring_overlay(Figure().add_subplot(), two_theta_map, rings, mask=mask.ravel(), label=False)

    # Assert
    assert np.all(_vertices(contours)[:, 0] >= 512.0)


def test_counts_and_overlay_share_pixel_placement():
    """plot_detector_counts (like the GUI's DetectorView) puts pixel (i0, i1) at (i0 + 0.5, i1 + 0.5)"""
    # Arrange - one bright pixel
    size, i0, i1 = 64, 10, 50
    counts = np.zeros((size, size))
    counts[i0, i1] = 1.0
    ax = Figure().add_subplot()

    # Act
    image = plot_detector_counts(ax, counts.ravel())

    # Assert - with extent (0, N, 0, N) and origin "upper", array row r is drawn at y = N - r - 0.5
    array = image.get_array()
    row, column = np.argwhere(array == 1.0)[0]
    assert (column + 0.5, size - row - 0.5) == (i0 + 0.5, i1 + 0.5)
    assert tuple(image.get_extent()) == (0.0, size, 0.0, size)


def test_overlay_labels_each_ring(two_theta_map):
    # Arrange
    rings = predicted_rings(NI, WAVELENGTH, (np.min(two_theta_map), np.max(two_theta_map)))
    ax = Figure().add_subplot()

    # Act
    plot_ring_overlay(ax, two_theta_map, rings)

    # Assert
    assert {text.get_text() for text in ax.texts} == {ring.label for ring in rings}


def test_overlay_without_rings_draws_nothing(two_theta_map):
    ax = Figure().add_subplot()
    assert plot_ring_overlay(ax, two_theta_map, []) is None
    assert not ax.collections


def test_overlay_mask_wrong_size_raises(two_theta_map):
    with pytest.raises(ValueError, match="mask has"):
        plot_ring_overlay(Figure().add_subplot(), two_theta_map, [PredictedRing("Ni 220", 76.3)], mask=np.ones(10))


def test_counts_not_square_raises():
    with pytest.raises(ValueError, match="square"):
        plot_detector_counts(Figure().add_subplot(), np.ones(10))


def test_remove_ring_overlay_removes_contours_and_labels(two_theta_map):
    # Arrange
    ax = Figure().add_subplot()
    rings = predicted_rings(NI, WAVELENGTH, (np.min(two_theta_map), np.max(two_theta_map)))
    contours = plot_ring_overlay(ax, two_theta_map, rings)
    assert ax.collections and ax.texts

    # Act
    remove_ring_overlay(contours)
    remove_ring_overlay(None)

    # Assert
    assert not ax.collections and not ax.texts


def _marks(index_0, index_1, prefix):
    return PointMarks(
        np.asarray(index_0, float), np.asarray(index_1, float), tuple(f"{prefix} {i}" for i in range(len(index_0)))
    )


def test_ring_points_drawn_at_their_pixels():
    # Arrange
    ax = Figure().add_subplot()
    accepted, rejected = _marks([100.25, 300.0], [512.0, 700.5], "point"), _marks([50.0], [80.0], "rejected")

    # Act
    artists = plot_ring_points(ax, accepted, rejected)

    # Assert - pixel (i0, i1) at (i0 + 0.5, i1 + 0.5), as the image and the rings; circles get a dark edge
    offsets = [artist.get_offsets() for artist in artists]
    np.testing.assert_array_equal(offsets[0], [[100.75, 512.5], [300.5, 701.0]])
    np.testing.assert_array_equal(offsets[1], offsets[0])
    np.testing.assert_array_equal(offsets[2], [[50.5, 80.5]])


def test_remove_artists_clears_marks():
    ax = Figure().add_subplot()
    remove_artists(plot_ring_points(ax, _marks([1.0], [2.0], "p"), _marks([3.0], [4.0], "r")))
    assert not ax.collections


def test_no_marks_draws_nothing():
    ax = Figure().add_subplot()
    assert plot_ring_points(ax, None, _marks([], [], "r")) == []


def test_marker_at_finds_nearest_within_radius():
    # Arrange
    ax = Figure().add_subplot()
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    accepted, rejected = _marks([100.0, 600.0], [100.0, 600.0], "point"), _marks([400.0], [400.0], "rejected")
    x, y = ax.transData.transform((600.5, 600.5))
    x_rejected, y_rejected = ax.transData.transform((400.5, 400.5))

    # Act / Assert
    assert marker_at(ax, [accepted, rejected], x + 2, y - 2) == "point 1"
    assert marker_at(ax, [accepted, rejected], x_rejected, y_rejected) == "rejected 0"
    assert marker_at(ax, [accepted, rejected], x + 50, y) is None


def test_point_marks_lengths_must_match():
    with pytest.raises(ValueError, match="one entry per position"):
        PointMarks(np.zeros(2), np.zeros(2), ("only one",))
