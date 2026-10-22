"""Unit tests for pyrs.calibration.powder_standards"""

import numpy as np
import pytest

from pyrs.calibration.powder_standards import HB2B_STANDARDS, PowderStandard, standard_for_sy

NI, FE, MO = HB2B_STANDARDS


def test_hb2b_standards_are_2020_reference_values():
    """The standards are the 2020 reference set (d6cd5e04), not FitCalibration's 2023 fitted values"""
    # Arrange
    expected = {"Ni": ("FCC", 3.523799438, 62), "Fe": ("BCC", 2.8663982, 12), "Mo": ("BCC", 3.14719963, -13)}

    # Act
    actual = {s.name: (s.structure, s.lattice_a, s.sy) for s in HB2B_STANDARDS}

    # Assert
    assert actual == expected


def test_d_spacings_ni_fcc():
    # Arrange
    a = NI.lattice_a

    # Act
    d = NI.d_spacings()

    # Assert - 111, 200, 220, 311, 222, 400, 331
    np.testing.assert_allclose(d, a / np.sqrt([3, 4, 8, 11, 12, 16, 19]))


def test_d_spacings_fe_bcc():
    # Act
    d = FE.d_spacings()

    # Assert - 110, 200, 211, 220, 310, 222
    np.testing.assert_allclose(d, FE.lattice_a / np.sqrt([2, 4, 6, 8, 10, 12]))


def test_fcc_includes_400_reflection():
    assert 16 in NI.hkl_squared


def test_two_theta_satisfies_bragg_law():
    # Arrange
    wavelength = 1.54

    # Act
    two_theta = NI.two_theta(wavelength)

    # Assert
    d = NI.d_spacings()[: two_theta.size]
    np.testing.assert_allclose(2 * d * np.sin(np.deg2rad(two_theta) / 2), wavelength)


def test_two_theta_drops_unreachable_reflections():
    # Arrange - 2 * d(331) of Ni is ~1.62 A, so a 1.7 A beam cannot reach it
    wavelength = 1.7

    # Act
    two_theta = NI.two_theta(wavelength)

    # Assert
    assert two_theta.size == 6
    assert np.all(np.isfinite(two_theta))


def test_two_theta_nonpositive_wavelength_raises():
    with pytest.raises(ValueError, match="wavelength"):
        NI.two_theta(0.0)


@pytest.mark.parametrize("sy, expected", [(62.0, NI), (13.5, FE), (-14.9, MO)])
def test_standard_for_sy_within_tolerance(sy, expected):
    assert standard_for_sy(sy) is expected


@pytest.mark.parametrize("sy", [64.0, 0.0, -11.0])
def test_standard_for_sy_no_match_returns_none(sy):
    assert standard_for_sy(sy) is None


def test_standard_for_sy_empty_standards_returns_none():
    assert standard_for_sy(62.0, standards=()) is None


def test_standard_for_sy_picks_closest():
    # Arrange
    near = PowderStandard("A", "FCC", 4.0, 1.0)
    far = PowderStandard("B", "BCC", 3.0, 2.5)

    # Act / Assert
    assert standard_for_sy(1.4, standards=(far, near)) is near


def test_invalid_structure_raises():
    with pytest.raises(ValueError, match="structure"):
        PowderStandard("X", "HCP", 3.0, 0.0)


def test_nonpositive_lattice_raises():
    with pytest.raises(ValueError, match="lattice_a"):
        PowderStandard("X", "FCC", -1.0, 0.0)
