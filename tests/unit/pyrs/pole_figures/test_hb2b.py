"""HB-2B constants: the monochromator wavelengths and the pole-figure 2theta range."""

import pytest

from pyrs.pole_figures.core.hb2b import HB2B_WAVELENGTHS, TWO_THETA_MAX, TWO_THETA_MIN, two_theta_allowed


def test_hb2b_wavelengths_are_monosetting_values() -> None:
    """The nominal wavelengths of PyRS's MonoSetting, typed in (pyrs.core.monosetting)."""
    # Arrange
    expected = {"Si333": 1.452, "Si511": 1.452, "Si422": 1.540, "Si331": 1.731, "Si400": 1.886, "Si311": 2.275,
                "Si220": 2.667}  # fmt: skip

    # Act
    table = dict(HB2B_WAVELENGTHS)

    # Assert
    assert table == expected


def test_hb2b_wavelengths_match_pyrs_monosetting() -> None:
    """The table holds every MonoSetting name (aliases included) with its value. Skipped where PyRS is absent."""
    # Arrange
    monosetting = pytest.importorskip("pyrs.core.monosetting")

    # Act
    expected = {name: float(m.value) for name, m in monosetting.MonoSetting.__members__.items()}

    # Assert
    assert dict(HB2B_WAVELENGTHS) == expected


@pytest.mark.parametrize(("two_theta", "allowed"), [(TWO_THETA_MIN, True), (TWO_THETA_MAX, True), (49.9, False),
                                                    (120.1, False)])  # fmt: skip
def test_two_theta_allowed_range_inclusive(two_theta: float, allowed: bool) -> None:
    # Arrange
    angle = two_theta

    # Act
    ok = two_theta_allowed(angle)

    # Assert
    assert ok is allowed
