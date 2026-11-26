"""Peak intensities: the window sum, the Gaussian fit and the peak-quality notes on synthetic spectra."""

import numpy as np
import pytest

from pyrs.pole_figures.core.peaks import PeakCandidate, fit_peak, integrate, peak_quality

X = np.linspace(80, 100, 401)  # 0.05 deg bins
SIGMA = 0.5
FWHM = 2.3548 * SIGMA
AREA_COUNTS = 50.0 / 0.05  # area 50 in deg x counts over 0.05 deg bins


def gauss(x: np.ndarray, area: float = 50.0, sigma: float = SIGMA, bg: float = 5.0, slope: float = 0.0) -> np.ndarray:
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 90) / sigma) ** 2) + bg + slope * (x - 90)


@pytest.fixture
def flat_box_peak() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(80, 90, 501)
    y = 10 + 0 * x
    y[np.abs(x - 85) <= 0.5] += 100.0
    return x, y


def test_integrate_flat_background_net(flat_box_peak: tuple[np.ndarray, np.ndarray]) -> None:
    # Arrange
    x, y = flat_box_peak

    # Act
    net, _ = integrate(x, y, center=85.0, half_width=0.5)

    # Assert
    assert net == pytest.approx(100.0 * np.sum(np.abs(x - 85) <= 0.5), rel=1e-9)


def test_integrate_flat_background_sigma(flat_box_peak: tuple[np.ndarray, np.ndarray]) -> None:
    # Arrange
    x, y = flat_box_peak

    # Act
    _, sig = integrate(x, y, center=85.0, half_width=0.5)

    # Assert
    assert sig > 0


def test_integrate_sloped_background_gives_area() -> None:
    # Arrange
    y = gauss(X, slope=0.8)  # a sloped background is removed too

    # Act
    win, _ = integrate(X, y, center=90.0, half_width=1.5 * FWHM)

    # Assert
    assert win == pytest.approx(AREA_COUNTS, rel=0.01)  # the window cuts the far tails


def test_fit_peak_sloped_background_gives_area() -> None:
    # Arrange
    y = gauss(X, slope=0.8)

    # Act
    fit, _ = fit_peak(X, y, center=90.0, fwhm=FWHM, half_width=1.5 * FWHM)

    # Assert
    assert fit == pytest.approx(AREA_COUNTS, rel=1e-3)


def test_fit_peak_sigma_not_negative() -> None:
    # Arrange
    y = gauss(X, slope=0.8)

    # Act
    _, sigma = fit_peak(X, y, center=90.0, fwhm=FWHM, half_width=1.5 * FWHM)

    # Assert
    assert sigma >= 0


def test_peak_quality_flags_weak_peak() -> None:
    # Arrange
    noise = np.random.default_rng(1).poisson(100.0, X.size).astype(float)

    # Act
    notes = peak_quality(X, noise, center=90.0, fwhm=0.5, hkl="111")

    # Assert
    assert any(n.startswith("weak peak") for n in notes)


def test_peak_quality_strong_peak_has_no_notes() -> None:
    # Arrange
    strong = gauss(X, area=500.0)

    # Act
    notes = peak_quality(X, strong, center=90.0, fwhm=1.18, hkl="111")

    # Assert
    assert notes == []


def test_peak_quality_flags_overlap() -> None:
    # Arrange
    strong = gauss(X, area=500.0)
    others = [PeakCandidate("200", 90.5, True)]

    # Act
    notes = peak_quality(X, strong, center=90.0, fwhm=1.18, hkl="111", others=others)

    # Assert
    assert any("overlaps {200}" in n for n in notes)


def test_peak_quality_flags_narrow_peak() -> None:
    # Arrange
    strong = gauss(X, area=500.0)

    # Act
    notes = peak_quality(X, strong, center=90.0, fwhm=0.05, hkl="111")

    # Assert
    assert any("under 2 bins" in n for n in notes)
