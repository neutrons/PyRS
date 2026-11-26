"""Measured pole figures: failed fits, ignored points, corrected rates, option checks and an analytic correction."""

from dataclasses import replace

import numpy as np
import pytest

import pyrs.pole_figures.core.polefig as polefig
from pyrs.pole_figures.core.gauge import gauge_points
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.polefig import PeakIntensity, PolePeak, measure_peaks
from pyrs.pole_figures.core.prediction import Beams
from pyrs.pole_figures.core.runs import RunData
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from tests.util.pole_figures_helpers import box_triangles, synthetic_pole_figure, synthetic_run_data


@pytest.fixture
def synthetic_run() -> RunData:
    return synthetic_run_data()


@pytest.fixture
def flaky_fit_peak(synthetic_run: RunData, monkeypatch: pytest.MonkeyPatch) -> PeakIntensity:
    """measure_peaks with a fit that does not converge in the second sub-run."""
    calls = iter(range(100))

    def flaky_fit(*args: object, **kwargs: object) -> tuple[float, float]:
        return (np.nan, np.nan) if next(calls) == 1 else (100.0, 10.0)

    monkeypatch.setattr(polefig, "fit_peak", flaky_fit)
    return measure_peaks(synthetic_run, [("110", 90.0)], method="fit")[0]


def test_measure_peaks_failed_fit_gives_nan_rate(flaky_fit_peak: PeakIntensity) -> None:
    # Arrange
    pk = flaky_fit_peak

    # Act
    rate = pk.rate

    # Assert
    assert np.isnan(rate[1])
    assert np.isfinite(rate[[0, 2, 3, 4, 5]]).all()


def test_measure_peaks_failed_fit_is_noted(flaky_fit_peak: PeakIntensity) -> None:
    # Arrange
    pk = flaky_fit_peak

    # Act
    notes = pk.notes

    # Assert
    assert notes == ["the fit failed in 1 of 6 sub-runs (those points are ignored)"]


def test_nan_rate_is_ignored() -> None:
    # Arrange
    base = synthetic_pole_figure()
    rate = np.where(np.arange(len(base.rate)) == 0, np.nan, 1.0)

    # Act
    r = replace(base, rate=rate)

    # Assert
    assert r.ignored[0]  # a NaN rate is never a live point


def test_measure_peaks_unknown_method_raises(synthetic_run: RunData) -> None:
    # Arrange
    run = synthetic_run

    # Act / Assert
    with pytest.raises(ValueError, match="method must be one of window, fit, got 'area'"):
        measure_peaks(run, [("110", 90.0)], method="area")  # type: ignore[arg-type]


# ---------------------------------------------------------------- corrected rates
@pytest.fixture
def corrected_peak() -> PolePeak:
    """Three points with corrections 1, 2.5 and inf (the gauge missed the sample)."""
    base = synthetic_pole_figure()
    keep = slice(0, 3)
    fields = ("chi", "phi", "count_s", "F", "volume_in_sample", "mean_path", "vy_offset", "vy_rms", "h")
    return replace(base, **{f: getattr(base, f)[keep] for f in fields}, rate=np.array([10.0, 20.0, 30.0]),
                   sigma=np.array([1.0, 2.0, 3.0]), correction=np.array([1.0, 2.5, np.inf]))  # fmt: skip


def test_corrected_is_rate_times_correction(corrected_peak: PolePeak) -> None:
    # Arrange
    pk = corrected_peak

    # Act
    corrected = pk.corrected

    # Assert
    np.testing.assert_array_equal(corrected, [10.0, 50.0, np.nan])


def test_corrected_sigma_is_sigma_times_correction(corrected_peak: PolePeak) -> None:
    # Arrange
    pk = corrected_peak

    # Act
    sigma = pk.corrected_sigma

    # Assert
    np.testing.assert_array_equal(sigma, [1.0, 5.0, np.nan])


def test_pole_peak_wrong_pole_shape_raises() -> None:
    # Arrange
    base = synthetic_pole_figure()

    message = rf"h must hold one pole vector per point, shape \({base.n}, 3\), got \(2, 3\)"

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        replace(base, h=base.h[:2])


# ---------------------------------------------------------------- correction with attenuation, analytic
def test_corrections_attenuated_block_matches_analytic_paths() -> None:
    """
    A 1 mm gauge deep in a 100 x 400 x 100 mm block, mu 0.05 /mm, omega 45, 2theta 90.

    At chi 0, phi 0 a point at height z goes in and out through the +vz face: path 2 sqrt(2) (50 - z). At phi 90
    one beam leaves through +vz and the other through -vz, so every point has path 100 sqrt(2). The correction at
    phi 90 is F(0, 0) / F = mean(exp(-mu 2 sqrt(2) (50 - z))) / exp(-100 sqrt(2) mu).
    """
    # Arrange
    mu = 0.05
    box = SampleMesh(box_triangles([-50.0, -200, -50], [50.0, 200, 50]))
    m = Mount(MountConventions(), omega=45, tth=90)
    beams = Beams(incident_width=1.0, incident_height=1.0, diffracted_width=1.0)
    pts, _ = gauge_points(m, incident_width=1.0, incident_height=1.0, diffracted_width=1.0, n_points=2000)
    expected = np.mean(np.exp(-mu * 2 * np.sqrt(2) * (50.0 - pts[:, 2]))) / np.exp(-100 * np.sqrt(2) * mu)

    # Act
    *_, corr = polefig.corrections(m, chi=np.array([0.0, 0.0]), phi=np.array([0.0, 90.0]), beams=beams, mesh=box,
                                   center=[0.0, 0.0, 0.0], mu=mu, n_points=2000)  # fmt: skip

    # Assert
    np.testing.assert_allclose(corr, [1.0, expected], rtol=1e-12)
