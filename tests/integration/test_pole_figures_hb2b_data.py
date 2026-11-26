"""
The full reduced reference runs (HB2B_6494-6497.h5) against PF's per-run CSVs.

The HB2B files are not in git: set HIDRA_PF_TEST_DATA, or the tests skip. Every sub-run is checked to the decimals PF
wrote: chi, phi, count time, raw rate and sigma.
"""

import json
from typing import Any

import numpy as np
import pytest

from pyrs.pole_figures.core.geometry import MountConventions
from pyrs.pole_figures.core.peaks import PeakIntensity, measure_peaks
from pyrs.pole_figures.core.phase import presets
from pyrs.pole_figures.core.polefig import cradle_angles
from pyrs.pole_figures.core.runs import RunData, read_run
from tests.util.pole_figures_helpers import REFERENCE, TEST_DATA, needs_data, read_csv_rows

pytestmark = [pytest.mark.integration, needs_data]

RUNS = {"6494": "222", "6495": "311", "6496": "220", "6497": "200"}
#: run number, the run, its measured peaks and PF's CSV rows for it
Measured = tuple[str, RunData, list[PeakIntensity], list[dict[str, str]]]


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(r[name]) for r in rows])


@pytest.fixture(scope="module")
def fitted() -> dict[str, Any]:
    return json.loads((REFERENCE / "fitted_peaks.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_conventions() -> MountConventions:
    """The reference mount (pole_figure_config.json), with its chi offset."""
    cfg = json.loads((REFERENCE / "pole_figure_config.json").read_text(encoding="utf-8"))
    return MountConventions.from_dict(cfg["conventions"])


@pytest.fixture(scope="module", params=list(RUNS))
def measured(request: pytest.FixtureRequest) -> Measured:
    """One run read, its IN718 peaks measured, and PF's CSV for it."""
    run = request.param
    d = read_run(TEST_DATA / f"HB2B_{run}.h5")
    peaks = measure_peaks(d, [(r.label, r.two_theta) for r in presets()["IN718 (fcc)"].reflections(d.wavelength_A)])
    return run, d, peaks, read_csv_rows(REFERENCE / f"pole_figure_{run}_{RUNS[run]}.csv")


def test_read_run_counts_subruns(measured: Measured) -> None:
    # Arrange
    run, d, _, _ = measured

    # Act
    got = (d.run, d.n_subruns, d.short_skipped, d.live)

    # Assert
    assert got == (run, 649, 1, False)


def test_read_run_beam_and_wavelength(
    measured: Measured,
) -> None:
    # Arrange
    _, d, _, _ = measured

    # Act
    beam = (d.incident_width, d.incident_height)

    # Assert
    assert beam == (4.0, 4.0)
    assert d.wavelength_A == pytest.approx(1.7213, abs=1e-4)


def test_measure_peaks_finds_arm_peak(
    measured: Measured,
) -> None:
    # Arrange
    run, _, peaks, _ = measured

    # Act
    hkls = [p.hkl for p in peaks]

    # Assert
    assert hkls == [RUNS[run]]


@pytest.mark.parametrize("axis", ["chi", "phi"])
def test_cradle_angles_match_pf(measured: Measured, ref_conventions: MountConventions, axis: str) -> None:
    # Arrange
    _, d, _, rows = measured
    conv = ref_conventions

    # Act
    chi, phi = cradle_angles(d, conv)

    # Assert
    np.testing.assert_array_equal(chi if axis == "chi" else phi, col(rows, axis))


def test_count_times_match_pf(measured: Measured) -> None:
    # Arrange
    _, d, _, rows = measured

    # Act
    count_s = d.count_s

    # Assert
    np.testing.assert_allclose(count_s, col(rows, "count_s"), atol=5.01e-3)


@pytest.mark.parametrize(("attr", "column"), [("rate", "raw_net_per_s"), ("sigma", "raw_sigma_per_s")])
def test_raw_intensities_match_pf(measured: Measured,
                                  attr: str, column: str) -> None:  # fmt: skip
    # Arrange
    _, _, peaks, rows = measured

    # Act
    got = getattr(peaks[0], attr)

    # Assert
    np.testing.assert_allclose(got, col(rows, column), atol=5.01e-5)


def test_fitted_peaks_fixture_matches_runs(measured: Measured,
                                           fitted: dict[str, Any]) -> None:  # fmt: skip
    """fitted_peaks.json (used by the parity tests that need no HB2B files) is what the reduced files give."""
    # Arrange
    run, d, peaks, _ = measured
    f = fitted[run]

    # Act
    got = (peaks[0].hkl, d.omega_median, d.wavelength_A)

    # Assert
    assert got == (f["hkl"], f["omega"], f["wavelength_A"])


def test_fitted_peak_shape_matches_fixture(measured: Measured,
                                           fitted: dict[str, Any]) -> None:  # fmt: skip
    # Arrange
    run, _, peaks, _ = measured
    f = fitted[run]

    # Act
    tth, fwhm = peaks[0].two_theta, peaks[0].fwhm

    # Assert
    assert tth == pytest.approx(f["two_theta"], abs=1e-9)
    assert fwhm == pytest.approx(f["fwhm"], abs=1e-9)
