"""
A real reduced HB2B run, small enough for git, against PF's per-run CSV.

HB2B_6494_subset.h5 holds 13 sub-runs of run 6494 (identifying text logs replaced). PF's CSV has one row per
sub-run, in order, so sub-run k is row k - 1. The window is placed with the run's peak center and FWHM from
fitted_peaks.json, because ``measure_peaks`` finds them on the sum of all 649 sub-runs. The full-run tests in
test_pole_figures_hb2b_data.py check that step when the full files are present.
"""

import json
from typing import Any

import numpy as np
import pytest

from pyrs.pole_figures.core.geometry import MountConventions
from pyrs.pole_figures.core.materials import presets
from pyrs.pole_figures.core.peaks import WINDOW_FWHM, integrate
from pyrs.pole_figures.core.polefig import cradle_angles, measure_peaks
from pyrs.pole_figures.core.runs import RunData, read_run
from tests.util.pole_figures_helpers import HB2B_SUBSET, REFERENCE, read_csv_rows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def run() -> RunData:
    return read_run(HB2B_SUBSET)


@pytest.fixture(scope="module")
def pf_rows(run: RunData) -> list[dict[str, str]]:
    """PF's CSV rows for the sub-runs in the subset."""
    rows = read_csv_rows(REFERENCE / "pole_figure_6494_222.csv")
    return [rows[int(k) - 1] for k in run.subruns]


@pytest.fixture(scope="module")
def peak() -> dict[str, Any]:
    return json.loads((REFERENCE / "fitted_peaks.json").read_text(encoding="utf-8"))["6494"]


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(r[name]) for r in rows])


def test_read_run_subset_keeps_every_subrun(run: RunData) -> None:
    # Arrange
    expected = list(range(1, 650, 54))

    # Act
    got = (run.run, run.n, run.short_skipped, run.live, [int(k) for k in run.subruns])

    # Assert
    assert got == ("6494", 13, 0, False, expected)


def test_read_run_subset_beam_and_wavelength(run: RunData, peak: dict[str, Any]) -> None:
    # Arrange
    expected_beam = (4.0, 4.0)

    # Act
    beam = (run.incident_width, run.incident_height)

    # Assert
    assert beam == expected_beam
    assert run.wavelength_A == peak["wavelength_A"]


@pytest.mark.parametrize("axis", ["chi", "phi"])
def test_cradle_angles_subset_match_pf(run: RunData, pf_rows: list[dict[str, str]], axis: str) -> None:
    # Arrange
    cfg = json.loads((REFERENCE / "pole_figure_config.json").read_text(encoding="utf-8"))
    conv = MountConventions.from_dict(cfg["conventions"])

    # Act
    chi, phi = cradle_angles(run, conv)

    # Assert
    np.testing.assert_array_equal(chi if axis == "chi" else phi, col(pf_rows, axis))


def test_count_times_subset_match_pf(run: RunData, pf_rows: list[dict[str, str]]) -> None:
    # Arrange
    expected = col(pf_rows, "count_s")

    # Act
    count_s = run.count_s

    # Assert
    np.testing.assert_allclose(count_s, expected, atol=5.01e-3)


def test_measure_peaks_subset_finds_the_222_peak(run: RunData, peak: dict[str, Any]) -> None:
    """
    The IN718 reflections at the run's wavelength: only {222} is central (arm 2theta 112).

    The subset sums 13 of the 649 sub-runs, so its centroid differs from the full run's (fitted_peaks.json) by
    0.026 deg, about one bin (0.024 deg). The tolerance is two bins. The FWHM is a span of whole bins, so it may
    differ by one bin.
    """
    # Arrange
    lam = run.wavelength_A
    candidates = [(r.label, r.two_theta) for r in presets()["IN718 (fcc)"].reflections(lam)]
    bin_deg = float(np.diff(run.x).mean())

    # Act
    found = measure_peaks(run, candidates)

    # Assert
    assert [p.hkl for p in found] == [peak["hkl"]]
    assert found[0].two_theta == pytest.approx(peak["two_theta"], abs=0.05)
    assert found[0].fwhm == pytest.approx(peak["fwhm"], abs=1.01 * bin_deg)


@pytest.mark.parametrize(("index", "column"), [(0, "raw_net_per_s"), (1, "raw_sigma_per_s")])
def test_integrate_subset_matches_pf(run: RunData, pf_rows: list[dict[str, str]], peak: dict[str, Any],
                                     index: int, column: str) -> None:  # fmt: skip
    # Arrange
    half = WINDOW_FWHM * peak["fwhm"]

    # Act
    net = np.array([integrate(run.x, run.y[i], center=peak["two_theta"], half_width=half) for i in range(run.n)])

    # Assert
    np.testing.assert_allclose(net[:, index] / run.count_s, col(pf_rows, column), atol=5.01e-5)
