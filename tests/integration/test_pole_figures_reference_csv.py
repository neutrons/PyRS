"""
The CSVs we write against PF's (the reference script's) files, column by column at the written rounding.

Predicted CSV: a prediction made from the reference config (as test_pole_figures_parity.py makes it) is written with
``write_predicted_csv`` and compared with predicted_correction.csv. Every 16th grid point of each plane is checked.

Per-run CSV: HB2B_6494_subset.h5 holds 13 sub-runs of run 6494 (test_pole_figures_hb2b_subset.py reads it). The pole
figure is built here as PF built run 6494, then written with ``write_pole_figure_csv``:

- the window uses the full run's peak center and FWHM (fitted_peaks.json), since the subset's own peak differs by
  about one bin (raw rates and sigmas, and so the corrected columns, depend on it)
- omega and the fitted 2theta of the mount also come from fitted_peaks.json
- the corrections are the gauge Monte Carlo with PF's settings (40,000 points, seed 3, the reference config's slab,
  gauge center, mu and 3 mm receiving slit, incident slits from the run's logs). The Monte Carlo is seeded, so F,
  volume, path, vy columns, correction and the ignored flag are reproduced, not copied from PF's file.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyrs.pole_figures.core.correction import Beams, gauge_corrections
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.grid import Grid, solid_angle_weights
from pyrs.pole_figures.core.peaks import WINDOW_FWHM, integrate
from pyrs.pole_figures.core.polefig import PolePeak, cradle_angles
from pyrs.pole_figures.core.prediction import Plane, Prediction, predict_plane
from pyrs.pole_figures.core.runs import read_run
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from pyrs.pole_figures.core.types import no_count_progress
from pyrs.pole_figures.io.polefig_csv import csv_name, write_pole_figure_csv
from pyrs.pole_figures.io.predicted_csv import write_predicted_csv
from tests.util.pole_figures_helpers import HB2B_SUBSET, REFERENCE, STL_DIR, read_csv_rows

pytestmark = pytest.mark.integration

Rows = tuple[list[dict[str, str]], list[dict[str, str]]]
#: predicted CSV columns and their written decimals (``sample_volume_in_gauge_mm3`` is PF's plate volume)
PREDICTED_DECIMALS = {"pole_vx": 5, "pole_vy": 5, "pole_vz": 5, "mean_path_mm": 3, "sample_volume_in_gauge_mm3": 3,
                      "correction": 4}  # fmt: skip
PF_NAME = {"sample_volume_in_gauge_mm3": "plate_volume_in_gauge_mm3"}


@pytest.fixture(scope="module")
def predicted_rows(tmp_path_factory: pytest.TempPathFactory) -> Rows:
    """Our predicted CSV rows for every 16th grid point of each plane, and PF's rows for the same points."""
    cfg = json.loads((REFERENCE / "pole_figure_config.json").read_text(encoding="utf-8"))
    conv = MountConventions.from_dict(cfg["conventions"])
    beams = Beams(incident_width=4.0, incident_height=4.0, diffracted_width=cfg["beams"]["diffracted_width_mm"])
    chi_step, phi_step = cfg["predict"]["chi_step"], cfg["predict"]["phi_step"]
    chi_all, _ = Grid(chi_step, phi_step).points()
    w_all = solid_angle_weights(chi_all, chi_step=chi_step, phi_step=phi_step)
    reference = read_csv_rows(REFERENCE / "predicted_correction.csv")
    mesh = SampleMesh.from_stl(STL_DIR / "slab.stl")
    planes, pf = [], []
    for p in cfg["predict"]["planes"]:
        sub = [r for r in reference if r["hkl"] == p["hkl"]][::16]
        pf += sub
        planes.append(predict_plane(
            mesh, center=cfg["sample"]["gauge_center_mm"], conv=conv, plane=Plane(p["hkl"], p["two_theta"], p["omega"]),
            beams=beams, mu=cfg["sample"]["mu_per_mm"], chi=np.array([float(r["chi"]) for r in sub]),
            phi=np.array([float(r["phi"]) for r in sub]), weights=w_all[::16], cutoff=cfg["ignore_if_correction_above"],
            n_points=cfg["monte_carlo_points"] // 2, best_phi_step=phi_step))  # fmt: skip
    pred = Prediction(planes=planes, center=np.asarray(cfg["sample"]["gauge_center_mm"], float), beams=beams,
                      mu=cfg["sample"]["mu_per_mm"], cutoff=cfg["ignore_if_correction_above"],
                      grid=Grid(chi_step, phi_step), n_points=cfg["monte_carlo_points"] // 2)  # fmt: skip
    path = Path(tmp_path_factory.mktemp("predicted")) / "predicted_correction.csv"
    write_predicted_csv(path, pred, cutoff=cfg["ignore_if_correction_above"])
    return read_csv_rows(path), pf


def test_write_predicted_csv_text_and_angles_equal_pf(predicted_rows: Rows) -> None:
    # Arrange
    ours, pf = predicted_rows
    names = ("hkl", "two_theta", "omega", "chi", "phi", "ignored")

    # Act
    got = [[float(r[n]) for n in names] for r in ours]

    # Assert
    assert len(ours) > 150
    assert got == [[float(r[n]) for n in names] for r in pf]


@pytest.mark.parametrize("column", sorted(PREDICTED_DECIMALS))
def test_write_predicted_csv_column_matches_pf(predicted_rows: Rows, column: str) -> None:
    # Arrange
    ours, pf = predicted_rows
    atol = 1.01 * 10.0 ** -PREDICTED_DECIMALS[column]

    # Act
    got = col(ours, column)

    # Assert
    np.testing.assert_allclose(got, col(pf, PF_NAME.get(column, column)), atol=atol, rtol=0)


#: columns written with d decimals. PF's value and ours may differ by one unit in the last place: the raw values agree
#: to half a unit, and rounding can then split them.
WRITTEN_DECIMALS = {"pole_vx": 5, "pole_vy": 5, "pole_vz": 5, "count_s": 2, "raw_net_per_s": 4, "raw_sigma_per_s": 4,
                    "F_mm3": 4, "plate_volume_in_gauge_mm3": 3, "mean_path_mm": 3, "sampled_vy_offset_mm": 3,
                    "sampled_vy_rms_mm": 3, "correction": 4}  # fmt: skip


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(r[name]) for r in rows])


@pytest.fixture(scope="module")
def rows(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Our written rows for the subset, and PF's rows for the same sub-runs (sub-run k is PF's row k - 1)."""
    run = read_run(HB2B_SUBSET)
    peak: dict[str, Any] = json.loads((REFERENCE / "fitted_peaks.json").read_text(encoding="utf-8"))["6494"]
    cfg = json.loads((REFERENCE / "pole_figure_config.json").read_text(encoding="utf-8"))
    conv = MountConventions.from_dict(cfg["conventions"])
    chi, phi = cradle_angles(run, conv)
    m = Mount(conv, omega=peak["omega"], tth=peak["two_theta"])
    beams = Beams(incident_width=run.incident_width, incident_height=run.incident_height,
                  diffracted_width=cfg["beams"]["diffracted_width_mm"])  # fmt: skip
    half = WINDOW_FWHM * peak["fwhm"]
    net = np.array([integrate(run.x, run.y[i], center=peak["two_theta"], half_width=half) for i in range(len(chi))])
    g = gauge_corrections(m, beams=beams, chi=chi, phi=phi, mesh=SampleMesh.from_stl(STL_DIR / "slab.stl"),
                          center=cfg["sample"]["gauge_center_mm"], mu=cfg["sample"]["mu_per_mm"],
                          n_points=cfg["monte_carlo_points"], progress=no_count_progress, workers="auto")  # fmt: skip
    r = PolePeak(run=run.run, hkl=peak["hkl"], title=run.title, two_theta=peak["two_theta"], fwhm=peak["fwhm"],
                 wavelength_A=run.wavelength_A, beams=beams, mount=m, chi=chi, phi=phi, count_s=run.count_s,
                 rate=net[:, 0] / run.count_s, sigma=net[:, 1] / run.count_s, gauge=g,
                 cutoff=cfg["ignore_if_correction_above"], mu=cfg["sample"]["mu_per_mm"], method="window",
                 h=m.poles(chi, phi))  # fmt: skip
    path = Path(tmp_path_factory.mktemp("subset")) / csv_name(r)
    write_pole_figure_csv(path, r)
    pf = read_csv_rows(REFERENCE / "pole_figure_6494_222.csv")
    return read_csv_rows(path), [pf[int(k) - 1] for k in run.subruns]


def test_write_csv_subset_text_and_angles_equal_pf(rows: tuple[list[dict[str, str]], list[dict[str, str]]]) -> None:
    # Arrange
    ours, pf = rows
    names = ("run", "hkl", "ignored")

    # Act
    got = [[r[n] for n in names] for r in ours]

    # Assert: the text columns exactly, chi and phi to the bit
    assert got == [[r[n] for n in names] for r in pf]
    for axis in ("chi", "phi"):
        np.testing.assert_array_equal(col(ours, axis), col(pf, axis))


@pytest.mark.parametrize("column", sorted(WRITTEN_DECIMALS))
def test_write_csv_subset_column_matches_pf(rows: tuple[list[dict[str, str]], list[dict[str, str]]],
                                            column: str) -> None:  # fmt: skip
    # Arrange
    ours, pf = rows
    atol = 1.01 * 10.0 ** -WRITTEN_DECIMALS[column]

    # Act
    got = col(ours, column)

    # Assert
    np.testing.assert_allclose(got, col(pf, column), atol=atol, rtol=0)


@pytest.mark.parametrize(("column", "raw"), [("corrected_per_s", "raw_net_per_s"),
                                             ("corrected_sigma_per_s", "raw_sigma_per_s")])  # fmt: skip
def test_write_csv_subset_corrected_matches_pf(rows: tuple[list[dict[str, str]], list[dict[str, str]]], column: str,
                                               raw: str) -> None:  # fmt: skip
    # Arrange: rate x correction, the rate and the correction each within half a unit of PF's raw value
    ours, pf = rows
    atol = 1.01e-4 + 0.5e-4 * col(pf, "correction") + 0.5e-4 * np.abs(col(pf, raw))

    # Act
    got = col(ours, column)

    # Assert
    assert np.all(np.abs(got - col(pf, column)) <= atol)
