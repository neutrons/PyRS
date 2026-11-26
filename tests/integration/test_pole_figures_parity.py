"""
Parity with PF (the reference script) on the slab runs: poles, measured corrections and predicted corrections.

Reads PF's outputs in the reference fixtures (pole_figure_config.json, fitted_peaks.json, predicted_correction.csv,
pole_figure_<run>_<hkl>.csv) and the sample slab.stl. The tolerances are the decimals PF printed. The Monte
Carlo is seeded, so any change to the sampling breaks these tests, as it should.
"""

import json
from typing import Any

import numpy as np
import pytest

from pyrs.pole_figures.core.correction import Beams, gauge_corrections
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.grid import Grid, solid_angle_weights
from pyrs.pole_figures.core.prediction import Plane, predict_plane
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from pyrs.pole_figures.core.types import no_count_progress
from tests.util.pole_figures_helpers import REFERENCE, STL_DIR, read_csv_rows

pytestmark = pytest.mark.integration

RUNS = {"6494": "222", "6495": "311", "6496": "220", "6497": "200"}
PLANES = ("222", "311", "220", "200")  # pole_figure_config.json, predict.planes


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    """One numeric column of PF's CSV (an empty cell is an infinite correction)."""
    return np.array([float(r[name]) if r[name] != "" else np.inf for r in rows])


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return json.loads((REFERENCE / "pole_figure_config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_conventions(cfg: dict[str, Any]) -> MountConventions:
    return MountConventions.from_dict(cfg["conventions"])


@pytest.fixture(scope="module")
def fitted() -> dict[str, Any]:
    """The omega and fitted 2theta of each run, so these tests need no HB2B files."""
    return json.loads((REFERENCE / "fitted_peaks.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def slab_stl() -> SampleMesh:
    return SampleMesh.from_stl(STL_DIR / "slab.stl")


@pytest.fixture(scope="module")
def predicted_rows() -> list[dict[str, str]]:
    return read_csv_rows(REFERENCE / "predicted_correction.csv")


def run_mount(conv: MountConventions, fitted: dict[str, Any], run: str) -> Mount:
    f = fitted[run]
    return Mount(conv, omega=f["omega"], tth=f["two_theta"])


# ---------------------------------------------------------------- conventions and poles
def test_from_dict_reference_conventions(ref_conventions: MountConventions) -> None:
    # Arrange
    want = MountConventions(incident_from="+vy")

    # Act
    got = ref_conventions

    # Assert
    assert got == want


def test_predicted_csv_has_four_planes_of_649(predicted_rows: list[dict[str, str]]) -> None:
    # Arrange
    rows = predicted_rows

    # Act
    n = len(rows)

    # Assert
    assert n == 2596


def test_poles_match_pf_predicted_csv(ref_conventions: MountConventions, predicted_rows: list[dict[str, str]]) -> None:
    """Every pole vector in PF's predicted_correction.csv (4 planes x 649 points), to the 5 decimals PF wrote."""
    # Arrange
    mounts: dict[tuple[str, str], Mount] = {}
    want = [[float(r["pole_vx"]), float(r["pole_vy"]), float(r["pole_vz"])] for r in predicted_rows]

    # Act
    got = []
    for r in predicted_rows:
        key = (r["two_theta"], r["omega"])
        if key not in mounts:
            mounts[key] = Mount(ref_conventions, omega=float(r["omega"]), tth=float(r["two_theta"]))
        got.append(mounts[key].pole(float(r["chi"]), float(r["phi"])))

    # Assert
    np.testing.assert_allclose(np.array(got), np.array(want), atol=5.01e-6)


@pytest.mark.parametrize("run", list(RUNS))
def test_poles_match_pf_run_csv(ref_conventions: MountConventions, fitted: dict[str, Any], run: str) -> None:
    """Every pole vector PF wrote (MTEX reads these): from omega, the fitted 2theta, chi and phi."""
    # Arrange
    rows = read_csv_rows(REFERENCE / f"pole_figure_{run}_{RUNS[run]}.csv")
    m = run_mount(ref_conventions, fitted, run)

    # Act
    h = m.poles(col(rows, "chi"), col(rows, "phi"))

    # Assert
    want = np.c_[col(rows, "pole_vx"), col(rows, "pole_vy"), col(rows, "pole_vz")]
    np.testing.assert_allclose(h, want, atol=5.01e-6)


# ---------------------------------------------------------------- measured-run corrections
@pytest.fixture(scope="module", params=["6494", "6497"])
def run_corrections(request: pytest.FixtureRequest, cfg: dict[str, Any], ref_conventions: MountConventions,
                    fitted: dict[str, Any], slab_stl: SampleMesh) -> dict[str, np.ndarray]:  # fmt: skip
    """
    Corrections on every 12th point plus every point with a correction of 2 to 5 (where the ignored flag can flip).

    chi and phi come from PF's CSV, omega and the fitted 2theta from fitted_peaks.json. The full runs 6494 to 6497
    matched PF (F, path, correction and all ignored flags).
    """
    run = request.param
    rows = read_csv_rows(REFERENCE / f"pole_figure_{run}_{RUNS[run]}.csv")
    want_corr = col(rows, "correction")
    keep = np.array([i % 12 == 0 or 2 < want_corr[i] < 5 for i in range(len(rows))])
    beams = Beams(incident_width=4.0, incident_height=4.0, diffracted_width=3.0)
    g = gauge_corrections(run_mount(ref_conventions, fitted, run), beams=beams, chi=col(rows, "chi")[keep],
                          phi=col(rows, "phi")[keep], mesh=slab_stl, center=cfg["sample"]["gauge_center_mm"], mu=0.140,
                          n_points=40000, progress=no_count_progress, workers="auto")  # fmt: skip
    got = {"F_mm3": g.F, "mean_path_mm": g.mean_path, "sampled_vy_offset_mm": g.vy_offset,
           "sampled_vy_rms_mm": g.vy_rms, "correction": g.correction}  # fmt: skip
    want = {name: col(rows, name)[keep] for name in (*got, "ignored")}
    return {"keep": keep, **{f"got_{k}": v for k, v in got.items()}, **{f"want_{k}": v for k, v in want.items()}}


@pytest.mark.parametrize(
    ("name", "atol"),
    [("F_mm3", 5.01e-5), ("mean_path_mm", 5.01e-4), ("sampled_vy_offset_mm", 5.01e-4),
     ("sampled_vy_rms_mm", 5.01e-4), ("correction", 5.01e-5)],
)  # fmt: skip
def test_corrections_match_pf(run_corrections: dict[str, np.ndarray], name: str, atol: float) -> None:
    # Arrange
    res = run_corrections

    # Act
    got = res[f"got_{name}"]

    # Assert
    np.testing.assert_allclose(got, res[f"want_{name}"], atol=atol)


def test_corrections_ignored_flags_match_pf(run_corrections: dict[str, np.ndarray]) -> None:
    # Arrange
    res = run_corrections

    # Act
    ignored = res["got_correction"] > 3.0

    # Assert
    np.testing.assert_array_equal(ignored, res["want_ignored"] == 1)


def test_corrections_check_enough_points(run_corrections: dict[str, np.ndarray]) -> None:
    # Arrange
    keep = run_corrections["keep"]

    # Act
    n = keep.sum()

    # Assert
    assert n > 60


# ---------------------------------------------------------------- predicted corrections
@pytest.fixture(scope="module", params=PLANES)
def plane_prediction(request: pytest.FixtureRequest, cfg: dict[str, Any], ref_conventions: MountConventions,
                     slab_stl: SampleMesh, predicted_rows: list[dict[str, str]]) -> dict[str, Any]:  # fmt: skip
    """
    One plane of PF's predicted_correction.csv: path, volume, correction and the ignored flag.

    PF took the beams and lambda from run 6494 (ISR gaps 4 x 4 mm, 3 mm receiving slit) and used
    monte_carlo_points // 2 = 20,000 points with seed 3. To keep the test fast it checks every 8th grid point and every
    point with a correction between 2 and 5 (where the ignored flag can flip). The full grid matched on 2026-11-16.
    """
    p = next(p for p in cfg["predict"]["planes"] if p["hkl"] == request.param)
    beams = Beams(incident_width=4.0, incident_height=4.0, diffracted_width=cfg["beams"]["diffracted_width_mm"])
    chi_step, phi_step = cfg["predict"]["chi_step"], cfg["predict"]["phi_step"]
    chi_all, _ = Grid(chi_step, phi_step).points()
    w_all = solid_angle_weights(chi_all, chi_step=chi_step, phi_step=phi_step)
    rows = [r for r in predicted_rows if r["hkl"] == p["hkl"]]
    keep = [i for i, r in enumerate(rows) if i % 8 == 0 or 2 < float(r["correction"]) < 5]
    sub = [rows[i] for i in keep]
    chi = np.array([float(r["chi"]) for r in sub])
    phi = np.array([float(r["phi"]) for r in sub])
    res = predict_plane(slab_stl, center=cfg["sample"]["gauge_center_mm"], conv=ref_conventions,
                        plane=Plane(p["hkl"], p["two_theta"], p["omega"]), beams=beams, mu=cfg["sample"]["mu_per_mm"],
                        chi=chi, phi=phi, weights=w_all[keep], cutoff=cfg["ignore_if_correction_above"],
                        n_points=cfg["monte_carlo_points"] // 2, best_phi_step=phi_step)  # fmt: skip
    return {"rows": rows, "sub": sub, "chi": chi, "chi_all": chi_all, "keep": keep, "res": res}


def test_predicted_plane_has_whole_grid(plane_prediction: dict[str, Any]) -> None:
    # Arrange
    rows, chi_all = plane_prediction["rows"], plane_prediction["chi_all"]

    # Act
    n = len(rows)

    # Assert
    assert n == len(chi_all)


def test_predicted_grid_matches_pf(plane_prediction: dict[str, Any]) -> None:
    # Arrange
    chi, chi_all, keep = plane_prediction["chi"], plane_prediction["chi_all"], plane_prediction["keep"]

    # Act
    want = chi_all[keep]

    # Assert
    np.testing.assert_array_equal(chi, want)


@pytest.mark.parametrize(
    ("attr", "column", "atol"),
    [("mean_path", "mean_path_mm", 5.01e-4), ("volume_in_sample", "plate_volume_in_gauge_mm3", 5.01e-4),
     ("correction", "correction", 5.01e-5)],
)  # fmt: skip
def test_predicted_matches_pf(plane_prediction: dict[str, Any], attr: str, column: str, atol: float) -> None:
    # Arrange
    sub = plane_prediction["sub"]

    # Act
    got = getattr(plane_prediction["res"].gauge, attr)

    # Assert
    np.testing.assert_allclose(got, [float(r[column]) for r in sub], atol=atol)


def test_predicted_ignored_flags_match_pf(plane_prediction: dict[str, Any]) -> None:
    # Arrange
    sub = plane_prediction["sub"]

    # Act
    ignored = plane_prediction["res"].ignored

    # Assert
    np.testing.assert_array_equal(ignored, [r["ignored"] == "1" for r in sub])


def test_predicted_parity_checks_enough_points(cfg: dict[str, Any], predicted_rows: list[dict[str, str]]) -> None:
    """The subset the plane tests check covers more than 400 of the 2596 points."""
    # Arrange
    rows_by_plane = [[r for r in predicted_rows if r["hkl"] == p["hkl"]] for p in cfg["predict"]["planes"]]

    # Act
    checked = sum(sum(1 for i, r in enumerate(rows) if i % 8 == 0 or 2 < float(r["correction"]) < 5)
                  for rows in rows_by_plane)  # fmt: skip

    # Assert
    assert checked > 400
