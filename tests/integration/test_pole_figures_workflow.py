"""
Multi-component workflows: a synthetic run through analyze_run, and worker processes against the serial code.

analyze_run joins runs, peaks, the gauge and the mesh. The parallel tests start worker processes (spawn), so they have
their own timeouts.
"""

from pathlib import Path

import numpy as np
import pytest

from pyrs.pole_figures.core import parallel
from pyrs.pole_figures.core.gauge import IntensityFactor, gauge_points, intensity_factor
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.polefig import PolePeak, analyze_run
from pyrs.pole_figures.core.prediction import Beams, Grid, Plane, Prediction, predict
from pyrs.pole_figures.core.runs import RunData, read_run
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, STL_DIR, slab_mesh, write_reduced_run

pytestmark = pytest.mark.integration

CONV = MountConventions(incident_from="+vy")


# ---------------------------------------------------------------- analyze_run on a synthetic run
@pytest.fixture(scope="module")
def slab() -> SampleMesh:
    return slab_mesh()


@pytest.fixture
def synthetic_run(tmp_path: Path) -> RunData:
    return read_run(write_reduced_run(tmp_path / "HB2B_1.h5"))


@pytest.fixture
def analyzed(synthetic_run: RunData, slab: SampleMesh) -> tuple[list[PolePeak], list[float]]:
    seen: list[float] = []
    res = analyze_run(synthetic_run, [("110", 90.0), ("200", 82.0)], mesh=slab, center=SLAB_CENTER_MM, conv=CONV,
                      receiving_slit_mm=3.0, mu=0.14, n_points=2000, progress=lambda f, _m: seen.append(f))  # fmt: skip
    return res, seen


def test_analyze_run_keeps_peaks_near_the_arm(analyzed: tuple[list[PolePeak], list[float]]) -> None:
    # Arrange
    res, _ = analyzed

    # Act
    hkls = [r.hkl for r in res]

    # Assert
    assert hkls == ["110"]  # 200 at 82 deg is not within 5 deg of the arm (90)


def test_analyze_run_takes_beams_from_logs(analyzed: tuple[list[PolePeak], list[float]]) -> None:
    # Arrange
    r = analyzed[0][0]

    # Act
    beams, omega = r.beams, r.omega

    # Assert
    assert beams == Beams(incident_width=4.0, incident_height=4.0, diffracted_width=3.0)
    assert omega == 45.0


def test_analyze_run_reference_correction_is_one(analyzed: tuple[list[PolePeak], list[float]]) -> None:
    # Arrange
    r = analyzed[0][0]

    # Act
    first = r.correction[0]

    # Assert
    assert first == pytest.approx(1.0)


def test_analyze_run_reports_progress_to_one(analyzed: tuple[list[PolePeak], list[float]]) -> None:
    # Arrange
    _, seen = analyzed

    # Act
    last = seen[-1]

    # Assert
    assert last == pytest.approx(1.0)


def test_analyze_run_clean_run_has_no_notes(analyzed: tuple[list[PolePeak], list[float]]) -> None:
    # Arrange
    r = analyzed[0][0]

    # Act
    notes = r.notes

    # Assert
    assert notes == ""


def test_analyze_run_without_arm_peak_is_empty(synthetic_run: RunData, slab: SampleMesh) -> None:
    # Arrange
    candidates = [("200", 82.0)]

    # Act
    res = analyze_run(synthetic_run, candidates, mesh=slab, center=SLAB_CENTER_MM, conv=CONV, receiving_slit_mm=3.0,
                      mu=0.14, n_points=500)  # fmt: skip

    # Assert
    assert res == []


# ---------------------------------------------------------------- worker processes against serial
@pytest.fixture(scope="module")
def parallel_and_serial() -> tuple[list[IntensityFactor], list[IntensityFactor], list[int], int]:
    """Intensity factors on the hollow sphere: serial, and in 2 worker processes with progress counts."""
    mesh = SampleMesh.from_stl(STL_DIR / "hollow_sphere.stl")
    m = Mount(MountConventions(), omega=30.0, tth=60.0)
    pts, vol = gauge_points(m, incident_width=2, incident_height=2, diffracted_width=2, n_points=800)
    chi, phi = Grid(30, 60).points()
    serial = [intensity_factor(m, chi=c, phi=f, points=pts, volume=vol, mesh=mesh, center=np.zeros(3), mu=0.1)
              for c, f in zip(chi, phi, strict=True)]  # fmt: skip
    seen: list[int] = []
    par = parallel.intensity_factors(m, chi=chi, phi=phi, points=pts, volume=vol, mesh=mesh, center=np.zeros(3), mu=0.1,
                                     progress=seen.append, workers=2)  # fmt: skip
    return par, serial, seen, len(chi)


@pytest.mark.timeout(120)
def test_parallel_intensity_factors_equal_serial(
    parallel_and_serial: tuple[list[IntensityFactor], list[IntensityFactor], list[int], int],
) -> None:
    # Arrange
    par, serial, _, _ = parallel_and_serial

    # Act
    got = np.array(par, float)

    # Assert
    assert len(par) == len(serial)
    np.testing.assert_array_equal(got, np.array(serial, float))


@pytest.mark.timeout(120)
def test_parallel_progress_reaches_total(
    parallel_and_serial: tuple[list[IntensityFactor], list[IntensityFactor], list[int], int],
) -> None:
    # Arrange
    _, _, seen, n = parallel_and_serial

    # Act
    last = seen[-1]

    # Assert
    assert last == n


def predict_with_workers(monkeypatch: pytest.MonkeyPatch, workers: str) -> Prediction:
    """Run predict on the in-memory slab with HIDRA_PF_WORKERS set (predict chooses its workers from it)."""
    monkeypatch.setenv("HIDRA_PF_WORKERS", workers)
    return predict(slab_mesh(), center=SLAB_CENTER_MM, conv=CONV,
                   planes=[Plane.symmetric("200", 57.0), Plane("222", 112.0, 56.0)], beams=Beams(), mu=0.14,
                   grid=Grid(30, 60), n_points=1000, best_phi=True)  # fmt: skip


@pytest.mark.timeout(120)
def test_predict_two_workers_equal_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    serial = predict_with_workers(monkeypatch, "1")

    # Act
    par = predict_with_workers(monkeypatch, "2")

    # Assert
    for a, b in zip(serial.planes, par.planes, strict=True):
        np.testing.assert_array_equal(np.c_[b.F, b.mean_path, b.correction, b.phi],
                                      np.c_[a.F, a.mean_path, a.correction, a.phi])  # fmt: skip
