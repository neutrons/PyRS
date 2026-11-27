"""Worker-count rules and result checks of the parallel tracer (no worker processes are started here)."""

import numpy as np
import pytest

from pyrs.pole_figures.core import parallel
from pyrs.pole_figures.core.gauge import IntensityFactor, gauge_points
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from tests.util.pole_figures_helpers import box_triangles


def test_n_workers_small_job_is_serial() -> None:
    # Arrange
    work = 1.0

    # Act
    n = parallel.n_workers(work)

    # Assert
    assert n == 1


def test_n_workers_large_job_is_capped() -> None:
    # Arrange
    work = 1e12

    # Act
    n = parallel.n_workers(work)

    # Assert
    assert 1 <= n <= parallel.MAX_WORKERS


def test_n_workers_requested_count_is_used() -> None:
    # Arrange
    requested = 3

    # Act
    n = parallel.n_workers(1e12, requested=requested)

    # Assert
    assert n == 3


def test_n_workers_zero_requested_raises() -> None:
    # Arrange
    requested = 0

    # Act / Assert
    with pytest.raises(ValueError, match="got 0"):
        parallel.n_workers(1e12, requested=requested)


def test_n_workers_environment_switches_parallel_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("HIDRA_PF_WORKERS", "1")  # the user can switch parallel work off

    # Act
    n = parallel.n_workers(1e12)

    # Assert
    assert n == 1


@pytest.mark.parametrize("value", ["four", "²"])  # "²" passes str.isdigit but not int()
def test_n_workers_bad_environment_warns(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # Arrange
    monkeypatch.setenv("HIDRA_PF_WORKERS", value)

    # Act
    with pytest.warns(UserWarning, match="not a positive whole number"):
        n = parallel.n_workers(1.0)

    # Assert
    assert n == 1


def test_usable_cpus_is_positive() -> None:
    # Arrange
    minimum = 1

    # Act
    n = parallel.usable_cpus()

    # Assert
    assert n >= minimum


def test_intensity_factors_missing_result_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An orientation without a result must raise, not shorten the list (that would misalign chi and phi)."""
    # Arrange
    mesh = SampleMesh(box_triangles([-5, -5, -5], [5, 5, 5]))
    m = Mount(MountConventions(), omega=45.0, tth=90.0)
    pts, vol = gauge_points(m, incident_width=1, incident_height=1, diffracted_width=1, n_points=10)
    results = iter([None, IntensityFactor(F=1.0, volume_in_sample=1.0, mean_path=1.0, vy_offset=0.0, vy_rms=0.0)])
    monkeypatch.setattr(parallel, "intensity_factor", lambda *a, **k: next(results))

    # Act / Assert
    with pytest.raises(RuntimeError, match="1 of 2 orientations returned no result"):
        parallel.intensity_factors(m, chi=np.zeros(2), phi=np.zeros(2), points=pts, volume=vol, mesh=mesh,
                                   center=np.zeros(3), mu=0.1, workers=1)  # fmt: skip
