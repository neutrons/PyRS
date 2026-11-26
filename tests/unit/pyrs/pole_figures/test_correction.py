"""Path-length correction step: beam sizes, the reference check, the analytic correction and selecting points."""

import numpy as np
import pytest

from pyrs.pole_figures.core.correction import Beams, GaugeCorrections, check_reference, gauge_corrections
from pyrs.pole_figures.core.gauge import gauge_points
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from pyrs.pole_figures.core.types import no_count_progress
from tests.util.pole_figures_helpers import box_triangles


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_beams_non_positive_size_raises(bad: float) -> None:
    # Arrange
    width = bad

    # Act / Assert
    with pytest.raises(ValueError, match="positive size"):
        Beams(incident_width=width)


# ---------------------------------------------------------------- correction with attenuation, analytic
def test_gauge_corrections_attenuated_block_analytic() -> None:
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
    g = gauge_corrections(m, beams=beams, chi=np.array([0.0, 0.0]), phi=np.array([0.0, 90.0]), mesh=box,
                          center=[0.0, 0.0, 0.0], mu=mu, n_points=2000, progress=no_count_progress,
                          workers=1)  # fmt: skip

    # Assert
    np.testing.assert_allclose(g.correction, [1.0, expected], rtol=1e-12)


@pytest.mark.parametrize("f_ref", [0.0, float("nan")])
def test_check_reference_empty_gauge_raises(f_ref: float) -> None:
    # Arrange
    center = (1.0, 2.0, 3.0)

    # Act / Assert
    with pytest.raises(ValueError, match=r"gauge volume at \(1, 2, 3\) mm holds no sample"):
        check_reference(f_ref, center)


def test_gauge_corrections_take_keeps_reference() -> None:
    # Arrange
    g = GaugeCorrections(F=np.array([1.0, 2.0, 0.0]), F_ref=2.0, volume_in_sample=np.array([1.0, 2.0, 0.0]),
                         mean_path=np.array([3.0, 1.0, np.nan]), vy_offset=np.zeros(3), vy_rms=np.ones(3),
                         correction=np.array([2.0, 1.0, np.inf]))  # fmt: skip

    # Act
    part = g.take([2, 0])

    # Assert
    assert (part.F_ref, part.F.tolist(), part.correction.tolist()) == (2.0, [0.0, 1.0], [np.inf, 2.0])
