"""
Path-length correction step: beam sizes, the reference check, the analytic correction and its parts.

Also selecting points, and the split of the correction into its gauge-volume and attenuation parts.
"""

import numpy as np
import pytest

from pyrs.pole_figures.core.correction import (
    PARTS,
    Beams,
    GaugeCorrections,
    check_reference,
    correction_parts,
    gauge_corrections,
)
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


def _split_gauge() -> tuple[GaugeCorrections, np.ndarray, np.ndarray]:
    """Four orientations: the reference, half the volume, more attenuation, and the gauge off the sample."""
    V = np.array([40.0, 20.0, 40.0, 0.0])
    A = np.array([0.5, 0.5, 0.25, np.nan])  # mean exp(-mu path) over the sample part of the gauge
    F = np.nan_to_num(V * A)
    with np.errstate(divide="ignore"):
        corr = np.where(F > 0, F[0] / np.where(F > 0, F, 1), np.inf)
    g = GaugeCorrections(F=F, F_ref=float(F[0]), volume_in_sample=V, mean_path=np.array([4.0, 4.0, 8.0, np.nan]),
                         vy_offset=np.zeros(4), vy_rms=np.ones(4), correction=corr)  # fmt: skip
    return g, np.array([0.0, 30.0, 60.0, 90.0]), np.zeros(4)


def test_correction_parts_split_volume_and_attenuation() -> None:
    # Arrange
    g, chi, phi = _split_gauge()

    # Act
    parts = correction_parts(g, chi=chi, phi=phi)

    # Assert
    np.testing.assert_allclose(parts["volume"][:3], [1.0, 2.0, 1.0])
    np.testing.assert_allclose(parts["attenuation"][:3], [1.0, 1.0, 2.0])
    np.testing.assert_allclose(parts["volume"][:3] * parts["attenuation"][:3], parts["total"][:3], rtol=1e-12)


def test_correction_parts_gauge_off_sample_inf_and_nan() -> None:
    # Arrange
    g, chi, phi = _split_gauge()

    # Act
    parts = correction_parts(g, chi=chi, phi=phi)

    # Assert
    assert np.isinf(parts["volume"][3]) and np.isnan(parts["attenuation"][3]) and np.isinf(parts["total"][3])
    assert tuple(parts) == PARTS


def test_correction_parts_without_chi_zero_raises() -> None:
    # Arrange: the orientations moved off chi 0
    g, chi, phi = _split_gauge()

    # Act / Assert
    with pytest.raises(ValueError, match="no grid point at chi 0, phi 0 among the 4 orientations"):
        correction_parts(g, chi=chi + 5.0, phi=phi)
