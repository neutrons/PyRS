"""Gauge volume Monte Carlo, the off-center slit band, the intensity factor in an analytic box, and beam sizes."""

import numpy as np
import pytest

from pyrs.pole_figures.core.gauge import (
    IntensityFactor,
    beam_value,
    gauge_points,
    intensity_factor,
    set_slit_offset,
)
from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from tests.util.pole_figures_helpers import box_triangles

CONV = MountConventions(incident_from="+vy")
EX = np.array([1.0, 0.0, 0.0])


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- gauge volume
def test_gauge_points_volume_at_90_deg() -> None:
    """At 2theta 90 the incident beam and the band are perpendicular: volume = wi * wd * hi."""
    # Arrange
    m = Mount(MountConventions(), omega=45, tth=90)

    # Act
    _, vol = gauge_points(m, incident_width=4, incident_height=4, diffracted_width=3, n_points=1000)

    # Assert
    assert vol == pytest.approx(48.0)


def test_gauge_points_shape() -> None:
    # Arrange
    m = Mount(MountConventions(), omega=45, tth=90)

    # Act
    pts, _ = gauge_points(m, incident_width=4, incident_height=4, diffracted_width=3, n_points=1000)

    # Assert
    assert pts.shape == (1000, 3)


def test_gauge_points_height_along_ex() -> None:
    # Arrange
    m = Mount(MountConventions(), omega=45, tth=90)

    # Act
    pts, _ = gauge_points(m, incident_width=4, incident_height=4, diffracted_width=3, n_points=1000)

    # Assert
    assert np.all(np.abs(pts[:, 0]) <= 2.0)


def test_gauge_volume_grows_as_one_over_sin_2theta() -> None:
    # Arrange
    m = Mount(MountConventions(), omega=28.5, tth=57.0)

    # Act
    _, vol = gauge_points(m, incident_width=4, incident_height=4, diffracted_width=3, n_points=10)

    # Assert
    assert vol == pytest.approx(48.0 / np.sin(np.radians(57.0)))


# ---------------------------------------------------------------- off-center slit band
def test_set_slit_offset_zero_slit_leaves_band() -> None:
    # Arrange
    m = Mount(CONV, omega=28.5, tth=57.0)
    before = m.band_offset

    # Act
    out = set_slit_offset(m, conv=CONV, omega=28.5, arm_tth=57.0, slit_mm=0.0)

    # Assert
    assert out.band_offset == before


@pytest.mark.parametrize(("peak_tth", "arm_tth", "slit_mm"), [(57.4, 57.0, 300.0), (84.6, 85.1, 250.0),
                                                              (112.3, 111.0, 120.0)])  # fmt: skip
def test_set_slit_offset_is_slit_distance_times_sine(peak_tth: float, arm_tth: float, slit_mm: float) -> None:
    """The band center line (along kf) passes slit_mm |sin(arm - peak)| from the gauge center."""
    # Arrange
    m = Mount(CONV, omega=arm_tth / 2, tth=peak_tth)

    # Act
    set_slit_offset(m, conv=CONV, omega=arm_tth / 2, arm_tth=arm_tth, slit_mm=slit_mm)

    # Assert
    assert abs(m.band_offset) == pytest.approx(slit_mm * abs(np.sin(np.radians(arm_tth - peak_tth))), rel=1e-12)


@pytest.mark.parametrize("side", ["+vy", "-vy"])
def test_set_slit_offset_band_passes_slit_center(side: str) -> None:
    """The slit center (slit_mm along the arm direction) lies on the band center line: offset * n_f + t kf."""
    # Arrange
    conv = MountConventions(incident_from=side)  # type: ignore[arg-type]
    omega, arm_tth, peak_tth, slit_mm = 42.5, 85.1, 84.2, 250.0
    m = Mount(conv, omega=omega, tth=peak_tth)
    k_arm = Mount(conv, omega=omega, tth=arm_tth).kf

    # Act
    set_slit_offset(m, conv=conv, omega=omega, arm_tth=arm_tth, slit_mm=slit_mm)

    # Assert
    off_line = slit_mm * k_arm - m.band_offset * unit(np.cross(EX, m.kf))
    np.testing.assert_allclose(np.cross(off_line, m.kf), 0.0, atol=1e-9)


@pytest.fixture
def band_shift() -> tuple[Mount, np.ndarray, float, float]:
    """The shift of every gauge point caused by a 0.8 mm band offset (same seed), and both volumes."""
    m0 = Mount(CONV, omega=40.0, tth=85.0)
    m1 = Mount(CONV, omega=40.0, tth=85.0)
    m1.set_band_offset(0.8)
    kw = {"incident_width": 2.0, "incident_height": 3.0, "diffracted_width": 1.5, "n_points": 500}
    p0, v0 = gauge_points(m0, **kw)  # type: ignore[arg-type]
    p1, v1 = gauge_points(m1, **kw)  # type: ignore[arg-type]
    return m1, p1 - p0, v0, v1


def test_band_offset_moves_gauge_along_band_normal(band_shift: tuple[Mount, np.ndarray, float, float]) -> None:
    # Arrange
    m, shift, _, _ = band_shift
    n_f = unit(np.cross(EX, m.kf))

    # Act
    along = shift @ n_f

    # Assert
    np.testing.assert_allclose(along, 0.8, atol=1e-12)


def test_band_offset_keeps_incident_beam_and_height(band_shift: tuple[Mount, np.ndarray, float, float]) -> None:
    # Arrange
    m, shift, _, _ = band_shift
    n_i = unit(np.cross(EX, m.ki))

    # Act
    across_beam, height = shift @ n_i, shift @ EX

    # Assert
    np.testing.assert_allclose(across_beam, 0.0, atol=1e-12)
    np.testing.assert_allclose(height, 0.0, atol=1e-12)


def test_band_offset_keeps_gauge_volume(band_shift: tuple[Mount, np.ndarray, float, float]) -> None:
    # Arrange
    _, _, v0, v1 = band_shift

    # Act
    ratio = v1 / v0

    # Assert
    assert ratio == pytest.approx(1.0, abs=1e-15)


# ---------------------------------------------------------------- intensity factor
@pytest.fixture(scope="module")
def deep_gauge() -> tuple[float, IntensityFactor]:
    """A gauge deep inside a large block, long in vy so both beams leave through the +vz face at 45 deg."""
    box = SampleMesh(box_triangles([-50.0, -200, -50], [50.0, 200, 50]))
    m = Mount(MountConventions(), omega=45, tth=90)
    pts, vol = gauge_points(m, incident_width=1, incident_height=1, diffracted_width=1, n_points=2000)
    return vol, intensity_factor(m, chi=0, phi=0, points=pts, volume=vol, mesh=box, center=[0, 0, 0], mu=0.0)


def test_intensity_factor_whole_gauge_inside(deep_gauge: tuple[float, IntensityFactor]) -> None:
    # Arrange
    vol, r = deep_gauge

    # Act
    inside = r.volume_in_sample

    # Assert
    assert inside == pytest.approx(vol)


def test_intensity_factor_without_attenuation_is_volume(deep_gauge: tuple[float, IntensityFactor]) -> None:
    # Arrange
    vol, r = deep_gauge

    # Act
    F = r.F

    # Assert
    assert F == pytest.approx(vol)


def test_intensity_factor_mean_path_through_top_face(deep_gauge: tuple[float, IntensityFactor]) -> None:
    """In and out through the +vz face at 45 deg: 50 / sin 45 each way (linear in the point, so the mean is exact)."""
    # Arrange
    _, r = deep_gauge

    # Act
    path = r.mean_path

    # Assert
    assert path == pytest.approx(2 * 50 * np.sqrt(2), rel=2e-3)


# ---------------------------------------------------------------- attenuation (mu > 0) in the same block
MU = 0.05  # 1/mm: exp(-mu x 141 mm) = 8.5e-4, so the attenuation dominates F


Attenuated = tuple[np.ndarray, float, IntensityFactor]  # gauge points, gauge volume, F at chi 0, phi 0


@pytest.fixture(scope="module")
def deep_gauge_attenuated() -> Attenuated:
    """The deep gauge with attenuation. At chi 0, phi 0 the cradle and sample frames agree (identity mount)."""
    box = SampleMesh(box_triangles([-50.0, -200, -50], [50.0, 200, 50]))
    m = Mount(MountConventions(), omega=45, tth=90)
    pts, vol = gauge_points(m, incident_width=1, incident_height=1, diffracted_width=1, n_points=2000)
    return pts, vol, intensity_factor(m, chi=0, phi=0, points=pts, volume=vol, mesh=box, center=[0, 0, 0], mu=MU)


def test_intensity_factor_attenuation_per_point(deep_gauge_attenuated: Attenuated) -> None:
    """Each point at height z goes in and out through the +vz face at 45 deg: path 2 sqrt(2) (50 - z)."""
    # Arrange
    pts, vol, r = deep_gauge_attenuated
    path = 2 * np.sqrt(2) * (50.0 - pts[:, 2])

    # Act
    expected = vol * np.mean(np.exp(-MU * path))

    # Assert
    assert r.F == pytest.approx(expected, rel=1e-12)


def test_intensity_factor_attenuation_closed_form(deep_gauge_attenuated: Attenuated) -> None:
    """
    F / vol = exp(-100 sqrt(2) mu) (sinh(mu) / mu)^2 over the gauge.

    The gauge is uniform in s1, s2 (incident and diffracted widths, 1 mm each) with z = (s1 + s2) / sqrt(2), so
    mean(exp(2 sqrt(2) mu z)) = (sinh(mu) / mu)^2. The tolerance is 4 standard errors of the 2000-point mean (the
    weight varies by about 4 % rms over the gauge, so one standard error is about 0.1 %).
    """
    # Arrange
    _, vol, r = deep_gauge_attenuated

    # Act
    closed_form = np.exp(-100 * np.sqrt(2) * MU) * (np.sinh(MU) / MU) ** 2

    # Assert
    assert r.F / vol == pytest.approx(closed_form, rel=4e-3)


# ---------------------------------------------------------------- beam sizes
def test_beam_value_number() -> None:
    # Arrange
    spec = 3.0

    # Act
    size = beam_value(spec, {})

    # Assert
    assert size == 3.0


def test_beam_value_log_median() -> None:
    # Arrange
    logs = {"gap": [4.0, 4.0, 5.0]}

    # Act
    size = beam_value("log:gap", logs)

    # Assert
    assert size == 4.0


@pytest.mark.parametrize(("spec", "message"), [("wide", "must be a number"), ("log:ISR:X:Gap", "names a log")])
def test_beam_value_bad_spec_raises(spec: str, message: str) -> None:
    # Arrange
    logs = {"gap": [4.0]}

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        beam_value(spec, logs)
