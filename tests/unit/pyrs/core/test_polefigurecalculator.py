"""Tests for the pole-figure calculation (#411).

``rotate_project_q`` projects the diffraction vector onto the pole sphere with He's equations
(*Two-Dimensional X-ray Diffraction*, §8.3; also B. He, "Fundamentals of Two-dimensional X-ray
Diffraction", 2023 lecture slides 8, 9 and 27):

- h₁ and h₂ are He's, term for term, with ψ = χ;
- β = ±cos⁻¹(h₁/√(h₁²+h₂²)), with β < 0 when h₂ < 0, exactly as He defines it.

PyRS's own conventions (confirmed as intended):

- α is the angle of the pole from the sample **normal**, 90° − He's α (= sin⁻¹|h₃|), as MTEX's
  ``.jul`` expects;
- He's γ is η + 270°, and η > 0 tilts the pole to the same side as χ > 0;
- ``calculate_pole_figure`` adds 90° to a negative ω.

The expected values come from the diffraction geometry, not from the code's output. In the symmetric
Bragg position (ω = θ, χ = 0, η = 0) the diffraction vector lies along the sample normal. Tilting (χ),
rocking (ω), moving round the diffraction cone (η) or spinning the sample about its normal (φ) then
moves the pole as He's equations say. No data files are read.
"""

from typing import Dict, Sequence

import numpy as np
import pytest

from pyrs.core.polefigurecalculator import PoleFigureCalculator, export_arrays_to_ascii

THETA = 41.3  # half of a typical 2θ ≈ 82.6°; any Bragg angle works


@pytest.fixture
def calculator() -> PoleFigureCalculator:
    """A calculator with no data."""
    return PoleFigureCalculator()


def _point_logs(
    two_theta: Sequence[float],
    omega: Sequence[float],
    chi: Sequence[float],
    phi: Sequence[float],
    eta: Sequence[float],
    intensity: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Log values for ``add_input_data_set``, one entry per measured point."""
    return {
        "center": np.asarray(two_theta, dtype=float),
        "omega": np.asarray(omega, dtype=float),
        "chi": np.asarray(chi, dtype=float),
        "phi": np.asarray(phi, dtype=float),
        "eta": np.asarray(eta, dtype=float),
        "intensity": np.asarray(intensity, dtype=float),
    }


def _angle_difference(a: float, b: float) -> float:
    """``a - b`` wrapped to [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


# --- the projection: rotate_project_q ---------------------------------------------------------


def test_symmetric_reflection_pole_at_normal(calculator):
    """With ω = θ, χ = 0 and η = 0 the diffraction vector is the sample normal: α = 0."""
    # Act
    alpha, _ = calculator.rotate_project_q(THETA, THETA, 0.0, 0.0, 0.0)

    # Assert
    assert alpha == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("chi", [5.0, 30.0, 60.0, -20.0])
def test_chi_tilt_moves_pole_by_chi(calculator, chi):
    """Tilting the sample by χ in the symmetric position moves the pole χ from the normal, at β = ∓90°."""
    # Act
    alpha, beta = calculator.rotate_project_q(THETA, THETA, chi, 0.0, 0.0)

    # Assert
    assert alpha == pytest.approx(abs(chi), abs=1e-6)
    assert beta == pytest.approx(-90.0 if chi > 0 else 90.0, abs=1e-6)


@pytest.mark.parametrize("delta", [2.0, 15.0, -10.0])
def test_omega_rock_moves_pole_by_delta(calculator, delta):
    """Rocking ω by δ away from θ moves the pole δ from the normal, within the scattering plane."""
    # Act
    alpha, beta = calculator.rotate_project_q(THETA, THETA + delta, 0.0, 0.0, 0.0)

    # Assert
    assert alpha == pytest.approx(abs(delta), abs=1e-6)
    # β's sign is undefined here (h₂ is zero up to rounding), so only |β| is checked
    assert abs(beta) == pytest.approx(180.0 if delta > 0 else 0.0, abs=1e-6)


@pytest.mark.parametrize("eta", [-10.0, 5.0, 20.0])
def test_eta_moves_pole_off_normal(calculator, eta):
    """Moving η round the diffraction cone tilts the pole: cos α = sin²θ + cos²θ·cos η.

    In the symmetric position the diffraction vector is (−sinθ, cosθ·cosη, cosθ·sinη) and the
    normal is (−sinθ, cosθ, 0), so the angle between them follows from their dot product.
    """
    # Arrange
    theta, eta_rad = np.deg2rad(THETA), np.deg2rad(eta)
    expected = np.rad2deg(np.arccos(np.sin(theta) ** 2 + np.cos(theta) ** 2 * np.cos(eta_rad)))

    # Act
    alpha, _ = calculator.rotate_project_q(THETA, THETA, 0.0, 0.0, eta)

    # Assert
    assert alpha == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("eta", [-10.0, 5.0, 20.0])
def test_eta_tilts_pole_to_chi_side(calculator, eta):
    """η > 0 tilts the pole to the same side as χ > 0 (β < 0), and η < 0 to the other side."""
    # Act
    _, beta_eta = calculator.rotate_project_q(THETA, THETA, 0.0, 0.0, eta)
    _, beta_chi = calculator.rotate_project_q(THETA, THETA, np.sign(eta) * 10.0, 0.0, 0.0)

    # Assert
    assert np.sign(beta_eta) == np.sign(beta_chi) == -np.sign(eta)


@pytest.mark.parametrize(
    "omega, chi, eta",
    [(35.0, 20.0, 5.0), (50.0, -40.0, -7.5), (41.3, 60.0, 0.0), (20.0, 10.0, 12.0)],
)
def test_phi_rotation_spins_pole_about_normal(calculator, omega, chi, eta):
    """Rotating φ (about the sample normal) keeps α and shifts β by +φ.

    He's h₁ + i·h₂ is multiplied by e^{iφ} when φ changes, so β turns by +φ.
    """
    # Arrange
    alpha_0, beta_0 = calculator.rotate_project_q(THETA, omega, chi, 0.0, eta)

    for phi in [15.0, 90.0, 137.0, 250.0]:
        # Act
        alpha, beta = calculator.rotate_project_q(THETA, omega, chi, phi, eta)

        # Assert
        assert alpha == pytest.approx(alpha_0, abs=1e-6)
        assert _angle_difference(beta - beta_0, phi) == pytest.approx(0.0, abs=1e-6)


def test_full_phi_turn_gives_same_pole(calculator):
    """A 360° turn of φ is no rotation at all."""
    # Act
    pole_0 = calculator.rotate_project_q(THETA, 33.0, 25.0, 40.0, 4.0)
    pole_360 = calculator.rotate_project_q(THETA, 33.0, 25.0, 400.0, 4.0)

    # Assert
    assert pole_360[0] == pytest.approx(pole_0[0], abs=1e-9)
    assert _angle_difference(pole_360[1], pole_0[1]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DD-16: at the rim (α = 90°) √(h₁²+h₂²) can round to just above 1, and arccos returns NaN",
)
def test_tilt_to_rim_gives_alpha_ninety(calculator):
    """Tilting χ to 90° puts the pole on the rim of the pole figure, α = 90°."""
    # Act
    alpha, _ = calculator.rotate_project_q(41.1, 41.1, 90.0, 0.0, 0.0)

    # Assert
    assert alpha == pytest.approx(90.0, abs=1e-5)


# --- the calculator: calculate_pole_figure and its bookkeeping ---------------------------------


def test_calculate_uses_half_two_theta_and_keeps_intensity(calculator):
    """Each point's pole is projected from θ = 2θ/2, and its intensity is carried unchanged."""
    # Arrange
    two_theta = [2 * THETA, 2 * THETA, 2 * THETA]
    omega, chi = [THETA, THETA, THETA + 10.0], [0.0, 30.0, 0.0]
    intensities = [10.0, 20.5, 7.25]
    calculator.add_input_data_set(1, _point_logs(two_theta, omega, chi, [0.0] * 3, [0.0] * 3, intensities))

    # Act
    calculator.calculate_pole_figure()
    poles = calculator.get_polefigure_array(1)

    # Assert
    assert poles.shape == (3, 3)
    np.testing.assert_allclose(poles[:, 0], [0.0, 30.0, 10.0], atol=1e-6)
    np.testing.assert_allclose(poles[:, 2], intensities)


def test_negative_omega_is_shifted_by_ninety(calculator):
    """PyRS's convention: a negative ω is used as ω + 90°; ω ≥ 0 is used as given.

    ω = θ − 90° therefore acts as ω = θ, and with χ = 20° that is the χ tilt: α = 20°, β = −90°.
    """
    # Arrange
    omega = [THETA - 90.0, THETA, THETA + 10.0 - 90.0]
    calculator.add_input_data_set(
        1, _point_logs([2 * THETA] * 3, omega, [20.0, 20.0, 0.0], [0.0] * 3, [0.0] * 3, [1.0] * 3)
    )

    # Act
    poles = calculator.get_polefigure_array(1)

    # Assert
    np.testing.assert_allclose(poles[0, :2], [20.0, -90.0], atol=1e-6)
    np.testing.assert_allclose(poles[1, :2], [20.0, -90.0], atol=1e-6)
    assert poles[2, 0] == pytest.approx(10.0, abs=1e-6)


def test_polefigure_array_calculated_on_demand(calculator):
    """Asking for a peak's pole figure before calculating it calculates it."""
    # Arrange
    calculator.add_input_data_set(2, _point_logs([2 * THETA], [THETA], [0.0], [0.0], [0.0], [5.0]))

    # Act
    poles = calculator.get_polefigure_array(2)

    # Assert
    # β is undefined at the normal, so only α and the intensity are checked
    assert poles[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert poles[0, 2] == 5.0


def test_adding_to_a_peak_appends_points(calculator):
    """Data added again for the same peak is appended, in order, to its points."""
    # Arrange
    calculator.add_input_data_set(1, _point_logs([2 * THETA], [THETA], [0.0], [0.0], [0.0], [1.0]))
    calculator.add_input_data_set(1, _point_logs([2 * THETA], [THETA], [30.0], [0.0], [0.0], [2.0]))

    # Act
    poles = calculator.get_polefigure_array(1)

    # Assert
    np.testing.assert_allclose(poles[:, 0], [0.0, 30.0], atol=1e-6)
    np.testing.assert_allclose(poles[:, 2], [1.0, 2.0])
    assert calculator.get_detector_ids() == [1]


def test_calculate_only_listed_peaks(calculator):
    """``calculate_pole_figure([1])`` calculates peak 1 and leaves the others alone."""
    # Arrange
    for peak in (1, 2):
        calculator.add_input_data_set(peak, _point_logs([2 * THETA], [THETA], [0.0], [0.0], [0.0], [1.0]))

    # Act
    calculator.calculate_pole_figure([1])

    # Assert
    assert list(calculator._pole_figure_dict) == [1]


def test_calculate_unknown_peak_raises(calculator):
    """Asking for a peak that has no data is refused."""
    # Arrange
    calculator.add_input_data_set(1, _point_logs([2 * THETA], [THETA], [0.0], [0.0], [0.0], [1.0]))

    # Act / Assert
    with pytest.raises(RuntimeError):
        calculator.calculate_pole_figure([7])


def test_one_point_matches_its_row(calculator):
    """``get_pole_figure_1_pt`` gives the (α, β) of the matching row of the pole figure."""
    # Arrange
    calculator.add_input_data_set(
        1, _point_logs([2 * THETA] * 2, [THETA] * 2, [0.0, 45.0], [0.0] * 2, [0.0] * 2, [1, 1])
    )
    calculator.calculate_pole_figure()

    # Act
    alpha, beta = calculator.get_pole_figure_1_pt(1, 1)

    # Assert
    assert alpha == pytest.approx(45.0, abs=1e-6)
    assert beta == pytest.approx(-90.0, abs=1e-6)


def test_reset_forgets_all_data(calculator):
    """After a reset there are no peaks and no pole figures."""
    # Arrange
    calculator.add_input_data_set(1, _point_logs([2 * THETA], [THETA], [0.0], [0.0], [0.0], [1.0]))
    calculator.calculate_pole_figure()

    # Act
    calculator.reset_calculator()

    # Assert
    assert calculator.get_detector_ids() == []
    assert calculator._pole_figure_dict == {}


def test_add_unknown_log_name_raises(calculator):
    """A log name the calculator does not know is refused."""
    # Arrange
    logs = _point_logs([2 * THETA], [THETA], [0.0], [0.0], [0.0], [1.0])
    logs["psi"] = np.array([0.0])

    # Act / Assert
    with pytest.raises(RuntimeError):
        calculator.add_input_data_set(1, logs)


def test_add_negative_peak_id_raises(calculator):
    """Peak ids start at 0."""
    # Act / Assert
    with pytest.raises(ValueError):
        calculator.add_input_data_set(-1, _point_logs([2 * THETA], [THETA], [0.0], [0.0], [0.0], [1.0]))


# --- export ------------------------------------------------------------------------------------


def test_export_mtex_writes_header_and_points(calculator, tmp_path):
    """The MTEX file is HB2B_<run>_<peak>.jul: two header lines, the user header, then α β intensity."""
    # Arrange
    calculator.add_input_data_set(
        1, _point_logs([2 * THETA] * 2, [THETA, THETA + 10.0], [30.0, 0.0], [0.0] * 2, [0.0] * 2, [4.0, 9.5])
    )
    calculator.calculate_pole_figure()

    # Act
    calculator.export_pole_figure(
        output_folder=str(tmp_path), peak_id_list=[1], peak_name_list=["Peak_1"], run_number=1599, file_header="hdr"
    )

    # Assert
    lines = (tmp_path / "HB2B_1599_Peak_1.jul").read_text().splitlines()
    assert lines[:3] == ["NRSF2", "alpha beta intensity", "hdr"]
    np.testing.assert_allclose(np.loadtxt(lines[3:]), calculator.get_polefigure_array(1), atol=1e-5)


def test_export_mtex_writes_one_file_per_peak(calculator, tmp_path):
    """Each peak's .jul file holds that peak's points only."""
    # Arrange
    calculator.add_input_data_set(1, _point_logs([2 * THETA], [THETA], [30.0], [0.0], [0.0], [1.0]))
    calculator.add_input_data_set(
        2, _point_logs([2 * THETA] * 2, [THETA] * 2, [10.0, 50.0], [0.0] * 2, [0.0] * 2, [2.0, 3.0])
    )
    calculator.calculate_pole_figure()

    # Act
    calculator.export_pole_figure(
        output_folder=str(tmp_path), peak_id_list=[1, 2], peak_name_list=["Peak_1", "Peak_2"], run_number=1599
    )

    # Assert
    for peak, name in ((1, "Peak_1"), (2, "Peak_2")):
        rows = np.atleast_2d(np.loadtxt(tmp_path / f"HB2B_1599_{name}.jul", skiprows=3))
        np.testing.assert_allclose(rows, calculator.get_polefigure_array(peak), atol=1e-5)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DD-15: export_arrays_to_ascii sorts each column on its own (np.sort(axis=0)), so a row's "
    "alpha, beta and intensity no longer belong to the same point",
)
def test_export_ascii_keeps_each_point_together(tmp_path):
    """Every row of the ASCII export is one measured point's (α, β, intensity)."""
    # Arrange
    poles = np.array([[30.0, -90.0, 1.0], [10.0, 180.0, 3.0], [20.0, 45.0, 2.0]])

    # Act
    export_arrays_to_ascii({1: poles}, [1], str(tmp_path / "poles"))

    # Assert
    written = np.loadtxt(tmp_path / "poles.txt")
    assert sorted(map(tuple, written)) == sorted(map(tuple, poles))
