"""Mount geometry: beams, rotation conventions, poles, mount matrices and the equal-area projection."""

import numpy as np
import pytest

from pyrs.pole_figures.core.geometry import (
    Mount,
    MountConventions,
    format_matrix,
    from_view,
    parse_matrix,
    project,
    rot,
    to_view,
)

CONV = MountConventions(incident_from="+vy")
#: the cylinder mount: turned about its long axis vz by phi
CYLINDER = MountConventions(up_axis="+vz", q_axis="+vx")
#: a tilted mount typed to 4 decimals
TILTED = (0, 0.2588, 0.9659, 0, 0.9659, -0.2588, -1, 0, 0)


# ---------------------------------------------------------------- rotations
def test_rot_is_right_handed() -> None:
    # Arrange
    R = rot([0, 0, 1], 90)

    # Act
    x_turned = R @ [1, 0, 0]

    # Assert
    np.testing.assert_allclose(x_turned, [0, 1, 0], atol=1e-15)


def test_rot_is_orthogonal() -> None:
    # Arrange
    R = rot([1, 2, 3], 37)

    # Act
    product = R @ R.T

    # Assert
    np.testing.assert_allclose(product, np.eye(3), atol=1e-15)


# ---------------------------------------------------------------- beams and poles
@pytest.mark.parametrize("side", ["+vy", "-vy"])
def test_incident_beam_follows_beam_side(side: str) -> None:
    # Arrange
    sb = -1 if side == "+vy" else 1
    w = np.radians(28.5)

    # Act
    m = Mount(MountConventions(incident_from=side), omega=28.5, tth=57.0)  # type: ignore[arg-type]

    # Assert
    np.testing.assert_allclose(m.ki, [0, sb * np.cos(w), -np.sin(w)])


@pytest.mark.parametrize("side", ["+vy", "-vy"])
def test_symmetric_reflection_puts_q_along_vz(side: str) -> None:
    # Arrange
    m = Mount(MountConventions(incident_from=side), omega=28.5, tth=57.0)  # type: ignore[arg-type]

    # Act
    pole = m.pole(0, 0)

    # Assert
    np.testing.assert_allclose(m.q, [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(pole, [0, 0, 1], atol=1e-15)


@pytest.mark.parametrize("phi", [0, 45, 200])
def test_pole_chi90_puts_vx_along_q(phi: float) -> None:
    # Arrange
    m = Mount(MountConventions(), omega=56.0, tth=112.0)

    # Act
    pole = m.pole(90, phi)

    # Assert
    np.testing.assert_allclose(pole, [1, 0, 0], atol=1e-12)


def test_pole_vx_down_reverses_vx_and_vy() -> None:
    # Arrange
    up = Mount(MountConventions(vx_up=True), omega=45, tth=90)
    down = Mount(MountConventions(vx_up=False), omega=45, tth=90)

    # Act
    p_up, p_down = up.pole(30, 40), down.pole(30, 40)

    # Assert
    np.testing.assert_allclose(p_down, p_up * [-1, -1, 1], atol=1e-12)


def test_pole_phi_sense_mirrors_phi() -> None:
    # Arrange
    cw = Mount(MountConventions(phi_sense="clockwise_from_above"), omega=45, tth=90)
    ccw = Mount(MountConventions(phi_sense="counterclockwise_from_above"), omega=45, tth=90)

    # Act
    p_cw, p_ccw_negative = cw.pole(0, 30), ccw.pole(0, -30)

    # Assert
    np.testing.assert_allclose(p_cw, p_ccw_negative, atol=1e-12)


def test_pole_phi_sense_changes_the_pole() -> None:
    # Arrange
    cw = Mount(MountConventions(phi_sense="clockwise_from_above"), omega=45, tth=90)
    ccw = Mount(MountConventions(phi_sense="counterclockwise_from_above"), omega=45, tth=90)

    # Act
    vy_cw, vy_ccw = cw.pole(0, 30)[1], ccw.pole(0, 30)[1]

    # Assert
    assert vy_cw != pytest.approx(vy_ccw)


@pytest.mark.parametrize(
    "conv",
    [MountConventions(), CONV, MountConventions(vx_up=False), CYLINDER, MountConventions(sample_to_instrument=TILTED),
     MountConventions(phi_sense="counterclockwise_from_above", chi_tilts_up_axis_toward_Q=False)],
    ids=["default", "plus_vy", "vx_down", "cylinder", "tilted", "ccw_chi_reversed"],
)  # fmt: skip
def test_pole_is_unit_vector(conv: MountConventions) -> None:
    # Arrange
    rng = np.random.default_rng(11)
    angles = rng.uniform(-180.0, 180.0, (50, 2))
    m = Mount(conv, omega=31.0, tth=73.0)

    # Act
    norms = [np.linalg.norm(m.pole(chi, phi)) for chi, phi in angles]

    # Assert
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_poles_vectorized_equal_pole() -> None:
    # Arrange
    m = Mount(CYLINDER, omega=45, tth=90)

    # Act
    poles = m.poles([0, 30, 90], [10, 20, 30])

    # Assert
    np.testing.assert_allclose(poles, [m.pole(0, 10), m.pole(30, 20), m.pole(90, 30)], atol=1e-15)


def test_mirrored_pole_at_phi0_is_vy_reflection() -> None:
    # Arrange
    rng = np.random.default_rng(5)
    m, mirror = Mount(CONV, omega=30, tth=70), Mount(CONV.mirrored(), omega=30, tth=70)

    # Act
    pairs = [(m.pole(chi, 0.0), mirror.pole(chi, 0.0)) for chi in rng.uniform(-90.0, 90.0, 20)]

    # Assert
    for pole, mirrored in pairs:
        np.testing.assert_allclose(mirrored, pole * [1, -1, 1], atol=1e-12)


def test_mirrored_pole_is_vy_reflection_at_negative_phi() -> None:
    """A reflection across vy turns the phi rotation (about e_x) the other way, so phi changes sign."""
    # Arrange
    rng = np.random.default_rng(6)
    m, mirror = Mount(CONV, omega=30, tth=70), Mount(CONV.mirrored(), omega=30, tth=70)

    # Act
    pairs = [(m.pole(chi, phi), mirror.pole(chi, -phi)) for chi, phi in rng.uniform(-180.0, 180.0, (20, 2))]

    # Assert
    for pole, mirrored in pairs:
        np.testing.assert_allclose(mirrored, pole * [1, -1, 1], atol=1e-12)


# ---------------------------------------------------------------- eta slices
@pytest.mark.parametrize("eta", [-12.0, 7.5, 20.0])
def test_eta_kf_is_unit_vector(eta: float) -> None:
    # Arrange
    tth = 85.0

    # Act
    m = Mount(CONV, omega=40.0, tth=tth, eta_deg=eta)

    # Assert
    assert np.linalg.norm(m.kf) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("eta", [-12.0, 7.5, 20.0])
def test_eta_kf_keeps_two_theta_to_ki(eta: float) -> None:
    # Arrange
    tth = 85.0

    # Act
    m = Mount(CONV, omega=40.0, tth=tth, eta_deg=eta)

    # Assert
    angle = np.degrees(np.arccos(np.clip(m.ki @ m.kf, -1, 1)))
    assert angle == pytest.approx(tth, abs=1e-9)


@pytest.mark.parametrize(("eta", "sign"), [(-12.0, 1.0), (7.5, 1.0), (20.0, -1.0)])
def test_eta_kf_leaves_plane_by_sin_eta(eta: float, sign: float) -> None:
    """On the Debye cone kf . e_x = sin(2theta) sin(eta), with the eta sign of the mount."""
    # Arrange
    tth = 85.0
    conv = MountConventions(incident_from="+vy", eta_up_sign=sign)

    # Act
    m = Mount(conv, omega=40.0, tth=tth, eta_deg=eta)

    # Assert
    assert m.kf[0] == pytest.approx(np.sin(np.radians(tth)) * np.sin(np.radians(sign * eta)), abs=1e-12)


def test_eta_q_is_unit_vector() -> None:
    # Arrange
    m = Mount(CONV, omega=40.0, tth=85.0, eta_deg=15.0)

    # Act
    q = m.q

    # Assert
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)


def test_eta_q_bisects_the_beams() -> None:
    # Arrange
    m = Mount(CONV, omega=40.0, tth=85.0, eta_deg=15.0)

    # Act
    q = m.q

    # Assert
    assert q @ m.kf == pytest.approx(-(q @ m.ki), abs=1e-12)


# ---------------------------------------------------------------- conventions
def test_cradle_chi_adds_offset() -> None:
    # Arrange
    c = MountConventions(chi_offset_deg=90.0)

    # Act
    chi = c.cradle_chi(-90.0)

    # Assert
    assert chi == 0.0


def test_logged_chi_inverts_cradle_chi() -> None:
    # Arrange
    c = MountConventions(chi_offset_deg=90.0)

    # Act
    back = c.logged_chi(c.cradle_chi(12.5))

    # Assert
    assert back == 12.5


def test_mirrored_flips_only_the_beam_side() -> None:
    # Arrange
    c = MountConventions(incident_from="+vy", chi_offset_deg=5)

    # Act
    mirrored = c.mirrored()

    # Assert
    assert mirrored == MountConventions(incident_from="-vy", chi_offset_deg=5)


def test_from_dict_misspelled_key_raises() -> None:
    # Arrange
    d = {"phi_sence": "clockwise_from_above"}

    # Act / Assert
    with pytest.raises(ValueError, match=r"phi_sence \(did you mean phi_sense\?\)"):
        MountConventions.from_dict(d)


def test_from_dict_ignores_note_keys() -> None:
    # Arrange
    d = {"_note": "a note", "incident_from": "+vy"}

    # Act
    conv = MountConventions.from_dict(d)

    # Assert
    assert conv.incident_from == "+vy"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"phi_sense": "clockwise"}, "phi_sense must be one of"),
        ({"vx_up": "false"}, "vx_up must be true or false"),
        ({"eta_up_sign": 0.0}, "eta_up_sign"),
        ({"up_axis": "+vz", "q_axis": "-vz"}, "different"),
        ({"sample_to_instrument": (1, 0, 0, 0, 2, 0, 0, 0, 1)}, "not a rotation"),
        ({"sample_to_instrument": (-1, 0, 0, 0, 1, 0, 0, 0, 1)}, "reflection"),
    ],
    ids=["phi_sense", "vx_up", "eta_up_sign", "parallel_axes", "not_rotation", "reflection"],
)
def test_conventions_bad_value_raises(kwargs: dict, message: str) -> None:
    # Arrange
    bad = kwargs

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        MountConventions(**bad)


# ---------------------------------------------------------------- mount matrices
def test_default_mount_is_identity() -> None:
    # Arrange
    conv = MountConventions()

    # Act
    M0 = conv.M0

    # Assert
    assert np.array_equal(M0, np.eye(3))


@pytest.mark.parametrize("conv", [MountConventions(vx_up=False), MountConventions(up_axis="-vx")],
                         ids=["vx_up_false", "up_axis_minus_vx"])  # fmt: skip
def test_vx_down_mount_reverses_vx_and_vy(conv: MountConventions) -> None:
    # Arrange
    want = np.diag([-1.0, -1.0, 1.0])

    # Act
    M0 = conv.M0

    # Assert
    assert np.array_equal(M0, want)


def test_axes_at_zero_names_cylinder_axes() -> None:
    # Arrange
    conv = CYLINDER

    # Act
    axes = conv.axes_at_zero()

    # Assert
    assert axes == {"up": "+vz", "across": "-vy", "Q": "+vx"}


def test_cylinder_mount_is_proper_rotation() -> None:
    # Arrange
    conv = CYLINDER

    # Act
    det = np.linalg.det(conv.M0)

    # Assert
    assert det == pytest.approx(1.0)


@pytest.mark.parametrize("phi", [0.0, 37.0, 180.0])
def test_cylinder_chi0_poles_lie_on_vz_equator(phi: float) -> None:
    # Arrange
    m = Mount(CYLINDER, omega=45, tth=90)  # omega = theta: Q along e_z at chi 0

    # Act
    pole = m.pole(0.0, phi)

    # Assert
    assert abs(pole[2]) < 1e-12


@pytest.mark.parametrize("phi", [0.0, 37.0, 180.0])
def test_cylinder_chi90_pole_is_phi_axis(phi: float) -> None:
    # Arrange
    m = Mount(CYLINDER, omega=45, tth=90)

    # Act
    pole = m.pole(90.0, phi)

    # Assert
    np.testing.assert_allclose(np.abs(pole), [0, 0, 1], atol=1e-12)


def test_matrix_mount_equals_axis_mount() -> None:
    # Arrange
    mat = parse_matrix("0 0 1; 0 -1 0; 1 0 0")  # rows: up, across, Q in vx, vy, vz

    # Act
    conv = MountConventions(sample_to_instrument=mat)

    # Assert
    assert mat is not None
    assert np.array_equal(conv.M0, CYLINDER.M0)


def test_format_matrix_writes_rows() -> None:
    # Arrange
    M0 = CYLINDER.M0

    # Act
    text = format_matrix(M0)

    # Assert
    assert text == "0 0 1; 0 -1 0; 1 0 0"


def test_typed_tilted_matrix_is_made_orthonormal() -> None:
    # Arrange
    typed = TILTED

    # Act
    conv = MountConventions(sample_to_instrument=typed)

    # Assert
    np.testing.assert_allclose(conv.M0 @ conv.M0.T, np.eye(3), atol=1e-12)


def test_axes_at_zero_marks_tilted_axis() -> None:
    # Arrange
    conv = MountConventions(sample_to_instrument=TILTED)

    # Act
    up = conv.axes_at_zero()["up"]

    # Assert
    assert up == "~+vz"


def test_parse_matrix_two_rows_raises() -> None:
    # Arrange
    text = "1 0 0; 0 1 0"

    # Act / Assert
    with pytest.raises(ValueError, match="3 rows"):
        parse_matrix(text)


def test_parse_matrix_blank_is_none() -> None:
    # Arrange
    text = "  "

    # Act
    mat = parse_matrix(text)

    # Assert
    assert mat is None


# ---------------------------------------------------------------- projection
def test_project_radii() -> None:
    # Arrange
    h = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1]]

    # Act
    xy = project(h)

    # Assert
    np.testing.assert_allclose(np.hypot(*xy.T), [0, 1, 1, 0], atol=1e-12)


@pytest.mark.parametrize(
    ("h", "view", "want"),
    [([1, 0, 0], "+vz", [1, 0]), ([1, 0, 0], "-vz", [-1, 0]), ([0, 1, 0], "+vx", [1, 0])],
    ids=["plus_vz_vx_right", "minus_vz_vx_left", "plus_vx_vy_right"],
)
def test_project_view_orientation(h: list[float], view: str, want: list[float]) -> None:
    # Arrange
    vector = h

    # Act
    xy = project(vector, view=view)  # type: ignore[arg-type]

    # Assert
    np.testing.assert_allclose(xy[0], want, atol=1e-12)


def test_project_is_equal_area() -> None:
    # Arrange
    a = np.radians(45)  # 45 deg from the center

    # Act
    r = np.hypot(*project([np.sin(a), 0, np.cos(a)])[0])

    # Assert
    assert r == pytest.approx(np.sqrt(2) * np.sin(a / 2))


def test_from_view_inverts_to_view() -> None:
    # Arrange
    h = np.array([[0.2, 0.3, 0.9], [1.0, 0.0, 0.0]])

    # Act
    back = from_view(to_view(h, view="+vx"), view="+vx")

    # Assert
    np.testing.assert_allclose(back, h)


def test_project_unknown_view_raises() -> None:
    # Arrange
    view = "vz"

    # Act / Assert
    with pytest.raises(ValueError, match=r"view must be one of \+vz, -vz, \+vx, got 'vz'"):
        project([0.0, 0.0, 1.0], view=view)  # type: ignore[arg-type]
