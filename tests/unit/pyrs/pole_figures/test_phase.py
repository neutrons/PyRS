"""Crystal phases: allowed reflections, 2theta, a phase's attenuation, CIF import, input checks and presets."""

from dataclasses import replace
from pathlib import Path

import pytest

from pyrs.pole_figures.core.neutron_xs import mu_per_mm
from pyrs.pole_figures.core.phase import Phase, Site, presets
from pyrs.pole_figures.core.reflections import MAX_HKL_INDEX, hkl_label

LAM_REF = 1.7212999080420324  # calibrated wavelength of the reference runs HB2B_6494-6497.h5


def labels(p: Phase, lam: float = 1.73, n: int = 6) -> list[str]:
    return [r.label for r in p.reflections(lam)[:n]]


def test_in718_mu_regression() -> None:
    """
    Regression check, not an independent one: the value in the reference pole_figure_config.json.

    That 0.140 /mm came from the same cross-section table, so it guards against changes, not against a wrong table.
    The independent checks are the Sears tests above.
    """
    # Arrange
    in718 = presets()["IN718 (fcc)"]

    # Act
    mu = in718.mu_per_mm(LAM_REF)

    # Assert
    assert mu == pytest.approx(0.140, abs=0.002)


# ---------------------------------------------------------------- reflections
def test_in718_first_reflections() -> None:
    # Arrange
    p = presets()["IN718 (fcc)"]

    # Act
    r = p.reflections(LAM_REF)

    # Assert
    assert [x.label for x in r[:5]] == ["111", "200", "220", "311", "222"]


def test_in718_multiplicities() -> None:
    # Arrange
    p = presets()["IN718 (fcc)"]

    # Act
    r = p.reflections(LAM_REF)

    # Assert
    assert [x.multiplicity for x in r[:5]] == [8, 6, 12, 24, 8]


@pytest.mark.parametrize(("hkl", "arm"), [("222", 111.99), ("311", 105.00), ("220", 85.10), ("200", 57.00)])
def test_in718_two_theta_matches_reference_arm(hkl: str, arm: float) -> None:
    """The detector arm 2theta of runs 6494-6497 sits on each reflection."""
    # Arrange
    p = presets()["IN718 (fcc)"]

    # Act
    tth = {x.label: x.two_theta for x in p.reflections(LAM_REF)}

    # Assert
    assert tth[hkl] == pytest.approx(arm, abs=0.25)


def test_bcc_series() -> None:
    # Arrange
    p = presets()["Ferritic steel (bcc)"]

    # Act
    got = labels(p, n=5)

    # Assert
    assert got == ["110", "200", "211", "220", "310"]


def test_hcp_series() -> None:
    # Arrange
    ti = presets()["Titanium (hcp)"]

    # Act
    got = labels(ti)

    # Assert
    assert got == ["100", "002", "101", "102", "110", "103"]


def test_d_spacing_hcp_002_is_half_c() -> None:
    # Arrange
    ti = presets()["Titanium (hcp)"]

    # Act
    d = ti.d_spacing((0, 0, 2))

    # Assert
    assert d == pytest.approx(4.6855 / 2)


def test_structure_factor_removes_special_position_extinctions() -> None:
    """Diamond Si: {200} and {222} pass the Fd-3m conditions for general positions but F = 0 for site 8a."""
    # Arrange
    si = Phase.from_dict(
        {
            "name": "Si",
            "space_group": "F d -3 m :1",
            "cell": [5.431],
            "sites": [{"element": "Si", "x": 0, "y": 0, "z": 0}],
            "composition": {"Si": 100},
            "density_g_cm3": 2.33,
        }  # fmt: skip
    )

    # Act
    got = labels(si, lam=1.54, n=5)

    # Assert
    assert got == ["111", "220", "311", "400", "331"]


def test_two_site_structure_keeps_odd_reflections() -> None:
    """Rock salt: odd hkl survive because b(Na) != b(Cl)."""
    # Arrange
    nacl = Phase.from_dict(
        {
            "name": "NaCl",
            "space_group": "Fm-3m",
            "cell": [5.64],
            "sites": [{"element": "Na", "x": 0, "y": 0, "z": 0}, {"element": "Cl", "x": 0.5, "y": 0.5, "z": 0.5}],
            "composition": {"Na": 39.34, "Cl": 60.66},
            "density_g_cm3": 2.165,
        }  # fmt: skip
    )

    # Act
    got = labels(nacl, n=4)

    # Assert
    assert got == ["111", "200", "220", "311"]


def test_reflections_stop_before_backscattering() -> None:
    # Arrange
    p = presets()["IN718 (fcc)"]

    # Act
    r = p.reflections(2.67)

    # Assert
    assert all(x.two_theta < 179.0 for x in r)


def test_reflections_fewer_at_long_wavelength() -> None:
    # Arrange
    p = presets()["IN718 (fcc)"]

    # Act
    n_long, n_short = len(p.reflections(2.67)), len(p.reflections(1.45))

    # Assert
    assert n_long < n_short


# ---------------------------------------------------------------- phases from dicts and CIF files
def test_from_dict_zero_density_raises() -> None:
    # Arrange
    d = {"name": "Cu", "space_group": "Fm-3m", "cell": [3.615], "composition": {"Cu": 100.0}, "density_g_cm3": 0.0}

    # Act / Assert
    with pytest.raises(ValueError, match="density must be positive"):
        Phase.from_dict(d)


def test_from_dict_without_composition_raises() -> None:
    # Arrange
    d = {"name": "Cu", "space_group": "Fm-3m", "cell": [3.615], "density_g_cm3": 8.96}

    # Act / Assert
    with pytest.raises(ValueError, match="lacks composition"):
        Phase.from_dict(d)


CU_CIF = """data_Cu
_cell_length_a 3.6149
_cell_length_b 3.6149
_cell_length_c 3.6149
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Cu1 Cu 0 0 0 1
"""

SI_ORIGIN2 = """_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F d -3 m'
_space_group_IT_coordinate_system_code 2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0.125 0.125 0.125 1
"""


def write_cif(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def copper(tmp_path: Path) -> Phase:
    return Phase.from_cif(write_cif(tmp_path / "cu.cif", CU_CIF))


def test_from_cif_composition_is_pure_copper(copper: Phase) -> None:
    # Arrange
    phase = copper

    # Act
    composition = phase.composition

    # Assert
    assert composition == pytest.approx({"Cu": 100.0})


def test_from_cif_density_from_cell(copper: Phase) -> None:
    # Arrange
    phase = copper

    # Act
    density = phase.density_g_cm3

    # Assert
    assert density == pytest.approx(8.935, abs=0.01)


def test_from_cif_reflections_match_copper_preset(copper: Phase) -> None:
    # Arrange
    preset = presets()["Copper (fcc)"]

    # Act
    got = labels(copper)

    # Assert
    assert got == labels(preset)


def test_from_cif_mu_uses_composition_and_density(copper: Phase) -> None:
    # Arrange
    want = mu_per_mm({"Cu": 100}, copper.density_g_cm3, 1.54)

    # Act
    mu = copper.mu_per_mm(1.54)

    # Assert
    assert mu == pytest.approx(want)


def test_from_cif_without_atoms_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_cif(tmp_path / "bad.cif", "data_x\n_cell_length_a 3\n")

    # Act / Assert
    with pytest.raises(ValueError, match="bad.cif"):
        Phase.from_cif(path)


@pytest.fixture
def silicon_origin2(tmp_path: Path) -> Phase:
    return Phase.from_cif(write_cif(tmp_path / "si.cif", "data_x\n" + SI_ORIGIN2))


def test_from_cif_keeps_origin_choice_2(silicon_origin2: Phase) -> None:
    """The plain H-M name would pick origin 1 (16 atoms per cell)."""
    # Arrange
    phase = silicon_origin2

    # Act
    space_group = phase.space_group

    # Assert
    assert space_group == "F d -3 m:2"


def test_from_cif_origin_choice_2_density(silicon_origin2: Phase) -> None:
    # Arrange
    phase = silicon_origin2

    # Act
    density = phase.density_g_cm3

    # Assert
    assert density == pytest.approx(2.329, abs=0.005)


def test_from_cif_origin_choice_2_extinctions(silicon_origin2: Phase) -> None:
    """Diamond Si in origin choice 2: 8 atoms per cell, so {222} and {200} are absent."""
    # Arrange
    phase = silicon_origin2

    # Act
    got = [r.label for r in phase.reflections(1.54)]

    # Assert
    assert got[:4] == ["111", "220", "311", "400"]
    assert "222" not in got and "200" not in got


def test_from_cif_without_cell_raises(tmp_path: Path) -> None:
    # Arrange
    body = SI_ORIGIN2.split("_symmetry_space_group_name_H-M")[1]
    path = write_cif(tmp_path / "nocell.cif", "data_x\n_symmetry_space_group_name_H-M" + body)

    # Act / Assert
    with pytest.raises(ValueError, match="no unit cell"):
        Phase.from_cif(path)


# ---------------------------------------------------------------- input checks
CUBIC = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)


def cu_phase(cell: tuple[float, ...] = CUBIC) -> Phase:
    return Phase(name="Cu", space_group="Fm-3m", cell=cell, sites=(Site(element="Cu", x=0.0, y=0.0, z=0.0),),  # type: ignore[arg-type]
                 composition={"Cu": 100.0}, density_g_cm3=8.96)  # fmt: skip


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        ((3.6, 3.6, 3.6, 90.0, 90.0), "six finite numbers"),
        ((3.6, 3.6, float("nan"), 90.0, 90.0, 90.0), "six finite numbers"),
        ((3.6, 0.0, 3.6, 90.0, 90.0, 90.0), "lengths must be positive"),
        ((3.6, 3.6, 3.6, 90.0, 180.0, 90.0), "angles must be between 0 and 180"),
        ((3.6, 3.6, 3.6, 0.0, 90.0, 90.0), "angles must be between 0 and 180"),
    ],
)
def test_phase_invalid_cell_raises(cell: tuple[float, ...], message: str) -> None:
    # Arrange
    bad = cell

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        cu_phase(bad)


@pytest.mark.parametrize("wavelength_A", [0.0, -1.54, float("nan")])
def test_reflections_bad_wavelength_raises(wavelength_A: float) -> None:
    # Arrange
    phase = cu_phase()

    # Act / Assert
    with pytest.raises(ValueError, match="wavelength must be positive"):
        phase.reflections(wavelength_A)


def test_reflections_huge_cell_is_refused() -> None:
    """A 100 A cell down to d 0.77 A needs hkl up to 131: refused instead of minutes of search."""
    # Arrange
    phase = cu_phase((100.0, 100.0, 100.0, 90.0, 90.0, 90.0))

    # Act / Assert
    with pytest.raises(ValueError, match=f"needs hkl up to 131, more than {MAX_HKL_INDEX}"):
        phase.reflections(1.54)


def test_from_dict_cell_of_three_values_raises() -> None:
    # Arrange
    d = {"name": "Cu", "space_group": "Fm-3m", "cell": [3.6, 3.6, 3.6], "composition": {"Cu": 100},
         "density_g_cm3": 8.96}  # fmt: skip

    # Act / Assert
    with pytest.raises(ValueError, match="cell must have 1 value .* or 6, got 3"):
        Phase.from_dict(d)


def test_from_dict_unknown_site_key_raises() -> None:
    # Arrange
    d = {"name": "Cu", "space_group": "Fm-3m", "cell": [3.6], "composition": {"Cu": 100}, "density_g_cm3": 8.96,
         "sites": [{"element": "Cu", "x": 0, "y": 0, "z": 0, "occ": 1.0}]}  # fmt: skip

    # Act / Assert
    with pytest.raises(ValueError, match="unknown: occ, missing: none"):
        Phase.from_dict(d)


def test_from_dict_site_without_coordinate_raises() -> None:
    # Arrange
    d = {"name": "Cu", "space_group": "Fm-3m", "cell": [3.6], "composition": {"Cu": 100}, "density_g_cm3": 8.96,
         "sites": [{"element": "Cu", "x": 0, "y": 0}]}  # fmt: skip

    # Act / Assert
    with pytest.raises(ValueError, match="unknown: none, missing: z"):
        Phase.from_dict(d)


def test_from_cif_unparsable_file_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_cif(tmp_path / "broken.cif", "data_x\nloop_\n_atom_site_label\n'unclosed\n")

    # Act / Assert
    with pytest.raises(ValueError, match="broken.cif: cannot read the CIF"):
        Phase.from_cif(path)


def test_from_cif_two_blocks_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_cif(tmp_path / "two.cif", CU_CIF + CU_CIF.replace("data_Cu", "data_Cu2"))

    # Act / Assert
    with pytest.raises(ValueError, match="two.cif: cannot read the CIF"):
        Phase.from_cif(path)


def test_from_cif_zero_occupancy_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_cif(tmp_path / "empty.cif", CU_CIF.replace("Cu1 Cu 0 0 0 1", "Cu1 Cu 0 0 0 0"))

    # Act / Assert
    with pytest.raises(ValueError, match="empty.cif: the atom occupancies in the CIF are all zero"):
        Phase.from_cif(path)


def test_presets_count() -> None:
    # Arrange
    table = presets()

    # Act
    names = list(table)

    # Assert
    assert len(names) >= 7


@pytest.mark.parametrize("name", sorted(presets()))
def test_preset_has_reflections(name: str) -> None:
    # Arrange
    p = presets()[name]

    # Act
    r = p.reflections(1.73)

    # Assert
    assert len(r) >= 4


@pytest.mark.parametrize("name", sorted(presets()))
def test_preset_mu_is_plausible(name: str) -> None:
    # Arrange
    p = presets()[name]

    # Act
    mu = p.mu_per_mm(1.73)

    # Assert
    assert 0 < mu < 1


@pytest.mark.parametrize(("hkl", "want"), [((3, 1, 1), "311"), ((1, -1, 0), "1 -1 0")])
def test_hkl_label(hkl: tuple[int, int, int], want: str) -> None:
    # Arrange
    index = hkl

    # Act
    label = hkl_label(index)

    # Assert
    assert label == want


def test_presets_are_read_only() -> None:
    # Arrange
    table = presets()

    # Act / Assert
    with pytest.raises(TypeError):
        table["new"] = table["IN718 (fcc)"]  # type: ignore[index]


def test_phase_composition_is_read_only_copy() -> None:
    # Arrange
    given = {"Cu": 100.0}
    phase = replace(cu_phase(), composition=given)

    # Act
    given["Cu"] = 50.0

    # Assert
    assert phase.composition["Cu"] == 100.0
    with pytest.raises(TypeError):
        phase.composition["Cu"] = 50.0  # type: ignore[index]


def test_phase_replace_checks_again() -> None:
    # Arrange
    phase = cu_phase()

    # Act / Assert
    with pytest.raises(ValueError, match="density must be positive"):
        replace(phase, density_g_cm3=0.0)
