"""Neutron attenuation against published cross sections, and compositions typed as text."""

import pytest

from pyrs.pole_figures.core.neutron_xs import mu_per_mm, parse_composition

# ---------------------------------------------------------------- published neutron cross sections
# Typed from V. F. Sears, Neutron News 3 (1992) 26-37, as tabulated by NIST
# (https://www.ncnr.nist.gov/resources/n-lengths/): bound total scattering cross section (coherent + incoherent) and
# absorption cross section for 2200 m/s neutrons (1.798 A), in barn. Absorption scales as 1/v, so linearly with the
# wavelength. Densities and standard atomic weights are the handbook values for the pure metals.
AVOGADRO = 6.02214076e23  # 1/mol
LAMBDA_2200_A = 1.798  # wavelength of a 2200 m/s neutron (A)
SEARS = {
    # element: (scattering xs (b), absorption xs at 1.798 A (b), atomic weight (g/mol), density (g/cm3))
    "Fe": (11.62, 2.56, 55.845, 7.874),
    "Ni": (18.5, 4.49, 58.6934, 8.908),
    "Al": (1.503, 0.231, 26.9815, 2.699),
}


def hand_mu_per_mm(element: str, wavelength_A: float) -> float:
    """
    Hand calculation: mu = n (sigma_s + sigma_a lambda / 1.798), n = rho N_A / M atoms per cm3.

    Units: 1 b = 1e-24 cm2, and 1/cm = 0.1/mm.
    """
    sigma_s, sigma_a, weight, density = SEARS[element]
    n_per_cm3 = density * AVOGADRO / weight
    sigma_b = sigma_s + sigma_a * wavelength_A / LAMBDA_2200_A
    return n_per_cm3 * sigma_b * 1e-24 * 0.1


@pytest.mark.parametrize(
    ("element", "wavelength_A", "approx_mu"),
    [
        ("Fe", 1.54, 0.1173),  # 8.491e22 /cm3 x (11.62 + 2.56 x 1.54/1.798 = 13.81 b)
        ("Fe", 1.80, 0.1204),  # 8.491e22 /cm3 x (11.62 + 2.56 x 1.80/1.798 = 14.18 b)
        ("Ni", 1.54, 0.2042),  # 9.140e22 /cm3 x (18.5 + 4.49 x 1.54/1.798 = 22.35 b)
        ("Al", 1.54, 0.01025),  # 6.024e22 /cm3 x (1.503 + 0.231 x 1.54/1.798 = 1.701 b)
    ],
)
def test_mu_per_mm_matches_sears_cross_sections(element: str, wavelength_A: float, approx_mu: float) -> None:
    # Arrange
    density = SEARS[element][3]
    want = hand_mu_per_mm(element, wavelength_A)

    # Act
    mu = mu_per_mm({element: 100.0}, density, wavelength_A)

    # Assert
    assert want == pytest.approx(approx_mu, rel=2e-3)  # the hand value in the comment
    assert mu == pytest.approx(want, rel=1e-3)


def test_mu_per_mm_absorption_scales_as_one_over_v() -> None:
    """Only absorption depends on the wavelength: mu(l2) - mu(l1) = n sigma_a (l2 - l1) / 1.798."""
    # Arrange
    sigma_a, weight, density = SEARS["Fe"][1], SEARS["Fe"][2], SEARS["Fe"][3]
    n_per_cm3 = density * AVOGADRO / weight
    want = n_per_cm3 * sigma_a * (2.67 - 1.45) / LAMBDA_2200_A * 1e-24 * 0.1

    # Act
    diff = mu_per_mm({"Fe": 100.0}, density, 2.67) - mu_per_mm({"Fe": 100.0}, density, 1.45)

    # Assert
    assert diff == pytest.approx(want, rel=1e-3)


def test_mu_per_mm_weights_elements_by_mass() -> None:
    """A 50/50 wt% Fe-Ni mixture at a given density: each element contributes by its own atoms per cm3."""
    # Arrange
    density = 8.3
    want = sum(
        density * 0.5 * AVOGADRO / SEARS[el][2] * (SEARS[el][0] + SEARS[el][1] * 1.54 / LAMBDA_2200_A) * 1e-25
        for el in ("Fe", "Ni")
    )

    # Act
    mu = mu_per_mm({"Fe": 50.0, "Ni": 50.0}, density, 1.54)

    # Assert
    assert mu == pytest.approx(want, rel=1e-3)


def test_mu_per_mm_element_without_data_raises() -> None:
    # Arrange
    composition = {"Rn": 100.0}

    # Act / Assert
    with pytest.raises(ValueError, match="no neutron cross sections for element 'Rn'"):
        mu_per_mm(composition, 1.0, 1.54)


def test_parse_composition_reads_pairs() -> None:
    # Arrange
    text = "ni:52.5, Cr:19 ,Fe:18.5"

    # Act
    composition = parse_composition(text)

    # Assert
    assert composition == {"Ni": 52.5, "Cr": 19.0, "Fe": 18.5}


def test_parse_composition_without_colon_raises() -> None:
    # Arrange
    text = "Ni 50"

    # Act / Assert
    with pytest.raises(ValueError, match="expected"):
        parse_composition(text)


def test_parse_composition_unknown_element_raises() -> None:
    # Arrange
    text = "Xx:10"

    # Act / Assert
    with pytest.raises(ValueError, match="not in the neutron"):
        parse_composition(text)
