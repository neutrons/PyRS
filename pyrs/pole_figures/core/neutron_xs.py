"""
Neutron attenuation of a mixture from the packaged cross-section table, and compositions as text.

- Attenuation: Beer-Lambert for a mixture, mu = sum of n_i (sigma_scatt + sigma_abs(lambda)) over the elements,
  with the absorption cross section scaled linearly with wavelength from the 1.798 A reference.
- Neutron data: natural-abundance coherent scattering lengths and cross sections from the NIST table (Sears,
  Neutron News 3 (1992) 26), packaged as ``data/isotope_xs_db.csv``. Elements without data there are refused.

Typical workflow:

1. Read a composition typed as text with :func:`parse_composition` (or build the dict).
2. Get the attenuation with :func:`mu_per_mm`.

Example:
    >>> round(mu_per_mm({"Fe": 100.0}, 7.874, 1.54), 4)
    0.1173
    >>> parse_composition("Ni:52.5, Cr:19")
    {'Ni': 52.5, 'Cr': 19.0}
"""

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources

import gemmi
import numpy as np

#: the package's data folder, found from this module's own package (core -> the package root)
DATA_PACKAGE = __name__.rsplit(".", 2)[0] + ".data"

_LAMBDA_REF_A = 1.798  # 2200 m/s neutron
AVOGADRO = 6.02214076e23  # 1/mol


@dataclass(frozen=True)
class ElementXS:
    """
    Natural-abundance neutron data for one element (NIST, Sears 1992). NaN where the table has no value.

    Attributes:
        b_coh_fm: Coherent scattering length (fm), complex for absorbing elements.
        scatt_xs_b: Total scattering cross section (barn).
        abs_xs_b: Absorption cross section at 1.798 Å (barn).
    """

    b_coh_fm: complex
    scatt_xs_b: float
    abs_xs_b: float


def _num(s: str) -> float:
    """A number from the table, NaN when it has none (so a missing value is never used as 0)."""
    try:
        return float(s)
    except ValueError:
        return math.nan


def _cnum(s: str) -> complex:
    try:
        return complex(s.strip().replace("i", "j"))
    except ValueError:
        return complex(math.nan, math.nan)


def _known(value: complex | float) -> bool:
    return bool(np.isfinite(value))


@cache
def scattering_table() -> dict[str, ElementXS]:
    """
    Natural elements from the packaged isotope table (cached).

    Returns:
        Element symbol -> neutron data, for the rows whose symbol is purely alphabetic (no isotope number).
    """
    text = resources.files(DATA_PACKAGE).joinpath("isotope_xs_db.csv").read_text(encoding="utf-8")
    out: dict[str, ElementXS] = {}
    for row in csv.DictReader(text.splitlines()):
        sym = row["Isotope"].strip()
        if sym.isalpha():
            out[sym] = ElementXS(_cnum(row["coh_b_fm"]), _num(row["scatt_xs_b"]), _num(row["abs_xs_b"]))
    return out


def atomic_mass(symbol: str) -> float:
    """
    Standard atomic weight.

    Args:
        symbol: Element symbol.

    Returns:
        The atomic weight (g/mol).

    Raises:
        ValueError: If gemmi does not know the element.
    """
    w = float(gemmi.Element(symbol).weight)
    if w <= 0:
        raise ValueError(f"unknown element '{symbol}'")
    return w


def _check_elements(composition: Mapping[str, float]) -> None:
    table = scattering_table()
    for el, wt in composition.items():
        if wt < 0:
            raise ValueError(f"weight percent for {el} is negative ({wt})")
        if el not in table:
            raise ValueError(f"element '{el}' is not in the neutron cross-section table")
        if not (_known(table[el].scatt_xs_b) and _known(table[el].abs_xs_b)):
            raise ValueError(f"no neutron cross sections for element '{el}' in the table, so mu cannot be computed")
    if sum(composition.values()) <= 0:
        raise ValueError(f"the composition has no weight, got {composition!r}")


def mu_per_mm(composition: Mapping[str, float], density_g_cm3: float, wavelength_A: float) -> float:
    """
    Linear attenuation coefficient of a mixture: scattering plus wavelength-scaled absorption.

    Args:
        composition: Element symbol -> weight percent (normalized internally).
        density_g_cm3: Bulk density (g/cm3).
        wavelength_A: Neutron wavelength (Å).

    Returns:
        mu (1/mm).

    Raises:
        ValueError: If the density or wavelength is not positive, a weight is negative, the weights sum to zero, or
            an element has no cross sections in the table.

    Example:
        >>> round(mu_per_mm({"Fe": 100.0}, 7.874, 1.54), 4)
        0.1173
    """
    if density_g_cm3 <= 0 or wavelength_A <= 0:
        raise ValueError(f"density and wavelength must be positive, got density {density_g_cm3!r} g/cm3 and "
                         f"wavelength {wavelength_A!r} A")  # fmt: skip
    _check_elements(composition)
    table = scattering_table()
    total = sum(composition.values())
    sigma_cm = 0.0
    for el, wt in composition.items():
        xs = table[el]
        sigma_tot = xs.scatt_xs_b + xs.abs_xs_b * (wavelength_A / _LAMBDA_REF_A)
        sigma_cm += density_g_cm3 * (wt / total) * AVOGADRO / atomic_mass(el) * sigma_tot * 1e-24
    return sigma_cm * 0.1


def atomic_fractions(composition: Mapping[str, float]) -> dict[str, float]:
    """
    Weight percent to atomic fraction.

    Args:
        composition: Element symbol -> weight percent (any total).

    Returns:
        Element symbol -> atomic fraction (summing to 1).
    """
    n = {el: wt / atomic_mass(el) for el, wt in composition.items()}
    s = sum(n.values())
    return {el: v / s for el, v in n.items()}


def parse_composition(text: str) -> dict[str, float]:
    """
    Read a composition typed as text.

    Args:
        text: ``Element:weight_percent`` pairs split by commas, e.g. ``"Ni:52.5, Cr:19, Fe:18.5"``. Symbols are
            title-cased and repeated elements are added up.

    Returns:
        Element symbol -> weight percent, e.g. ``{"Ni": 52.5, "Cr": 19.0, "Fe": 18.5}``.

    Raises:
        ValueError: If a pair has no colon or no number, the composition is empty, or an element is not usable (see
            :func:`mu_per_mm`).
    """
    out: dict[str, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"'{token}': expected Element:weight_percent, e.g. Fe:97.0")
        el, wt = token.split(":", 1)
        sym = el.strip().title()
        try:
            out[sym] = out.get(sym, 0.0) + float(wt)
        except ValueError as exc:
            raise ValueError(f"'{wt.strip()}' is not a number (element {sym})") from exc
    if not out:
        raise ValueError(f"the composition is empty, got {text!r}")
    _check_elements(out)
    return out


def format_composition(composition: Mapping[str, float]) -> str:
    """
    Write a composition as text that :func:`parse_composition` reads back.

    Args:
        composition: Element symbol -> weight percent.

    Returns:
        E.g. ``"Ni:52.5, Cr:19, Fe:18.5"``.
    """
    return ", ".join(f"{el}:{wt:g}" for el, wt in composition.items())
