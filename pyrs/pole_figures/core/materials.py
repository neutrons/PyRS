"""
Crystal phases: presets, CIF import, allowed reflections, 2theta at a wavelength, and neutron attenuation.

- Allowed reflections: every hkl that is not systematically absent in the space group and whose neutron structure
  factor (coherent scattering lengths, all sites in the cell) is not zero. Symmetry-equivalent hkl are grouped into
  one family with its multiplicity (Friedel pairs included).
- Attenuation: Beer-Lambert for a mixture, mu = sum of n_i (sigma_scatt + sigma_abs(lambda)) over the elements,
  with the absorption cross section scaled linearly with wavelength from the 1.798 A reference.
- Neutron data: natural-abundance coherent scattering lengths and cross sections from the NIST table (Sears,
  Neutron News 3 (1992) 26), packaged as ``data/isotope_xs_db.csv``. Elements without data there are refused.

Typical workflow:

1. Take a preset with :func:`presets`, read a CIF with :meth:`Phase.from_cif`, or build a :class:`Phase`.
2. List the reflections at the wavelength with :meth:`Phase.reflections` and keep those inside the HB-2B 2theta range
   (:func:`two_theta_allowed`).
3. Get the attenuation with :meth:`Phase.mu_per_mm`.

Example:
    >>> fe = Phase(name="Fe", space_group="Im-3m", cell=(2.8665, 2.8665, 2.8665, 90.0, 90.0, 90.0),
    ...            sites=(Site("Fe", 0.0, 0.0, 0.0),), composition={"Fe": 100.0}, density_g_cm3=7.874)
    >>> [r.label for r in fe.reflections(1.54)][:3]
    ['110', '200', '211']
    >>> round(fe.two_theta((2, 1, 1), 1.54), 2)
    82.29
    >>> round(fe.mu_per_mm(1.54), 4)
    0.1173
"""

import csv
import json
import math
import warnings
from dataclasses import dataclass, fields
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from pyrs.core.monosetting import MonoSetting

#: the package's data folder, found from this module's own package (core -> the package root)
DATA_PACKAGE = __name__.rsplit(".", 2)[0] + ".data"

#: HB-2B monochromator setting -> nominal wavelength (Å), from PyRS's
#: ``pyrs.core.monosetting.MonoSetting`` (every name, so Si511 is listed with the same wavelength as Si333). Calibrated
#: values per run come from the data files.
HB2B_WAVELENGTHS: dict[str, float] = {name: float(m.value) for name, m in MonoSetting.__members__.items()}

# 2theta range allowed for pole-figure reflections on HB-2B. Outside it the peak is not used.
TWO_THETA_MIN, TWO_THETA_MAX = 50.0, 120.0


def two_theta_allowed(two_theta: float) -> bool:
    """
    True when a reflection's 2theta is inside the allowed HB-2B range (inclusive).

    Args:
        two_theta: 2theta of the reflection (deg).

    Returns:
        Whether ``TWO_THETA_MIN <= two_theta <= TWO_THETA_MAX``.
    """
    return TWO_THETA_MIN <= two_theta <= TWO_THETA_MAX


#: the hkl search runs over -n..n in h, k and l with n = max(a, b, c) / d_min + 1. Above this n the search (n^3
#: Python iterations) takes minutes, so it is refused: lower two_theta_max or check the cell.
MAX_HKL_INDEX = 60

_LAMBDA_REF_A = 1.798  # 2200 m/s neutron
_NA = 6.02214076e23
AVERAGE = "avg"  # site element meaning "the composition-weighted average atom" (alloys on one site)


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


def _check_elements(composition: dict[str, float]) -> None:
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


def mu_per_mm(composition: dict[str, float], density_g_cm3: float, wavelength_A: float) -> float:
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
        sigma_cm += density_g_cm3 * (wt / total) * _NA / atomic_mass(el) * sigma_tot * 1e-24
    return sigma_cm * 0.1


def atomic_fractions(composition: dict[str, float]) -> dict[str, float]:
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


def format_composition(composition: dict[str, float]) -> str:
    """
    Write a composition as text that :func:`parse_composition` reads back.

    Args:
        composition: Element symbol -> weight percent.

    Returns:
        E.g. ``"Ni:52.5, Cr:19, Fe:18.5"``.
    """
    return ", ".join(f"{el}:{wt:g}" for el, wt in composition.items())


@dataclass(frozen=True)
class Site:
    """
    Atom site of the asymmetric unit.

    Attributes:
        element: Element symbol, or ``"avg"`` for the composition-average atom (alloys on one site).
        x: Fractional coordinate along a.
        y: Fractional coordinate along b.
        z: Fractional coordinate along c.
        occupancy: Site occupancy (default: 1).
    """

    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0


@dataclass(frozen=True)
class Reflection:
    """
    One family of symmetry-equivalent reflections.

    Attributes:
        hkl: Representative indices (fewest negative indices, then the largest h, k, l).
        d_A: d-spacing (Å).
        two_theta: 2theta at the wavelength used (deg), NaN beyond 180.
        multiplicity: Number of equivalent hkl, Friedel pairs included.
        f2: Neutron structure factor squared, ``|F|^2`` per cell (fm^2).
    """

    hkl: tuple[int, int, int]
    d_A: float
    two_theta: float
    multiplicity: int
    f2: float  # |F|^2 per cell, fm^2

    @property
    def label(self) -> str:
        """The hkl label (see :func:`hkl_label`)."""
        return hkl_label(self.hkl)


def hkl_label(hkl: tuple[int, int, int]) -> str:
    """
    Miller indices as a label.

    Args:
        hkl: Miller indices.

    Returns:
        Compact ``"311"`` when every index is a single non-negative digit, else ``"1 -1 0"``.
    """
    if all(0 <= i <= 9 for i in hkl):
        return "".join(str(i) for i in hkl)
    return " ".join(str(i) for i in hkl)


@dataclass(frozen=True, kw_only=True)
class Phase:
    """
    A crystal phase with its composition and density (for attenuation).

    Attributes:
        name: Display name.
        space_group: Hermann-Mauguin symbol (any form gemmi accepts, e.g. ``"Fm-3m"`` or ``"P 63/m m c"``).
        cell: a, b, c (Å), alpha, beta, gamma (deg).
        sites: Asymmetric-unit sites.
        composition: Element -> weight percent.
        density_g_cm3: Bulk density (g/cm3).
        source: Where the data came from (default: "").

    Raises:
        ValueError: If the cell is not six finite numbers with positive lengths and angles between 0 and 180 deg,
            the composition is empty or the density is not positive.

    Example:
        >>> al = Phase(name="Al", space_group="Fm-3m", cell=(4.0495, 4.0495, 4.0495, 90.0, 90.0, 90.0),
        ...            sites=(Site("Al", 0.0, 0.0, 0.0),), composition={"Al": 100.0}, density_g_cm3=2.70)
        >>> [r.label for r in al.reflections(1.54)][:3]
        ['111', '200', '220']
    """

    name: str
    space_group: str
    cell: tuple[float, float, float, float, float, float]
    sites: tuple[Site, ...]
    composition: dict[str, float]
    density_g_cm3: float
    source: str = ""

    def __post_init__(self) -> None:
        """Refuse a bad cell, or a phase without a composition or a positive density (attenuation needs both)."""
        cell = np.asarray(self.cell, float).reshape(-1)
        if cell.size != 6 or not np.isfinite(cell).all():
            raise ValueError(f"{self.name}: the cell must be six finite numbers (a, b, c, alpha, beta, gamma), "
                             f"got {self.cell!r}")  # fmt: skip
        if not (cell[:3] > 0).all():
            raise ValueError(f"{self.name}: the cell lengths must be positive (A), got {tuple(self.cell[:3])}")
        if not ((cell[3:] > 0) & (cell[3:] < 180)).all():
            raise ValueError(f"{self.name}: the cell angles must be between 0 and 180 deg, got {tuple(self.cell[3:])}")
        if not self.composition:
            raise ValueError(f"{self.name}: the composition is empty, got {self.composition!r}")
        if not (np.isfinite(self.density_g_cm3) and self.density_g_cm3 > 0):
            raise ValueError(f"{self.name}: the density must be positive (g/cm3), got {self.density_g_cm3!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Phase":
        """
        Build from a dict such as a preset JSON file.

        Args:
            d: Keys name, space_group, cell, composition and density_g_cm3, and optionally sites (a list of
                :class:`Site` fields, default one composition-average atom at the origin) and source. A cell of one
                value is a cubic a.

        Returns:
            The phase.

        Raises:
            ValueError: If a required key is missing, the cell does not have 1 or 6 values, a site has an unknown or
                missing key, the cell is invalid, the composition is empty or the density is not positive.
        """
        name = d.get("name", "?")
        missing = [k for k in ("name", "space_group", "cell", "composition", "density_g_cm3") if k not in d]
        if missing:
            raise ValueError(f"phase {name!r} lacks {', '.join(missing)}")
        cell = list(d["cell"])
        if len(cell) not in (1, 6):
            raise ValueError(f"phase {name!r}: the cell must have 1 value (cubic a) or 6, got {len(cell)}: {cell!r}")
        if len(cell) == 1:
            cell = [cell[0]] * 3 + [90.0] * 3
        site_dicts = d.get("sites", [{"element": AVERAGE, "x": 0, "y": 0, "z": 0}])
        known = {f.name for f in fields(Site)}
        required = known - {"occupancy"}
        for s in site_dicts:
            unknown, absent = sorted(set(s) - known), sorted(required - set(s))
            if unknown or absent:
                raise ValueError(
                    f"phase {name!r}: a site takes element, x, y, z and optionally occupancy, got {s!r} "
                    f"(unknown: {', '.join(unknown) or 'none'}, missing: {', '.join(absent) or 'none'})"
                )
        return cls(
            name=d["name"],
            space_group=d["space_group"],
            cell=tuple(float(c) for c in cell),  # type: ignore[arg-type]
            sites=tuple(Site(**s) for s in site_dicts),
            composition={k: float(v) for k, v in d["composition"].items()},
            density_g_cm3=float(d["density_g_cm3"]),
            source=d.get("source", ""),
        )

    @classmethod
    def from_cif(cls, path: str | Path) -> "Phase":
        """
        Read a small-molecule or inorganic CIF. Composition and density come from the cell contents.

        A density outside 0.5 to 25 g/cm3 gives a warning (check the cell and the occupancies).

        Args:
            path: The CIF file.

        Returns:
            The phase, with the space group in its CIF setting (extended H-M symbol).

        Raises:
            ValueError: If the file cannot be read or parsed as a CIF, holds more than one data block, has no space
                group gemmi recognizes, no unit cell, no atom sites, or atoms whose occupancies are all zero. The
                message names the file.
        """
        try:
            doc = gemmi.cif.read(str(path)).sole_block()
            st = gemmi.read_small_structure(str(path))
        except (RuntimeError, ValueError) as exc:  # gemmi reports parse errors and several blocks as RuntimeError
            raise ValueError(f"{path}: cannot read the CIF: {exc}") from exc
        # the space group with its setting (origin choice, hexagonal or rhombohedral axes) from the CIF's own
        # setting code, Hall symbol or symmetry operators: the plain H-M name alone picks the default setting
        sg = st.spacegroup
        if sg is None:
            raise ValueError(f"{path}: no space group that gemmi recognizes in the CIF ({st.spacegroup_hm!r})")
        if not all(doc.find_value(f"_cell_length_{k}") for k in "abc"):
            raise ValueError(f"{path}: the CIF has no unit cell (_cell_length_a, b, c)")
        sites = tuple(
            Site(s.element.name, float(s.fract.x), float(s.fract.y), float(s.fract.z), float(s.occ)) for s in st.sites
        )
        if not sites:
            raise ValueError(f"{path}: no atom sites in the CIF")
        c = st.cell
        cell = (c.a, c.b, c.c, c.alpha, c.beta, c.gamma)
        mass: dict[str, float] = {}
        for s in st.get_all_unit_cell_sites():
            el = s.element.name
            mass[el] = mass.get(el, 0.0) + float(s.occ) * atomic_mass(el)
        total = sum(mass.values())
        if not total > 0:
            raise ValueError(f"{path}: the atom occupancies in the CIF are all zero, so it has no composition")
        density = total / _NA / (c.volume * 1e-24)
        if not 0.5 <= density <= 25.0:
            warnings.warn(f"{path}: the cell contents give a density of {density:.3g} g/cm3, outside 0.5 to 25. "
                          "Check the cell and the occupancies.", stacklevel=2)  # fmt: skip
        name = st.name or Path(path).stem
        return cls(
            name=name,
            space_group=sg.xhm(),
            cell=cell,
            sites=sites,
            composition={el: 100 * m / total for el, m in mass.items()},
            density_g_cm3=density,
            source=f"CIF {Path(path).name}",
        )

    def with_changes(self, **kw: Any) -> "Phase":
        """
        Copy with some fields changed (the new phase is checked again).

        Args:
            **kw: Fields to change, e.g. ``density_g_cm3=8.2``.

        Returns:
            The new phase.

        Raises:
            ValueError: If the changed phase is invalid.
        """
        d = {f: getattr(self, f) for f in ("name", "space_group", "cell", "sites", "composition", "density_g_cm3")}
        d["source"] = self.source
        d.update(kw)
        return Phase(**d)

    def unit_cell(self) -> gemmi.UnitCell:
        """The cell as a gemmi unit cell."""
        return gemmi.UnitCell(*self.cell)

    def spacegroup(self) -> gemmi.SpaceGroup:
        """
        The space group as a gemmi space group.

        Raises:
            ValueError: If gemmi does not know the symbol.
        """
        sg = gemmi.find_spacegroup_by_name(self.space_group)
        if sg is None:
            raise ValueError(f"unknown space group '{self.space_group}'")
        return sg

    def d_spacing(self, hkl: tuple[int, int, int]) -> float:
        """
        d-spacing of a reflection.

        Args:
            hkl: Miller indices.

        Returns:
            d (Å).
        """
        return float(self.unit_cell().calculate_d(list(hkl)))

    def two_theta(self, hkl: tuple[int, int, int], wavelength_A: float) -> float:
        """
        Bragg angle 2theta of a reflection.

        Args:
            hkl: Miller indices.
            wavelength_A: Neutron wavelength (Å).

        Returns:
            2theta (deg), NaN when the reflection is beyond 2theta = 180.
        """
        s = wavelength_A / (2 * self.d_spacing(hkl))
        return float(2 * np.degrees(np.arcsin(s))) if s < 1 else float("nan")

    def _site_b(self) -> list[tuple[complex, float, list[list[float]]]]:
        """(b, occupancy, all positions in the cell) per site."""
        table = scattering_table()
        b_avg = 0j
        if any(s.element == AVERAGE for s in self.sites):
            if not self.composition:
                raise ValueError("an 'avg' site needs a composition")
            b_avg = sum(table[el].b_coh_fm * x for el, x in atomic_fractions(self.composition).items())
        ops = self.spacegroup().operations()
        out = []
        for s in self.sites:
            b = b_avg if s.element == AVERAGE else table[gemmi.Element(s.element).name].b_coh_fm
            if not _known(b):
                raise ValueError(f"no coherent scattering length for element '{s.element}' in the neutron table")
            pos: list[list[float]] = []
            for op in ops:
                p = np.mod(op.apply_to_xyz([s.x, s.y, s.z]), 1.0)
                if not any(np.allclose(np.mod(p - q + 0.5, 1.0) - 0.5, 0, atol=1e-4) for q in pos):
                    pos.append(list(p))
            out.append((b, s.occupancy, pos))
        return out

    def reflections(self, wavelength_A: float, *, two_theta_max: float = 179.0) -> list[Reflection]:
        """
        Allowed reflection families up to ``two_theta_max``, in order of decreasing d.

        A family is listed when it is not systematically absent and its neutron structure factor is above 1e-6 of
        the strongest possible.

        Args:
            wavelength_A: Neutron wavelength (Å).
            two_theta_max: Largest 2theta to list (deg, default: 179).

        Returns:
            The reflection families.

        Raises:
            ValueError: If the wavelength is not positive, the space group is unknown, a site element has no
                coherent scattering length, no site scatters, or the hkl search would go beyond ``MAX_HKL_INDEX``
                (a very large cell or a very short d-spacing limit).
        """
        if not (np.isfinite(wavelength_A) and wavelength_A > 0):
            raise ValueError(f"{self.name}: the wavelength must be positive (A), got {wavelength_A!r}")
        dmin = wavelength_A / (2 * np.sin(np.radians(two_theta_max / 2)))
        n = int(np.ceil(max(self.cell[:3]) / dmin)) + 1
        if n > MAX_HKL_INDEX:
            raise ValueError(f"{self.name}: listing reflections down to d {dmin:.3g} A in a cell of "
                             f"{max(self.cell[:3]):g} A needs hkl up to {n}, more than {MAX_HKL_INDEX}. Lower "
                             "two_theta_max or check the cell.")  # fmt: skip
        cell = self.unit_cell()
        sg = self.spacegroup()
        ops = sg.operations()
        rots = list(ops.sym_ops)
        sites = self._site_b()
        b_sum = sum(abs(b) * occ * len(p) for b, occ, p in sites)
        if not b_sum > 0:
            raise ValueError(f"{self.name}: the sites scatter nothing (all coherent scattering lengths are 0)")
        seen: set[tuple[int, int, int]] = set()
        out: list[Reflection] = []
        for h in range(-n, n + 1):
            for k in range(-n, n + 1):
                for l in range(-n, n + 1):  # noqa: E741
                    hkl = (h, k, l)
                    if hkl == (0, 0, 0) or hkl in seen:
                        continue
                    d = float(cell.calculate_d([h, k, l]))
                    if d < dmin - 1e-9:
                        continue
                    eq = {tuple(op.apply_to_hkl([h, k, l])) for op in rots}
                    eq |= {(-a, -b, -c) for a, b, c in eq}
                    seen |= eq  # type: ignore[arg-type]
                    if ops.is_systematically_absent([h, k, l]):
                        continue
                    hv = np.array(hkl, float)
                    F = sum(b * occ * np.exp(2j * np.pi * (np.asarray(ps) @ hv)).sum() for b, occ, ps in sites)
                    f2 = float(abs(F) ** 2)
                    if f2 < 1e-6 * b_sum**2:
                        continue
                    rep = min(eq, key=lambda v: (sum(i < 0 for i in v), -v[0], -v[1], -v[2]))
                    out.append(Reflection(rep, d, self.two_theta(rep, wavelength_A), len(eq), f2))  # type: ignore[arg-type]
        out.sort(key=lambda r: (-r.d_A, r.hkl))
        return out

    def mu_per_mm(self, wavelength_A: float) -> float:
        """
        Linear attenuation coefficient of the phase (see the module function :func:`mu_per_mm`).

        Args:
            wavelength_A: Neutron wavelength (Å).

        Returns:
            mu (1/mm).

        Raises:
            ValueError: If the wavelength is not positive or an element has no cross sections.
        """
        return mu_per_mm(self.composition, self.density_g_cm3, wavelength_A)


@cache
def presets() -> dict[str, Phase]:
    """
    Packaged preset phases (``data/materials/*.json``, cached).

    Returns:
        Phase name -> phase, in file-name order.
    """
    root = resources.files(DATA_PACKAGE).joinpath("materials")
    out = {}
    for f in sorted(root.iterdir(), key=lambda p: p.name):
        if f.name.endswith(".json"):
            p = Phase.from_dict(json.loads(f.read_text(encoding="utf-8")))
            out[p.name] = p
    return out
