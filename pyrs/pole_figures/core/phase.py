"""
Crystal phases: presets, CIF import, allowed reflections and 2theta at a wavelength.

- Allowed reflections: every hkl that is not systematically absent in the space group and whose neutron structure
  factor (coherent scattering lengths, all sites in the cell) is not zero. Symmetry-equivalent hkl are grouped into
  one family with its multiplicity (Friedel pairs included).
- Attenuation of a phase: :meth:`Phase.mu_per_mm`, from its composition and density (``neutron_xs.mu_per_mm``).

Typical workflow:

1. Take a preset with :func:`presets`, read a CIF with :meth:`Phase.from_cif`, or build a :class:`Phase`.
2. List the reflections at the wavelength with :meth:`Phase.reflections` and keep those inside the HB-2B 2theta range
   (``hb2b.two_theta_allowed``).
3. Get the attenuation with :meth:`Phase.mu_per_mm`.

Example:
    >>> fe = Phase(name="Fe", space_group="Im-3m", cell=(2.8665, 2.8665, 2.8665, 90.0, 90.0, 90.0),
    ...            sites=(Site(element="Fe", x=0.0, y=0.0, z=0.0),), composition={"Fe": 100.0}, density_g_cm3=7.874)
    >>> [r.label for r in fe.reflections(1.54)][:3]
    ['110', '200', '211']
    >>> round(fe.two_theta((2, 1, 1), 1.54), 2)
    82.29
    >>> round(fe.mu_per_mm(1.54), 4)
    0.1173
"""

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gemmi
import numpy as np

from .neutron_xs import AVOGADRO, DATA_PACKAGE, atomic_fractions, atomic_mass, mu_per_mm, scattering_table
from .reflections import Reflection, ScatteringSite, bragg_two_theta, hkl_search_limit, reflection_families

AVERAGE = "avg"  # site element meaning "the composition-weighted average atom" (alloys on one site)


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
class Phase:
    """
    A crystal phase with its composition and density (for attenuation).

    Attributes:
        name: Display name.
        space_group: Hermann-Mauguin symbol (any form gemmi accepts, e.g. ``"Fm-3m"`` or ``"P 63/m m c"``).
        cell: a, b, c (Å), alpha, beta, gamma (deg).
        sites: Asymmetric-unit sites.
        composition: Element -> weight percent, kept as a read-only mapping (a copy of the one given).
        density_g_cm3: Bulk density (g/cm3).
        source: Where the data came from (default: "").

    Raises:
        ValueError: If the cell is not six finite numbers with positive lengths and angles between 0 and 180 deg,
            the composition is empty or the density is not positive.

    Example:
        >>> al = Phase(name="Al", space_group="Fm-3m", cell=(4.0495, 4.0495, 4.0495, 90.0, 90.0, 90.0),
        ...            sites=(Site(element="Al", x=0.0, y=0.0, z=0.0),), composition={"Al": 100.0}, density_g_cm3=2.70)
        >>> [r.label for r in al.reflections(1.54)][:3]
        ['111', '200', '220']
    """

    name: str
    space_group: str
    cell: tuple[float, float, float, float, float, float]
    sites: tuple[Site, ...]
    composition: Mapping[str, float]
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
        object.__setattr__(self, "composition", MappingProxyType(dict(self.composition)))
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
            Site(
                element=s.element.name,
                x=float(s.fract.x),
                y=float(s.fract.y),
                z=float(s.fract.z),
                occupancy=float(s.occ),
            )
            for s in st.sites
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
        density = total / AVOGADRO / (c.volume * 1e-24)
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
        return bragg_two_theta(self.unit_cell(), hkl, wavelength_A)

    def _site_b(self) -> list[ScatteringSite]:
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
            if not np.isfinite(b):
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
                coherent scattering length, no site scatters, or the hkl search would go beyond
                ``reflections.MAX_HKL_INDEX`` (a very large cell or a very short d-spacing limit).
        """
        dmin, n = hkl_search_limit(name=self.name, cell_lengths=self.cell[:3], wavelength_A=wavelength_A,
                                   two_theta_max=two_theta_max)  # fmt: skip
        ops = self.spacegroup().operations()
        return reflection_families(name=self.name, cell=self.unit_cell(), ops=ops, sites=self._site_b(), dmin=dmin,
                                   n=n, wavelength_A=wavelength_A)  # fmt: skip

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


def presets() -> Mapping[str, Phase]:
    """
    Packaged preset phases (``data/materials/*.json``, read once).

    Returns:
        Phase name -> phase, in file-name order, read-only (the phases are frozen too), so a caller cannot change
        the presets every other caller sees.
    """
    return MappingProxyType(_read_presets())


@cache
def _read_presets() -> dict[str, Phase]:
    root = resources.files(DATA_PACKAGE).joinpath("materials")
    out = {}
    for f in sorted(root.iterdir(), key=lambda p: p.name):
        if f.name.endswith(".json"):
            p = Phase.from_dict(json.loads(f.read_text(encoding="utf-8")))
            out[p.name] = p
    return out
