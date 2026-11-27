"""
Experiment setup tab inputs: phase, wavelength, beams and slits, attenuation and the reflections to use.

The material presenter fills the phase fields from a preset or a CIF file, computes mu from the composition and
density unless it is overridden, and lists the reflections. The prediction reads :meth:`MaterialModel.beams`.

Example:
    >>> MaterialModel().beams()
    Beams(incident_width=4.0, incident_height=4.0, diffracted_width=3.0)
"""

from typing import Literal

from pydantic import BaseModel, Field

from pyrs.pole_figures.core.correction import Beams
from pyrs.pole_figures.core.hb2b import HB2B_WAVELENGTHS

PHASE_FIELDS = ["name", "space_group", "a", "b", "c", "alpha", "beta", "gamma", "composition", "density_g_cm3"]
WAVELENGTH_FIELDS = ["wavelength_setting", "wavelength_A"]
MU_FIELDS = ["mu_override", "mu_value"]
BEAM_FIELDS = ["incident_width", "incident_height", "diffracted_width"]

WavelengthSetting = Literal[
    "Si333 (1.452 A)",
    "Si511 (1.452 A)",
    "Si422 (1.54 A)",
    "Si331 (1.731 A)",
    "Si400 (1.886 A)",
    "Si311 (2.275 A)",
    "Si220 (2.667 A)",
    "Custom",
]
CUSTOM = "Custom"
#: the monochromator dropdown labels -> nominal wavelength (Å), built from ``core.hb2b.HB2B_WAVELENGTHS``
SETTING_WAVELENGTHS: dict[str, float] = {f"{name} ({lam:g} A)": lam for name, lam in HB2B_WAVELENGTHS.items()}
SELECT = "Select material..."  # nothing chosen yet


class MaterialModel(BaseModel):
    """
    Inputs of the Experiment setup tab.

    No material is chosen at start (preset = SELECT). The phase fields hold IN718, the starting point for Custom. The
    other defaults are PF's reference case: the Si331 wavelength, 4 x 4 mm incident slits (ISR:X/Y:Gap) and a 3 mm
    receiving slit.

    Attributes:
        preset: Preset material name, CUSTOM, or SELECT (nothing chosen yet).
        cif_path: CIF file the phase was read from, or "".
        name: Phase name shown in the figures.
        space_group: Hermann-Mauguin symbol, for example Fm-3m.
        a: Lattice parameter a (Å).
        b: Lattice parameter b (Å).
        c: Lattice parameter c (Å).
        alpha: Lattice angle alpha (deg).
        beta: Lattice angle beta (deg).
        gamma: Lattice angle gamma (deg).
        composition: "Element:weight percent" pairs, comma separated, normalized to 100.
        density_g_cm3: Density (g/cm3).
        wavelength_setting: Monochromator setting, or CUSTOM.
        wavelength_A: Wavelength (Å).
        incident_width: Horizontal incident slit, ISR:X:Gap (mm).
        incident_height: Vertical incident slit, ISR:Y:Gap (mm).
        diffracted_width: Receiving slit width (mm).
        mu_override: Use ``mu_value`` instead of mu from the composition.
        mu_value: Linear attenuation coefficient used when overridden (1/mm).
        selected_hkl: Reflections to use, for example ["111", "200"].
    """

    preset: str = Field(default=SELECT, title="Preset", description="A preset, Custom, or not chosen yet")
    cif_path: str = Field(default="", title="CIF file")

    name: str = Field(default="IN718 (fcc)", title="Name")
    space_group: str = Field(default="Fm-3m", title="Space group", description="Hermann-Mauguin, e.g. Fm-3m, P63/mmc")
    a: float = Field(default=3.597, gt=0, le=100, title="a (A)")
    b: float = Field(default=3.597, gt=0, le=100, title="b (A)")
    c: float = Field(default=3.597, gt=0, le=100, title="c (A)")
    alpha: float = Field(default=90.0, gt=0, lt=180, title="alpha (deg)")
    beta: float = Field(default=90.0, gt=0, lt=180, title="beta (deg)")
    gamma: float = Field(default=90.0, gt=0, lt=180, title="gamma (deg)")
    composition: str = Field(
        default="Ni:52.5, Cr:19, Fe:18.5, Nb:5.1, Mo:3.05, Ti:0.9, Al:0.5",
        title="Composition (wt%)",
        description="Element:weight percent, comma separated. Normalized to 100.",
    )
    density_g_cm3: float = Field(default=8.19, gt=0, le=30, title="Density (g/cm3)")

    wavelength_setting: WavelengthSetting = Field(default="Si331 (1.731 A)", title="Monochromator")
    wavelength_A: float = Field(
        default=1.731,
        gt=0.5,
        lt=5,
        title="Wavelength (A)",
        description="Use the calibrated value of the runs if known",
    )

    incident_width: float = Field(
        default=4.0, gt=0, le=50, title="Incident width (mm)", description="Horizontal incident slit, ISR:X:Gap"
    )
    incident_height: float = Field(
        default=4.0, gt=0, le=50, title="Incident height (mm)", description="Vertical incident slit, ISR:Y:Gap"
    )
    diffracted_width: float = Field(
        default=3.0, gt=0, le=50, title="Receiving slit width (mm)", description="Diffracted beam slit width"
    )

    mu_override: bool = Field(default=False, title="Override mu", description="Use a measured or other mu value")
    mu_value: float = Field(default=0.14, ge=0, le=10, title="mu (1/mm)")

    selected_hkl: list[str] = Field(default_factory=list, title="Reflections to use")

    def beams(self) -> Beams:
        """The incident and receiving slit sizes (mm) as core :class:`Beams`."""
        return Beams(
            incident_width=self.incident_width,
            incident_height=self.incident_height,
            diffracted_width=self.diffracted_width,
        )
