"""
Data and Corrections tab inputs. The incident slits and wavelength come from each reduced file.

The gauge position during the runs (vx, vy, vz in the STL frame) has no default. The user enters it here, so the
measured data are never corrected at a position that was only assumed (the Sample tab's center is for Predict).

Typical workflow:

1. The presenter fills the gauge position from the run logs, or the user types it.
2. The presenter reads :meth:`DataModel.gauge_center`, :attr:`DataModel.core_method` and
   :attr:`DataModel.core_scale` and passes them to the core pole-figure and combine functions.

Example:
    >>> model = DataModel(gauge_vx=29.4, gauge_vy=19.75, gauge_vz=0.0)
    >>> model.gauge_center()
    array([29.4 , 19.75,  0.  ])
    >>> model.core_method
    'window'
"""

from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from pyrs.pole_figures.core.correction import DEFAULT_CUTOFF
from pyrs.pole_figures.core.peaks import CENTRAL_HALF_WIDTH_DEG
from .predict import GUI_MC_POINTS

MethodName = Literal["Window (+-1.5 FWHM)", "Pseudo-Voigt fit"]
WINDOW: Final = "Window (+-1.5 FWHM)"
FIT: Final = "Pseudo-Voigt fit"
GAUGE_FIELDS = ["gauge_vx", "gauge_vy", "gauge_vz"]
SETTING_FIELDS = ["receiving_slit_mm", "cutoff", "mc_points", "central_hw", "method"]
COMBINE_FIELDS = ["combine", "combine_scale"]
ScaleName = Literal["None (parts of one scan)", "Median (detector regions)"]
NO_SCALE: Final = "None (parts of one scan)"
MEDIAN: Final = "Median (detector regions)"


class DataModel(BaseModel):
    """
    Inputs of the Data and Corrections tab.

    Defaults are PF's, except the gauge position (blank until entered) and the Monte Carlo points (the GUI's faster
    default).

    Attributes:
        gauge_vx: vx of the gauge center (rotation point) during the runs, mm in the STL (sample) frame, or None.
        gauge_vy: vy of the gauge center, mm, or None.
        gauge_vz: vz of the gauge center, mm, or None.
        receiving_slit_mm: Diffracted beam slit width (mm). It is not in the reduced file.
        cutoff: Points with a correction above this are ignored.
        mc_points: Monte Carlo gauge points per orientation.
        central_hw: Only peaks within this many degrees of the detector arm 2theta are used (deg).
        method: Intensity from PF's window integration (+-1.5 FWHM) or a pseudo-Voigt fit.
        combine: Merge runs that measured the same hkl into one pole figure.
        combine_scale: Scale between merged runs: none (parts of one scan) or median (detector regions).
    """

    gauge_vx: float | None = Field(
        default=None, ge=-10_000, le=10_000, title="vx (mm)",
        description="vx of the gauge center (rotation point) during the runs, in the STL frame. Filled from the "
                    "run logs (CUBOID frame), which match the STL only if CUBOID was set up to the STL's origin.",
    )  # fmt: skip
    gauge_vy: float | None = Field(
        default=None, ge=-10_000, le=10_000, title="vy (mm)",
        description="vy of the gauge center (rotation point) during the runs, in the STL frame. Filled from the "
                    "run logs (CUBOID frame), which match the STL only if CUBOID was set up to the STL's origin.",
    )  # fmt: skip
    gauge_vz: float | None = Field(
        default=None, ge=-10_000, le=10_000, title="vz (mm)",
        description="vz of the gauge center (rotation point) during the runs, in the STL frame. Filled from the "
                    "run logs (CUBOID frame), which match the STL only if CUBOID was set up to the STL's origin.",
    )  # fmt: skip

    receiving_slit_mm: float = Field(
        default=3.0, gt=0, le=50, title="Receiving slit (mm)",
        description="Diffracted beam slit width. Not stored in the reduced file.",
    )  # fmt: skip
    cutoff: float = Field(
        default=DEFAULT_CUTOFF, gt=1, le=1000, title="Ignore correction >",
        description="Points with a correction above this are ignored (long beam paths). Redraws without recomputing.",
    )  # fmt: skip
    mc_points: int = Field(
        default=GUI_MC_POINTS, ge=1000, le=1_000_000, title="Monte Carlo points",
        description="Gauge points per orientation. 40,000 matches PF's per-run corrections. 5,000 is close and "
                    "8 times faster.",
    )  # fmt: skip
    central_hw: float = Field(
        default=CENTRAL_HALF_WIDTH_DEG, gt=0, le=10, title="Peaks within (deg of arm)",
        description="Only peaks this close to the detector arm 2theta are used (edge pixels see another volume)",
    )  # fmt: skip
    method: MethodName = Field(default=WINDOW, title="Intensity", description="PF's window integration or a fit")

    combine: bool = Field(
        default=False, title="Combine runs with the same hkl",
        description="One pole figure split over several reduced files becomes one pole figure",
    )  # fmt: skip
    combine_scale: ScaleName = Field(
        default=NO_SCALE, title="Scale between runs",
        description="None for parts of one scan. Median: each run x median(first) / median(run) of corrected rates.",
    )  # fmt: skip

    def gauge_center(self) -> npt.NDArray[np.float64] | None:
        """The gauge position during the runs (mm, STL frame), or None until all three values are entered."""
        v = (self.gauge_vx, self.gauge_vy, self.gauge_vz)
        if any(x is None for x in v):
            return None
        return np.array(v, dtype=float)

    @property
    def core_scale(self) -> Literal["none", "median"]:
        """The combine scale as the core names it: "median" or "none"."""
        return "median" if self.combine_scale == MEDIAN else "none"

    @property
    def core_method(self) -> Literal["window", "fit"]:
        """The intensity method as the core names it: "fit" or "window"."""
        return "fit" if self.method == FIT else "window"
