"""
Predict tab inputs: the chi/phi grid and the Monte Carlo sampling.

The predict presenter builds the core grid with :meth:`PredictModel.grid` and passes ``mc_points`` and ``best_phi``
to the prediction.

Example:
    >>> grid = PredictModel(chi_step=15, phi_step=10).grid()
"""

from pydantic import BaseModel, Field

from pyrs.pole_figures.core.grid import Grid

#: the GUI's default, 4 to 8 times faster than 20,000. On a thin, strongly absorbing IN718 slab (the test fixture) the
#: predicted correction at 5,000 points differs from that at 20,000 by 0.5 % (median), 1.5 % (95th percentile) and
#: 2.8 % at most; 3 of 2596 points cross the 3.0 cutoff. On the 50 mm Al cylinder 500 points were
#: within 0.04 %. 40,000 stays the core default for parity with the reference script.
GUI_MC_POINTS = 5_000

GRID_FIELDS = ["chi_step", "phi_step", "equal_spacing"]
MC_FIELDS = ["mc_points", "best_phi"]


class PredictModel(BaseModel):
    """
    Inputs of the Predict tab. Defaults: 10 x 5 deg grid and the GUI's Monte Carlo points (PF uses 20,000).

    Attributes:
        chi_step: chi step (deg). chi runs from 0 to below 90, plus 90.
        phi_step: phi step (deg). phi runs from 0 to below 360.
        equal_spacing: Coarser phi steps as chi grows, so points stay about ``phi_step`` apart on the sphere.
        mc_points: Monte Carlo gauge points per orientation.
        best_phi: Where phi is free (chi 90, phi axis along Q), try phi 0 to 360 in steps of ``phi_step`` and keep
            the shortest mean beam path (default: on; off keeps phi 0, as PF).
    """

    chi_step: float = Field(
        default=10.0, ge=1, le=45, title="chi step (deg)", description="chi 0 to below 90, plus 90"
    )
    phi_step: float = Field(default=5.0, ge=1, le=90, title="phi step (deg)", description="phi 0 to below 360")
    equal_spacing: bool = Field(
        default=False, title="Equal spacing on the sphere",
        description="Step phi more coarsely as chi grows, so the points stay about phi step apart on the sphere "
                    "(each ring takes a whole-degree step that divides 360). 10 x 5 deg: 454 points instead of 649.",
    )  # fmt: skip
    mc_points: int = Field(
        default=GUI_MC_POINTS,
        ge=500,
        le=1_000_000,
        title="Monte Carlo points",
        description="Gauge points per orientation. 20,000 matches PF's prediction. 5,000 is close and 4 times "
        "faster. Fewer is faster and noisier.",
    )

    best_phi: bool = Field(
        default=True,
        title="Free phi: shortest path",
        description="At chi 90 (phi axis along Q) every phi gives the same pole. Try phi in the grid step and keep "
        "the one with the shortest beam path. Off: phi 0, as PF.",
    )

    def grid(self) -> Grid:
        """The core chi/phi grid for these steps."""
        return Grid(self.chi_step, self.phi_step, equal_spacing=self.equal_spacing)
