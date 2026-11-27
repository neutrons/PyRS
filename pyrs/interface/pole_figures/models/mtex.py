"""
MTEX tab inputs (the reference script's ``mtex`` config block). Defaults are its settings.

The MTEX presenter resolves the crystal symmetry (AUTO takes the Laue class from the space group) and calls
:meth:`MtexModel.options` to get the core :class:`MtexOptions` for the ODF fit and plots.

Example:
    >>> opts = MtexModel().options("m-3m")
    >>> opts.pf_hkl
    ['111', '200', '220']
"""

from typing import Literal

from pydantic import BaseModel, Field

from pyrs.pole_figures.core.mtex import FROM_DATA, ColorMax, MtexOptions, parse_hkl_list

ODF_FIELDS = ["crystal_symmetry", "specimen_symmetry", "halfwidth_deg", "resolution_deg", "intensity_column",
              "drop_ignored", "merge_chi90"]  # fmt: skip
AUTO: Literal["Auto (from space group)"] = "Auto (from space group)"
#: Auto, then the Laue classes in MTEX notation (trigonal -3m in both settings)
CrystalSymmetry = Literal["Auto (from space group)", "m-3m", "m-3", "6/mmm", "6/m", "-3m1", "-31m", "-3", "4/mmm",
                          "4/m", "mmm", "2/m", "-1"]  # fmt: skip
PLOT_FIELDS = ["pf_hkl", "ipf_vx", "ipf_vy", "ipf_vz", "cmax_odf", "cmax_pf", "cmax_ipf", "contour_levels",
               "phi2_deg", "white_background", "white_below_mrd", "plotkowski", "plotkowski_hkl",
               "plotkowski_max", "export_data"]  # fmt: skip


class MtexModel(BaseModel):
    """
    Inputs of the MTEX tab. A color range of 0 means 'from the data' (rounded up to 0.5 mrd).

    Attributes:
        crystal_symmetry: Laue class in MTEX notation, or AUTO (from the material's space group).
        specimen_symmetry: Symmetry imposed on the sample frame: "1" (none), "-1", "2/m" or "mmm".
        halfwidth_deg: de la Vallee Poussin kernel halfwidth (deg).
        resolution_deg: ODF grid resolution (deg).
        intensity_column: Pole-figure column fitted: "corrected_per_s" or "raw_net_per_s" (a check).
        drop_ignored: Leave out rows flagged as ignored (correction above the cutoff).
        merge_chi90: Average the repeated +vx pole at chi 90 to one point.
        pf_hkl: Recalculated pole figures, three-digit hkl, comma or space separated.
        ipf_vx: Plot the inverse pole figure along +vx.
        ipf_vy: Plot the inverse pole figure along +vy.
        ipf_vz: Plot the inverse pole figure along +vz.
        cmax_odf: Color maximum of the ODF section (mrd), 0 for the data's.
        cmax_pf: Color maximum of the pole figures (mrd), 0 for the data's.
        cmax_ipf: Color maximum of the inverse pole figures (mrd), 0 for the data's.
        contour_levels: Number of contour levels.
        phi2_deg: phi2 of the ODF section (deg).
        white_background: Also make versions with values below ``white_below_mrd`` in white.
        white_below_mrd: Level below which the white-background versions are white (mrd).
        plotkowski: Also plot the colorful recalculated pole figures (eight log-spaced bands).
        plotkowski_hkl: hkl of the colorful pole figures.
        plotkowski_max: Color maximum of the colorful pole figures (mrd), 0 for the data's.
        export_data: Write the numbers behind each plot to CSV.
    """

    crystal_symmetry: CrystalSymmetry = Field(
        default=AUTO, title="Crystal symmetry",
        description="Laue class of the crystal (MTEX). Auto takes it from the material's space group, e.g. Fm-3m -> "
                    "m-3m. Cubic m-3m, m-3; hexagonal 6/mmm, 6/m; trigonal -3m1, -31m, -3; tetragonal 4/mmm, 4/m; "
                    "orthorhombic mmm; monoclinic 2/m; triclinic -1",
    )  # fmt: skip
    specimen_symmetry: Literal["1", "-1", "2/m", "mmm"] = Field(
        default="1", title="Specimen symmetry",
        description="Symmetry imposed on the sample frame. 1 = none (full pole figures, as Peterson et al. 2021). "
                    "mmm = orthorhombic, e.g. a rolled sheet. 2/m = one mirror. -1 = inversion only",
    )  # fmt: skip
    halfwidth_deg: float = Field(default=5.0, ge=1, le=20, title="Kernel halfwidth (deg)",
                                 description="de la Vallee Poussin kernel (Peterson 2021: 5)")  # fmt: skip
    resolution_deg: float = Field(default=5.0, ge=1, le=20, title="Resolution (deg)")
    intensity_column: Literal["corrected_per_s", "raw_net_per_s"] = Field(
        default="corrected_per_s", title="Intensity", description="raw_net_per_s only as a check"
    )
    drop_ignored: bool = Field(default=True, title="Drop ignored points", description="Rows with ignored = 1")
    merge_chi90: bool = Field(
        default=False,
        title="Merge the chi 90 ring",
        description="Average the repeated +vx pole at chi 90 to one point",
    )

    pf_hkl: str = Field(
        default="111, 200, 220", title="Recalculated PFs (hkl)", description="Recalculated - simple: three digits each"
    )
    ipf_vx: bool = Field(default=True, title="IPF along +vx")
    ipf_vy: bool = Field(default=True, title="IPF along +vy")
    ipf_vz: bool = Field(default=True, title="IPF along +vz")
    cmax_odf: float = Field(default=0.0, ge=0, le=1000, title="Square max (mrd, 0 = data)")
    cmax_pf: float = Field(default=0.0, ge=0, le=1000, title="Pole figure max (mrd, 0 = data)")
    cmax_ipf: float = Field(default=0.0, ge=0, le=1000, title="IPF max (mrd, 0 = data)")
    contour_levels: int = Field(default=20, ge=4, le=100, title="Contour levels")
    phi2_deg: float = Field(default=45.0, ge=0, le=90, title="Square: ODF section phi2 (deg)")
    white_background: bool = Field(default=True, title="Make white-below-random versions",
                                   description="Also plot with values below the level in white")  # fmt: skip
    white_below_mrd: float = Field(default=1.0, gt=0, le=10, title="White below (mrd)")
    plotkowski: bool = Field(
        default=True, title="Recalculated - colorful", description="Eight log-spaced color bands, Plotkowski 2021"
    )
    plotkowski_hkl: str = Field(default="100, 110, 111", title="Colorful hkl")
    plotkowski_max: float = Field(default=0.0, ge=0, le=1000, title="Colorful max (0 = data)")
    export_data: bool = Field(default=True, title="Export plot data (CSV)",
                              description="Write the numbers behind each plot to mtex/plot_data/*.csv (with a README) "
                                          "so the plots can be redrawn in other plotting code")  # fmt: skip

    def options(self, crystal_symmetry: str) -> MtexOptions:
        """
        Core options for the ODF fit and plots. Color maxima of 0 become ``FROM_DATA``.

        Args:
            crystal_symmetry: The resolved Laue class (AUTO already replaced by the caller).

        Returns:
            The options for ``core.mtex``.

        Raises:
            ValueError: If no inverse pole figure direction is chosen, or an hkl list entry is not three digits.
        """
        dirs = [d for d, on in (("vx", self.ipf_vx), ("vy", self.ipf_vy), ("vz", self.ipf_vz)) if on]
        if not dirs:
            raise ValueError("choose at least one inverse pole figure direction")

        def rng(v: float) -> ColorMax:
            return FROM_DATA if v <= 0 else v

        return MtexOptions(
            crystal_symmetry=crystal_symmetry, specimen_symmetry=self.specimen_symmetry,
            intensity_column=self.intensity_column, drop_ignored=self.drop_ignored,
            halfwidth_deg=self.halfwidth_deg, resolution_deg=self.resolution_deg, merge_chi90=self.merge_chi90,
            ipf_directions=dirs, pf_hkl=parse_hkl_list(self.pf_hkl), cmax_odf=rng(self.cmax_odf),
            cmax_pf=rng(self.cmax_pf), cmax_ipf=rng(self.cmax_ipf), contour_levels=self.contour_levels,
            phi2_deg=self.phi2_deg, white_below_mrd=self.white_below_mrd if self.white_background else None,
            plotkowski=self.plotkowski, plotkowski_hkl=parse_hkl_list(self.plotkowski_hkl),
            plotkowski_max=rng(self.plotkowski_max), export_data=self.export_data,
        )  # fmt: skip
