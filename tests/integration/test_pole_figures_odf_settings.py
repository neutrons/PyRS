"""The ODF step's settings (core.mtex.settings) against the reference script's odf_settings.json: no pymtex needed."""

import json
from pathlib import Path

import pytest

from pyrs.pole_figures.core.mtex import MtexOptions, settings
from tests.util.pole_figures_helpers import REFERENCE

pytestmark = pytest.mark.integration


def test_settings_match_reference_script(tmp_path: Path) -> None:
    """The mtex block equals the reference script's odf_settings.json (plus the cell)."""
    # Arrange
    ref = json.loads((REFERENCE / "mtex" / "odf_settings.json").read_text(encoding="utf-8"))
    csvs = sorted(REFERENCE.glob("pole_figure_*.csv"))

    # Act
    s = settings(csvs, out_dir=tmp_path / "mtex", phase_name="IN718", cell=(3.597, 3.597, 3.597, 90, 90, 90),
                 view_from="+vz", vz_name="BD", opts=MtexOptions())  # fmt: skip

    # Assert
    assert [Path(f).name for f in s["csv_files"]] == [Path(f).name for f in ref["csv_files"]]
    assert (s["lattice_A"], s["phase_name"], s["view_from"], s["vz_label"]) == (3.597, "IN718", "+vz", "BD")
    assert s["cell"] == [3.597, 3.597, 3.597, 90, 90, 90]
    for k in ("crystal_symmetry", "specimen_symmetry", "intensity_column", "drop_ignored", "halfwidth_deg",
              "resolution_deg", "merge_chi90", "ipf_directions", "plot_view", "output_subfolder"):  # fmt: skip
        assert s["mtex"][k] == ref["mtex"][k], k
    for k in ("colormap", "contour_levels", "contour_lines", "odf_section", "pf_hkl", "cmax_odf", "cmax_pf"):
        assert s["mtex"]["plot"][k] == ref["mtex"]["plot"][k], k
    assert s["mtex"]["plot"]["plotkowski"]["hkl"] == ref["mtex"]["plot"]["plotkowski"]["hkl"]
