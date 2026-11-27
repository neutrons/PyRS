"""
ODF with MTEX for Python (pymtex) against the stored reference values of the reference data set (4 pole figures).

The reference values (``mtex/odf_summary.json`` and ``mtex/odf_5deg.txt`` beside the pole-figure CSVs) were made once
with MTEX 7.2.0. pymtex solves the same ill-posed inversion with the same algorithm. They agree on the fit (RP errors
to 3 decimals) and on the ODF to a few hundredths of an mrd. The peak height is the least certain number: it moves by
about 0.1 mrd with the solver's iteration limits alone. Tolerances below are set from that, not to the last digit.

The tests that use the fit need pymtex (an optional dependency) and are skipped where it is not installed.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyrs.pole_figures.core.mtex import OUTPUT_PLOTS, MtexOptions, plot_data_files, plot_files
from pyrs.pole_figures.core.odf import OdfResult, compute, read_pole_figure
from pyrs.pole_figures.io import odf as io_odf
from tests.util.pole_figures_helpers import REFERENCE

pytestmark = pytest.mark.integration

CELL = (3.597, 3.597, 3.597, 90.0, 90.0, 90.0)


@pytest.fixture(scope="module")
def result() -> OdfResult:
    """The ODF of the four reference pole figures (about 10 s, plus numba compiling on a first run)."""
    # in the fixture, not at module level: a module-level skip would also show in tiers that deselect this module
    pytest.importorskip("mtex", reason="MTEX for Python (pymtex) is an optional dependency, installed separately")
    stages: list[str] = []
    res = compute(sorted(REFERENCE.glob("pole_figure_*.csv")), cell=CELL, phase_name="IN718", view_from="+vz",
                  opts=MtexOptions(), stage=stages.append)  # fmt: skip
    assert stages == ["start", "read", "odf", "errors", "grids"]
    return res


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    """The stored reference summary (MTEX 7.2.0)."""
    return dict(json.loads((REFERENCE / "mtex" / "odf_summary.json").read_text(encoding="utf-8")))


def test_compute_points_and_fit_match_reference(result: OdfResult, ref: dict[str, Any]) -> None:
    # Arrange
    measured = result.measured

    # Act
    labels = [m.label for m in measured]
    used = [len(m.intensity) for m in measured]

    # Assert
    assert labels == ref["pole_figures"] and used == ref["points_used"]
    assert [m.n_in_csv for m in measured] == ref["points_in_csv"]
    np.testing.assert_allclose(result.rp_error, ref["rp_error"], atol=2e-3)


def test_compute_texture_numbers_match_reference(result: OdfResult, ref: dict[str, Any]) -> None:
    # Arrange
    corners = ref["ipf_corner_mrd"]

    # Act
    ranges = {k: getattr(result, k) for k in ("cmax_odf", "cmax_pf", "cmax_ipf")}

    # Assert
    assert result.texture_index == pytest.approx(ref["texture_index_norm2"], rel=0.01)
    assert result.entropy == pytest.approx(ref["entropy"], abs=0.01)
    assert result.odf_max == pytest.approx(ref["odf_max_mrd"], abs=0.15)
    assert ranges == {k: ref[k] for k in ranges}
    for d in ("vx", "vy", "vz"):
        for c in ("c001", "c011", "c111"):
            assert result.ipf_corner[d][c] == pytest.approx(corners[d][c], abs=0.06), (d, c)


def test_compute_odf_grid_matches_reference(result: OdfResult) -> None:
    # Arrange
    stored = np.loadtxt(REFERENCE / "mtex" / "odf_5deg.txt", comments="%")

    # Act
    diff = result.odf_table[:, 3] - stored[:, 3]

    # Assert
    assert result.odf_table.shape == stored.shape  # the Bunge grid: 72 x 19 x 18 points for m-3m at 5 deg
    np.testing.assert_allclose(result.odf_table[:, :3], stored[:, :3])
    assert np.sqrt(np.mean(diff**2)) < 0.05 and np.corrcoef(result.odf_table[:, 3], stored[:, 3])[0, 1] > 0.995


def test_compute_measured_scaled_to_the_odf(result: OdfResult) -> None:
    # Arrange
    pairs = zip(result.measured_scaled, result.measured_recalc, strict=True)

    # Act
    ratios = [float(np.median(s / r)) for s, r in pairs]

    # Assert
    assert result.measured_in_mrd and all(0.9 < q < 1.1 for q in ratios)


def test_compute_grids_in_the_view(result: OdfResult) -> None:
    # Arrange
    g = result.pf_csv["111"]

    # Act
    p1, PP, f = result.section_plot
    xyz, fv = result.ipf["vz"]

    # Assert
    assert list(result.pf_plot) == ["111", "200", "220", "222", "311", "100", "110"]
    assert len(g.f) == 37 * 145 and g.polar.max() == 90.0 and g.azimuth.max() == 360.0
    np.testing.assert_allclose(g.pole[g.polar == 0], [[0, 0, 1]] * 145, atol=1e-12)  # seen from +vz
    assert f.shape == (len(PP), len(p1)) and np.nanmax(f) <= result.cmax_odf
    assert len(xyz) == len(fv) > 500 and np.all(xyz[:, 2] > 0.5)  # the 001-101-111 sector


def test_read_pole_figure_reference_csv_drops_ignored() -> None:
    # Arrange
    src = sorted(REFERENCE.glob("pole_figure_*.csv"))[0]

    # Act
    kept = read_pole_figure(src, MtexOptions())
    raw = read_pole_figure(src, MtexOptions(drop_ignored=False))

    # Assert
    assert (kept.hkl, kept.n_in_csv) == ("222", 649) and len(raw.intensity) > len(kept.intensity)


def test_writers_write_every_output(result: OdfResult, tmp_path: Path) -> None:
    """Every plot, the plot-data CSVs, the ODF table and the summary, with the reference script's names and columns."""
    # Arrange
    opts = MtexOptions()

    # Act
    io_odf.write_plots(result, tmp_path, opts=opts, vzl="BD")
    io_odf.write_plot_data(result, tmp_path, opts=opts, vzl="BD", view_x=False)
    io_odf.write_odf_table(result, tmp_path / "odf_5deg.txt", crystal_symmetry="m-3m", specimen_symmetry="1")
    io_odf.write_summary(result, tmp_path, opts=opts, vzl="BD")

    # Assert
    files = plot_files(tmp_path)
    assert [n for n, _, _ in files] == [n for n, _, _ in OUTPUT_PLOTS]
    assert [w is not None for _, _, w in files] == [False, True, False, True, False]
    names = sorted(p.name for p in plot_data_files(tmp_path))
    assert names == ["README.txt", "color_scales.csv", "colormap_WhiteJetColorMap.csv", "inverse_pole_figures.csv",
                     "odf_section_phi2_45.csv", "plotkowski_bands.csv", "pole_figures_measured.csv",
                     "pole_figures_recalculated.csv"]  # fmt: skip
    head = (tmp_path / "plot_data" / "pole_figures_measured.csv").read_text(encoding="utf-8").splitlines()[0]
    assert head == ("pole_figure,pole_vx,pole_vy,pole_vz,polar_deg,azimuth_deg,x_plot,y_plot,measured,"
                    "recalculated_mrd,measured_in_mrd,RP_error")  # fmt: skip
    np.testing.assert_allclose(np.loadtxt(tmp_path / "odf_5deg.txt", comments="%"), result.odf_table, atol=1e-5)
    s = json.loads((tmp_path / "odf_summary.json").read_text(encoding="utf-8"))
    assert s["points_used"] == [610, 613, 567, 549] and s["mtex_version"].startswith("pymtex-")
    assert "RP error 0.106" in (tmp_path / "odf_summary.txt").read_text(encoding="utf-8")
