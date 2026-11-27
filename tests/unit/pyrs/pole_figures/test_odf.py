"""
The ODF step's helpers that do not need MTEX for Python (pymtex).

CSV reading, grids, ranges, progress, and the behavior without pymtex. The fit itself against the stored MTEX 7.2.0
reference values is in ``tests/integration/test_pole_figures_odf_reference.py``.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pyrs.pole_figures.core.mtex import FROM_DATA, MtexOptions
from pyrs.pole_figures.core.odf import (
    STAGE_KEYS,
    all_hkl,
    compute,
    equal_area,
    hemisphere_grid,
    no_stage,
    pymtex_installed,
    range_max,
    read_pole_figure,
    stage_progress,
    view_axes,
)
from pyrs.pole_figures.io.odf import OdfCancelledError, run_odf
from tests.util.pole_figures_helpers import synthetic_pole_figure

CELL = (3.597, 3.597, 3.597, 90.0, 90.0, 90.0)
HEADER = "chi,pole_vx,pole_vy,pole_vz,corrected_per_s,correction,ignored"


def _write_pf(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_range_max_from_data_rounds_up_half() -> None:
    # Arrange
    data_max = 2.27

    # Act
    top = range_max(FROM_DATA, data_max)

    # Assert
    assert top == 2.5


def test_range_max_user_value_kept() -> None:
    # Arrange
    data_max = 2.27

    # Act
    top = range_max(4.0, data_max)

    # Assert
    assert top == 4.0


def test_equal_area_rim_is_one() -> None:
    # Arrange
    polar = [0.0, 90.0]

    # Act
    r = equal_area(polar)

    # Assert
    np.testing.assert_allclose(r, [0.0, 1.0])


def test_stage_progress_increases_through_steps() -> None:
    # Arrange
    keys = STAGE_KEYS

    # Act
    fracs = [stage_progress(k)[0] for k in keys]
    f0, left0 = stage_progress("start")

    # Assert
    assert f0 == 0.0 and 5 < left0 < 30
    assert fracs == sorted(fracs) and fracs[-1] < 1.0


def test_stage_progress_long_step_stays_below_next() -> None:
    # Arrange
    f_odf, _ = stage_progress("odf")

    # Act
    f_late, _ = stage_progress("odf", seconds_in_stage=1000.0)

    # Assert
    assert f_odf < f_late < stage_progress("errors")[0]


def test_read_pole_figure_drops_ignored_rows(tmp_path: Path) -> None:
    # Arrange
    src = _write_pf(tmp_path / "pole_figure_12_111.csv", ["0,0,0,1,5,1,0", "30,0.5,0,0.866,4,2,0", "60,1,0,0,3,9,1"])

    # Act
    kept = read_pole_figure(src, MtexOptions())
    raw = read_pole_figure(src, MtexOptions(drop_ignored=False))

    # Assert
    assert (kept.run, kept.hkl, kept.label) == ("12", "111", "{111} run 12")
    assert len(kept.intensity) == 2 and len(raw.intensity) == 3 and kept.n_in_csv == 3


def test_read_pole_figure_nan_intensity_dropped(tmp_path: Path) -> None:
    # Arrange
    src = _write_pf(
        tmp_path / "pole_figure_12_111.csv", ["0,0,0,1,nan,1,0", "30,0.5,0,0.866,4,nan,0", "60,1,0,0,3,1,0"]
    )

    # Act
    m = read_pole_figure(src, MtexOptions())

    # Assert
    np.testing.assert_allclose(m.intensity, [3.0])


def test_read_pole_figure_merge_chi90_one_point(tmp_path: Path) -> None:
    # Arrange
    src = _write_pf(tmp_path / "pole_figure_9_200.csv", ["0,0,0,1,5,1,0", "90,1,0,0,2,1,0", "90,1,0,0,4,1,0",
                                                          "90,1,0,0,6,1,0"])  # fmt: skip

    # Act
    merged = read_pole_figure(src, MtexOptions(merge_chi90=True))
    plain = read_pole_figure(src, MtexOptions())

    # Assert
    np.testing.assert_allclose(merged.intensity, [5.0, 4.0])
    np.testing.assert_allclose(merged.pole, [[0, 0, 1], [1, 0, 0]])
    assert len(plain.intensity) == 4


def test_read_pole_figure_bad_name_raises(tmp_path: Path) -> None:
    # Arrange
    bad = tmp_path / "pf.csv"
    bad.write_text("a\n1\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="pf.csv: the name must be pole_figure_<run>_<hkl>"):
        read_pole_figure(bad, MtexOptions())


def test_read_pole_figure_no_column_raises(tmp_path: Path) -> None:
    # Arrange
    src = _write_pf(tmp_path / "pole_figure_12_111.csv", ["0,0,0,1,5,1,0"])

    # Act / Assert
    with pytest.raises(ValueError, match="has no column raw_net_per_s"):
        read_pole_figure(src, MtexOptions(intensity_column="raw_net_per_s"))


def test_view_axes_vz_and_vx_right_handed() -> None:
    # Arrange
    views = ("+vz", "-vz", "+vx")

    # Act
    axes = {v: view_axes(v) for v in views}

    # Assert
    np.testing.assert_allclose(axes["+vz"], [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(axes["-vz"], axes["+vz"])
    np.testing.assert_allclose(axes["+vx"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_view_axes_unknown_view_raises() -> None:
    # Arrange
    view = "+vy"

    # Act / Assert
    with pytest.raises(ValueError, match="vy"):
        view_axes(view)


def test_hemisphere_grid_unit_poles_center_is_view() -> None:
    # Arrange
    axes = view_axes("+vz")

    # Act
    po, az, poles = hemisphere_grid(30.0, axes)

    # Assert
    assert len(po) == 4 * 13 and po.max() == 90.0 and az.max() == 360.0
    np.testing.assert_allclose(np.linalg.norm(poles, axis=1), 1.0)
    np.testing.assert_allclose(poles[po == 0], [[0, 0, 1]] * 13, atol=1e-12)


def test_all_hkl_no_repeats_in_order(tmp_path: Path) -> None:
    # Arrange
    src = _write_pf(tmp_path / "pole_figure_12_311.csv", ["0,0,0,1,5,1,0"])
    measured = [read_pole_figure(src, MtexOptions())]

    # Act
    hkls = all_hkl(MtexOptions(pf_hkl=["111", "200"], plotkowski_hkl=["100", "111"]), measured)

    # Assert
    assert hkls == ["111", "200", "311", "100"]


def test_pymtex_installed_follows_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    # Act
    found = pymtex_installed()

    # Assert
    assert found is False


def test_compute_without_pymtex_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange
    monkeypatch.setitem(sys.modules, "mtex", None)  # as if pymtex were not installed
    csvs = [_write_pf(tmp_path / f"pole_figure_{k}_111.csv", ["0,0,0,1,5,1,0"]) for k in (1, 2)]

    # Act / Assert
    with pytest.raises(ModuleNotFoundError, match="mtex"):
        compute(csvs, cell=CELL, phase_name="x", view_from="+vz", opts=MtexOptions(), stage=no_stage)


def test_run_odf_should_stop_cancels_before_fit(tmp_path: Path) -> None:
    # Arrange
    pfs = [synthetic_pole_figure("1001", "200"), synthetic_pole_figure("1002", "111", seed=1)]
    seen: list[str] = []

    # Act / Assert
    with pytest.raises(OdfCancelledError, match="cancelled before the step 'start'"):
        run_odf(tmp_path, pfs, phase_name="x", cell=CELL, view_from="+vz", vz_name="BD", opts=MtexOptions(),
                on_stage=seen.append, should_stop=lambda: True)  # fmt: skip
    assert seen == [] and (tmp_path / "mtex" / "odf_settings.json").is_file()
    assert sorted(p.name for p in tmp_path.glob("pole_figure_*.csv")) == ["pole_figure_1001_200.csv",
                                                                          "pole_figure_1002_111.csv"]  # fmt: skip
