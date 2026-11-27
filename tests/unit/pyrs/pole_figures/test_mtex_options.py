"""The ODF step's settings and checks (core.mtex): no pymtex needed."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrs.pole_figures.core.mtex import (
    FROM_DATA,
    MtexOptions,
    chi_rings,
    laue_group,
    parse_hkl_list,
    plot_data_files,
    plot_files,
    settings,
    usable,
    vz_label,
)
from tests.util.pole_figures_helpers import synthetic_pole_figure

CELL = (3.597, 3.597, 3.597, 90.0, 90.0, 90.0)


def test_chi_rings_full_grid_usable() -> None:
    # Arrange
    r = synthetic_pole_figure()

    # Act
    rings = chi_rings(r)

    # Assert
    assert rings >= 2 and usable(r)


def test_chi_rings_one_ring_kept_not_usable() -> None:
    # Arrange
    r = synthetic_pole_figure()
    one = replace(r, gauge=replace(r.gauge, correction=np.where(np.abs(r.chi) < 1e-9, 1.0, 99.0)))  # chi 0 kept only

    # Act
    rings = chi_rings(one)

    # Assert
    assert rings == 1 and not usable(one)


@pytest.mark.parametrize(
    ("space_group", "laue"),
    [("Fm-3m", "m-3m"), ("P63/mmc", "6/mmm"), ("R-3m", "-3m1"), ("P-31m", "-31m"), ("P-3m1", "-3m1"),
     ("Ia-3", "m-3"), ("C2/c", "2/m")],
)  # fmt: skip
def test_laue_group_mtex_names(space_group: str, laue: str) -> None:
    # Arrange
    sg = space_group

    # Act
    got = laue_group(sg)

    # Assert
    assert got == laue


def test_vz_label_names() -> None:
    # Arrange
    names = ("BD", "vz", "+vz", "hoop")

    # Act
    labels = [vz_label(n) for n in names]

    # Assert
    assert labels == ["BD", "+vz", "+vz", "+vz (hoop)"]


def test_parse_hkl_list_commas_and_spaces() -> None:
    # Arrange
    text = "111, 200 220"

    # Act
    hkls = parse_hkl_list(text)

    # Assert
    assert hkls == ["111", "200", "220"]


def test_parse_hkl_list_two_digits_raises() -> None:
    # Arrange
    text = "11"

    # Act / Assert
    with pytest.raises(ValueError, match="'11' must be three digits"):
        parse_hkl_list(text)


def test_block_from_data_written_as_data() -> None:
    # Arrange
    opts = MtexOptions(cmax_pf=3.0)

    # Act
    plot = opts.block()["plot"]

    # Assert
    assert opts.cmax_odf == FROM_DATA and plot["cmax_odf"] == "data" and plot["cmax_pf"] == 3.0
    assert plot["plotkowski"]["max"] == "data" and plot["white_background"] == {"below_mrd": 1.0}


def test_block_no_white_plots_false() -> None:
    # Arrange
    opts = MtexOptions(white_below_mrd=None)

    # Act
    plot = opts.block()["plot"]

    # Assert
    assert plot["white_background"] is False


def test_settings_one_pole_figure_raises(tmp_path: Path) -> None:
    # Arrange
    csvs = [tmp_path / "pole_figure_1_111.csv"]

    # Act / Assert
    with pytest.raises(ValueError, match="at least two pole figures, got 1"):
        settings(csvs, out_dir=tmp_path, phase_name="x", cell=CELL, view_from="+vz", vz_name="BD",
                 opts=MtexOptions())  # fmt: skip


def test_settings_no_ipf_direction_raises(tmp_path: Path) -> None:
    # Arrange
    csvs = [tmp_path / "pole_figure_1_111.csv", tmp_path / "pole_figure_2_200.csv"]

    # Act / Assert
    with pytest.raises(ValueError, match="inverse pole figure direction, got \\[\\]"):
        settings(csvs, out_dir=tmp_path, phase_name="x", cell=CELL, view_from="+vz", vz_name="BD",
                 opts=MtexOptions(ipf_directions=[]))  # fmt: skip


def test_plot_files_only_existing_in_order(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "odf_phi2_45.png").write_bytes(b"x")
    (tmp_path / "inverse_pole_figures.png").write_bytes(b"x")

    # Act
    files = plot_files(tmp_path)

    # Assert
    assert [n for n, _, _ in files] == ["Square", "Inverse pole figures"]
    assert all(w is None for _, _, w in files)


def test_plot_files_white_version_and_phi2(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "odf_phi2_30.png").write_bytes(b"x")
    (tmp_path / "odf_phi2_30_white.png").write_bytes(b"x")

    # Act
    files = plot_files(tmp_path, phi2_deg=30.0)

    # Assert
    assert files == [("Square", tmp_path / "odf_phi2_30.png", tmp_path / "odf_phi2_30_white.png")]


def test_plot_data_files_csvs_then_readme(tmp_path: Path) -> None:
    # Arrange
    d = tmp_path / "plot_data"
    d.mkdir()
    for name in ("README.txt", "b.csv", "a.csv"):
        (d / name).write_text("x", encoding="utf-8")

    # Act
    files = plot_data_files(tmp_path)

    # Assert
    assert [p.name for p in files] == ["a.csv", "b.csv", "README.txt"]
