"""The per-run pole-figure CSVs: file names, labels, columns and values, and failed writes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.polefig import PolePeak
from pyrs.pole_figures.io import polefig_csv
from pyrs.pole_figures.io.polefig_csv import COLUMNS, csv_name, write_all, write_pole_figure_csv
from tests.util.pole_figures_helpers import read_csv_rows, synthetic_pole_figure

#: the reference script's header (pole_figure_<run>_<hkl>.csv), which MTEX scripts read
REFERENCE_HEADER = ["run", "hkl", "chi", "phi", "pole_vx", "pole_vy", "pole_vz", "count_s", "raw_net_per_s",
                    "raw_sigma_per_s", "F_mm3", "plate_volume_in_gauge_mm3", "mean_path_mm", "sampled_vy_offset_mm",
                    "sampled_vy_rms_mm", "correction", "corrected_per_s", "corrected_sigma_per_s",
                    "ignored"]  # fmt: skip


def _offset_pole_figure() -> PolePeak:
    """The synthetic pole figure on a mount with a 5 deg chi offset, and no correction at point 3 (NaN)."""
    r = synthetic_pole_figure("2001", "200")
    conv = MountConventions(incident_from="+vy", chi_offset_deg=5.0)
    corr = r.correction.copy()
    corr[3] = np.nan
    return replace(r, mount=Mount(conv, omega=r.omega, tth=r.two_theta), gauge=replace(r.gauge, correction=corr))


def test_csv_name_single_run() -> None:
    # Arrange
    r = synthetic_pole_figure("2001", "311")

    # Act
    name = csv_name(r)

    # Assert
    assert name == "pole_figure_2001_311.csv"


def test_csv_name_combined_takes_first_run() -> None:
    # Arrange: a combined pole figure (MTEX scripts read the run digits from the name)
    r = replace(synthetic_pole_figure("2001", "200"), run="2001+2002", source_runs=("2001", "2002"))

    # Act
    name = csv_name(r)

    # Assert
    assert name == "pole_figure_2001_200.csv"


def test_csv_name_path_in_hkl_stays_in_folder() -> None:
    # Arrange
    r = synthetic_pole_figure("2001", "../../../tmp/evil")

    # Act
    name = csv_name(r)

    # Assert
    assert name == "pole_figure_2001_tmp_evil.csv"


@pytest.mark.parametrize(("field", "value"), [("hkl", "../../../tmp/evil"), ("hkl", "=1+1"), ("run", "@SUM(A1)"),
                                              ("run", "2001|cmd")])  # fmt: skip
def test_write_all_bad_label_raises_before_writing(field: str, value: str, tmp_path: Path) -> None:
    # Arrange
    good = synthetic_pole_figure("2001", "200")
    other = synthetic_pole_figure("2002", "222")
    bad = replace(other, hkl=value) if field == "hkl" else replace(other, run=value)

    # Act / Assert
    with pytest.raises(ValueError, match=f"{field} label"):
        write_all(tmp_path / "work", [good, bad])
    assert not (tmp_path / "work").exists()


def test_write_pole_figure_csv_bad_source_run_raises(tmp_path: Path) -> None:
    # Arrange
    r = replace(synthetic_pole_figure("2001", "200"), source_runs=("2001", "=HYPERLINK()"))

    # Act / Assert
    with pytest.raises(ValueError, match="run label '=HYPERLINK\\(\\)'"):
        write_pole_figure_csv(tmp_path / "pf.csv", r)


def test_write_all_pole_figure_csv_columns(tmp_path: Path) -> None:
    # Arrange
    r = synthetic_pole_figure("2001", "200")

    # Act
    (path,) = write_all(tmp_path / "work", [r])

    # Assert: the reference script's header, one row per point
    rows = read_csv_rows(path)
    assert REFERENCE_HEADER == COLUMNS
    assert list(rows[0]) == REFERENCE_HEADER and len(rows) == r.n_points
    assert path.name == "pole_figure_2001_200.csv"


def test_write_pole_figure_csv_logged_chi_and_values(tmp_path: Path) -> None:
    # Arrange
    r = _offset_pole_figure()
    path = tmp_path / "pf.csv"

    # Act
    write_pole_figure_csv(path, r)

    # Assert
    rows = read_csv_rows(path)
    np.testing.assert_array_equal([float(x["chi"]) for x in rows], r.chi - 5.0)
    fin = np.isfinite(r.correction)
    want = np.round(r.rate * r.correction, 4)[fin]
    np.testing.assert_array_equal([float(x["corrected_per_s"]) for x, f in zip(rows, fin, strict=True) if f], want)
    np.testing.assert_array_equal([float(x["correction"]) for x, f in zip(rows, fin, strict=True) if f],
                                  np.round(r.correction[fin], 4))  # fmt: skip
    assert [int(x["ignored"]) for x in rows] == [int(v) for v in r.ignored]


def test_write_pole_figure_csv_no_correction_empty_cells(tmp_path: Path) -> None:
    # Arrange: point 3 has no correction (NaN)
    r = _offset_pole_figure()
    path = tmp_path / "pf.csv"

    # Act
    write_pole_figure_csv(path, r)

    # Assert
    row = read_csv_rows(path)[3]
    assert (row["correction"], row["corrected_per_s"], row["corrected_sigma_per_s"]) == ("", "", "")
    assert row["ignored"] == "1" and row["raw_net_per_s"] != ""


def test_write_pole_figure_csv_failed_write_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the writer stops after the header
    class StopAfterHeader:
        def __init__(self, fh: object) -> None:
            self.fh = fh

        def writerows(self, rows: list[list[object]]) -> None:
            self.fh.write(",".join(map(str, rows[0])) + "\n")  # type: ignore[attr-defined]
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(polefig_csv.csv, "writer", StopAfterHeader)
    path = tmp_path / "pf.csv"

    # Act / Assert
    with pytest.raises(OSError, match="No space"):
        write_pole_figure_csv(path, synthetic_pole_figure())
    assert list(tmp_path.iterdir()) == []
