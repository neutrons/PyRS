"""Files written from core results: table-scan workbooks, CSVs, figure images and beam-path GIFs."""

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest
from matplotlib.figure import Figure
from PIL import Image, ImageChops

from pyrs.pole_figures.core.correction import Beams
from pyrs.pole_figures.core.geometry import MountConventions
from pyrs.pole_figures.core.grid import Grid
from pyrs.pole_figures.core.prediction import Plane, Prediction, predict
from pyrs.pole_figures.core.table_scan import plan_scan
from pyrs.pole_figures.io import predicted_csv
from pyrs.pole_figures.io.predicted_csv import COLUMNS as PREDICTED_COLUMNS
from pyrs.pole_figures.io.predicted_csv import write_predicted_csv
from pyrs.pole_figures.io.table_scan import COLUMNS, write_table_scans
from pyrs.pole_figures.plots.beam_gif import GifSettings, measured_beam_gif, predicted_beam_gif
from pyrs.pole_figures.plots.save import save_figure
from tests.util.pole_figures_helpers import (
    REFERENCE,
    SLAB_CENTER_MM,
    read_csv_rows,
    slab_mesh,
    slab_prediction,
    synthetic_pole_figure,
)

pytestmark = pytest.mark.integration

CONV = MountConventions(incident_from="+vy")


@pytest.fixture(scope="module")
def pred() -> Prediction:
    return slab_prediction()


# ---------------------------------------------------------------- table-scan workbooks
def test_write_table_scans_xlsx_round_trip(pred: Prediction, tmp_path: Path) -> None:
    # Arrange
    scans = [plan_scan(q, cutoff=3.0, base_s=10.0) for q in pred.planes]

    # Act
    paths = write_table_scans(tmp_path, scans, prefix="IN718 plate", phase_name="IN718")

    # Assert
    assert [p.name for p in paths] == ["IN718_plate_200.xlsx", "IN718_plate_222.xlsx"]
    ws = openpyxl.load_workbook(paths[0]).active
    assert ws is not None
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    assert tuple(rows[0]) == COLUMNS
    assert len(rows) == 2 + len(scans[0].points)
    assert sum(float(r[7]) for r in rows[2:]) == pytest.approx(scans[0].total_s)  # type: ignore[arg-type]


def test_write_table_scans_run_row_round_trip(pred: Prediction, tmp_path: Path) -> None:
    # Arrange
    sc = plan_scan(pred.planes[1], cutoff=3.0, base_s=10.0)

    # Act
    (path,) = write_table_scans(tmp_path, [sc], prefix="IN718 plate", phase_name="IN718")

    # Assert: row 2 sets the run title, phase label, omega and 2theta, and nothing else
    ws = openpyxl.load_workbook(path).active
    assert ws is not None
    row2 = [c.value for c in ws[2]]
    assert row2[1:4] == ["IN718 222", 56.0, 112.0] and row2[4:] == [None] * 4
    assert str(row2[0]).startswith("IN718 plate | {222} pole figure | 2theta 112 omega 56 | ")


def test_write_table_scans_skips_empty_plane(pred: Prediction, tmp_path: Path) -> None:
    # Arrange: a cutoff below every correction keeps no point of {200}
    scans = [plan_scan(pred.planes[0], cutoff=0.1, base_s=10.0), plan_scan(pred.planes[1], cutoff=3.0, base_s=10.0)]
    assert not scans[0].points and scans[1].points

    # Act
    paths = write_table_scans(tmp_path, scans, prefix="IN718 plate", phase_name="IN718")

    # Assert
    assert [p.name for p in paths] == ["IN718_plate_222.xlsx"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["IN718_plate_222.xlsx"]


# ---------------------------------------------------------------- predicted CSV
def test_write_predicted_csv_one_row_per_point(pred: Prediction, tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "predicted_correction.csv"

    # Act
    write_predicted_csv(path, pred, cutoff=3.0)

    # Assert
    rows = read_csv_rows(path)
    assert len(rows) == sum(len(q.chi) for q in pred.planes)
    q = pred.planes[0]
    ignored = [int(r["ignored"]) for r in rows[: len(q.chi)]]
    assert ignored == [int(c > 3.0) for c in q.correction]


def test_write_predicted_csv_header_matches_reference(pred: Prediction, tmp_path: Path) -> None:
    # Arrange: the reference script's header, with its plate volume named for any sample
    with open(REFERENCE / "predicted_correction.csv", newline="", encoding="utf-8") as fh:
        reference = next(csv.reader(fh))
    want = ["sample_volume_in_gauge_mm3" if c == "plate_volume_in_gauge_mm3" else c for c in reference]
    path = tmp_path / "predicted_correction.csv"

    # Act
    write_predicted_csv(path, pred, cutoff=3.0)

    # Assert
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == want == PREDICTED_COLUMNS


def test_write_predicted_csv_logged_chi(tmp_path: Path) -> None:
    # Arrange: a goniometer whose chi zero is 5 deg off the cradle's
    pred = slab_prediction(chi_offset_deg=5.0)
    path = tmp_path / "predicted_correction.csv"

    # Act
    write_predicted_csv(path, pred, cutoff=3.0)

    # Assert
    rows = read_csv_rows(path)
    q = pred.planes[0]
    np.testing.assert_array_equal([float(r["chi"]) for r in rows[: len(q.chi)]], q.chi - 5.0)


def test_write_predicted_csv_failed_write_keeps_old_file(
    pred: Prediction, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: an older file, then a writer that stops after the header
    path = tmp_path / "predicted_correction.csv"
    path.write_text("old", encoding="utf-8")

    class StopAfterHeader:
        def __init__(self, fh: Any) -> None:
            self.fh = fh

        def writerows(self, rows: list[list[object]]) -> None:
            self.fh.write(",".join(map(str, rows[0])) + "\n")
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(predicted_csv.csv, "writer", StopAfterHeader)

    # Act / Assert
    with pytest.raises(OSError, match="No space"):
        write_predicted_csv(path, pred, cutoff=3.0)
    assert path.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


# ---------------------------------------------------------------- figure images
def test_save_figure_writes_png(tmp_path: Path) -> None:
    # Arrange
    fig = Figure(figsize=(2, 1))
    fig.add_subplot().plot([0, 1], [0, 1])
    path = tmp_path / "line.png"

    # Act
    save_figure(fig, path, dpi=50)

    # Assert
    with Image.open(path) as img:
        assert img.size == (100, 50)
    assert [p.name for p in tmp_path.iterdir()] == ["line.png"]


def test_save_figure_failed_save_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: savefig writes part of the image, then fails
    fig = Figure(figsize=(2, 1))

    def broken(self: Figure, fname: Any, **kw: Any) -> None:
        Path(fname).write_bytes(b"\x89PNG partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Figure, "savefig", broken)

    # Act / Assert
    with pytest.raises(OSError, match="No space"):
        save_figure(fig, tmp_path / "line.pdf")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- beam-path GIFs
@pytest.mark.timeout(120)
def test_beam_path_gif_parallel_frames_equal_serial(tmp_path: Path) -> None:
    # Arrange: a 30 x 30 deg grid (37 orientations), drawn in one process and in two
    pred = predict(slab_mesh(), center=SLAB_CENTER_MM, conv=CONV, planes=[Plane("200", 57.0, 28.5)], beams=Beams(),
                   mu=0.14, grid=Grid(30, 30), cutoff=3.0, n_points=300)  # fmt: skip
    q = pred.planes[0]
    a, b = tmp_path / "serial.gif", tmp_path / "parallel.gif"

    # Act
    n_serial = predicted_beam_gif(a, slab_mesh(), plane=q, center=SLAB_CENTER_MM, beams=Beams(), cutoff=3.0,
                                  settings=GifSettings(dpi=40, workers=1))  # fmt: skip
    predicted_beam_gif(b, slab_mesh(), plane=q, center=SLAB_CENTER_MM, beams=Beams(), cutoff=3.0,
                       settings=GifSettings(dpi=40, workers=2))  # fmt: skip

    # Assert
    with Image.open(a) as ia, Image.open(b) as ib:
        frames_a, frames_b = getattr(ia, "n_frames", 1), getattr(ib, "n_frames", 1)  # GifImageFile only
        assert n_serial == frames_a == frames_b == 37
        for k in range(frames_a):
            ia.seek(k)
            ib.seek(k)
            assert ImageChops.difference(ia.convert("RGB"), ib.convert("RGB")).getbbox() is None


@pytest.mark.parametrize("every", [1, 4])
def test_measured_beam_gif_one_frame_per_nth_point(every: int, tmp_path: Path) -> None:
    # Arrange: a measured pole figure of 37 points
    peak = synthetic_pole_figure()
    path = tmp_path / "measured.gif"

    # Act
    n = measured_beam_gif(path, slab_mesh(), peak=peak, center=SLAB_CENTER_MM,
                          settings=GifSettings(dpi=20, every=every, workers=1))  # fmt: skip

    # Assert
    with Image.open(path) as img:
        assert n == getattr(img, "n_frames", 1) == math.ceil(peak.n_points / every)


def test_measured_beam_gif_failed_save_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the GIF writer stops part way
    def broken(self: Image.Image, fp: Any, *args: Any, **kw: Any) -> None:
        Path(fp).write_bytes(b"GIF89a partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Image.Image, "save", broken)
    path = tmp_path / "measured.gif"

    # Act / Assert
    with pytest.raises(OSError, match="No space"):
        measured_beam_gif(path, slab_mesh(), peak=synthetic_pole_figure(), center=SLAB_CENTER_MM,
                          settings=GifSettings(dpi=20, every=12, workers=1))  # fmt: skip
    assert list(tmp_path.iterdir()) == []
