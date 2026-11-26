"""Files written from core results: table-scan workbooks, CSVs, figure images and beam-path GIFs."""

import csv
from pathlib import Path

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
from pyrs.pole_figures.io.polefig_csv import COLUMNS as PF_COLUMNS
from pyrs.pole_figures.io.polefig_csv import write_all
from pyrs.pole_figures.io.table_scan import COLUMNS, write_table_scans
from pyrs.pole_figures.plots.beam_gif import GifSettings, beam_path_gif
from pyrs.pole_figures.plots.predicted import write_predicted_csv
from pyrs.pole_figures.plots.save import save_figure
from tests.util.pole_figures_helpers import REFERENCE, SLAB_CENTER_MM, read_csv_rows, slab_mesh, synthetic_pole_figure

pytestmark = pytest.mark.integration

CONV = MountConventions(incident_from="+vy")


@pytest.fixture(scope="module")
def pred() -> Prediction:
    return predict(slab_mesh(), center=SLAB_CENTER_MM, conv=CONV,
                   planes=[Plane.symmetric("200", 57.0), Plane("222", 112.0, 56.0)], beams=Beams(), mu=0.14,
                   grid=Grid(10, 15), n_points=2000)  # fmt: skip


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


def test_write_all_pole_figure_csv_columns(tmp_path: Path) -> None:
    # Arrange
    r = synthetic_pole_figure("2001", "200")

    # Act
    (path,) = write_all(tmp_path / "work", [r])

    # Assert: the reference script's header, one row per point
    with open(REFERENCE / "pole_figure_6494_222.csv", newline="", encoding="utf-8") as fh:
        reference_header = next(csv.reader(fh))
    rows = read_csv_rows(path)
    assert PF_COLUMNS == reference_header
    assert list(rows[0]) == reference_header and len(rows) == r.n_points
    assert path.name == "pole_figure_2001_200.csv"


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


@pytest.mark.timeout(120)
def test_beam_path_gif_parallel_frames_equal_serial(tmp_path: Path) -> None:
    # Arrange: a 30 x 30 deg grid (37 orientations), drawn in one process and in two
    pred = predict(slab_mesh(), center=SLAB_CENTER_MM, conv=CONV, planes=[Plane("200", 57.0, 28.5)], beams=Beams(),
                   mu=0.14, grid=Grid(30, 30), cutoff=3.0, n_points=300)  # fmt: skip
    q, center = pred.planes[0], np.array(SLAB_CENTER_MM)
    a, b = tmp_path / "serial.gif", tmp_path / "parallel.gif"

    # Act
    n_serial = beam_path_gif(a, slab_mesh(), center=center, q=q, beams=Beams(), cutoff=3.0,
                             settings=GifSettings(dpi=40, workers=1))  # fmt: skip
    beam_path_gif(b, slab_mesh(), center=center, q=q, beams=Beams(), cutoff=3.0,
                  settings=GifSettings(dpi=40, workers=2))  # fmt: skip

    # Assert
    with Image.open(a) as ia, Image.open(b) as ib:
        frames_a, frames_b = getattr(ia, "n_frames", 1), getattr(ib, "n_frames", 1)  # GifImageFile only
        assert n_serial == frames_a == frames_b == 37
        for k in range(frames_a):
            ia.seek(k)
            ib.seek(k)
            assert ImageChops.difference(ia.convert("RGB"), ib.convert("RGB")).getbbox() is None
