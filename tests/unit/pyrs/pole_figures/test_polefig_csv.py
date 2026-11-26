"""File names of the per-run pole-figure CSVs."""

from dataclasses import replace

from pyrs.pole_figures.io.polefig_csv import csv_name
from tests.util.pole_figures_helpers import synthetic_pole_figure


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
